// Collects the expert statistics of HOPE (Higher-Order Pruning of Experts, arXiv 2609.18916) from a MoE model.
//
// For every MoE layer and every token, with the selected experts e_k, their final router weights g_k and the
// norms n_k = ||f_k(x)|| of their (unweighted) outputs:
//   norm_prod_sums [e_i][e_j] += g_i g_j n_i n_j   for all pairs (i, j) of selected experts, i == j included
//   coselect_counts[e_i][e_j] += 1
// plus the per-expert first-order sums used by REAP and the other baselines. The layout matches the HDF5 file
// of github.com/awslabs/HOPE (`hope calibrate`); ignite/quant/hope_obs.py converts this tool's output to it, so
// `hope solve` runs unchanged.
//
// Input (-f) is JSONL, one document per line: {"text": "...", "set": k, "src": "name"}. Text is tokenized with
// special tokens parsed (chat-template markers are single tokens). A document longer than -c tokens is split into
// pieces of at most -c tokens; pieces of one set are packed into batches of at most -b tokens, one sequence each.
// Statistics are accumulated per set (--n-sets, default 4, e.g. half A/B x original/generated), so the convergence
// check and the ablations need no second pass; the solver input is the sum of the chosen sets.
//
// Only the last token of each piece is an output; the MTP hidden-state output (unmasked) keeps the last layer's
// FFN running on all rows, so the logits buffer stays small.
//
// Check a GPU run against a CPU run (-dev none -ngl 0) on the same text: fused GPU paths must not hide the tensors.

#include "llama.h"

#include "arg.h"
#include "common.h"
#include "log.h"
#include "../src/llama-ext.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -f pool.jsonl -o hope-obs.bin -c 16384 -b 4096 -dev none -ngl 0 \\\n"
        "       [--n-sets 4] [--max-seqs 16] [--snapshot-every 500000] [--output-frequency 20] \\\n"
        "       [--in-file hope-obs.bin (resume)]\n", argv[0]);
    LOG("\n");
}

static const char   HOPE_MAGIC[8] = { 'H', 'O', 'P', 'E', 'O', 'B', 'S', '2' };

struct layer_stats {
    int64_t              total_tokens = 0;
    std::vector<double>  norm_prod_sums;   // E x E
    std::vector<int64_t> coselect_counts;  // E x E
    std::vector<double>  gate_sums, norm_sums, norm_square_sums, gate_norm_sums, gate_norm_square_sums; // E

    void init(int n_expert) {
        const size_t e = n_expert;
        norm_prod_sums.assign(e*e, 0.0);
        coselect_counts.assign(e*e, 0);
        for (auto * v : { &gate_sums, &norm_sums, &norm_square_sums, &gate_norm_sums, &gate_norm_square_sums }) {
            v->assign(e, 0.0);
        }
    }
};

struct hope_collector {
    int32_t n_layer  = 0;
    int32_t n_expert = 0;
    int32_t n_used   = 0;
    int64_t n_docs_done   = 0;   // documents fully processed, for resuming
    int64_t n_tokens_done = 0;
    int     cur_set  = 0;        // set the running batch accumulates into

    std::vector<std::vector<layer_stats>> sets; // [set][layer]
    std::map<std::string, int64_t> src_tokens;  // tokens per source, saved in the .json sidecar

    // final router weights of the current ubatch per layer, by priority of the name they were seen under
    struct weights_buf { int prio = -1; int64_t n_tokens = 0; std::vector<float> w; };
    std::map<int, weights_buf> weights;

    double  max_gate_sum_err = 0.0;
    int64_t n_obs = 0;
    std::mutex mtx;
    std::vector<uint8_t> tmp;

    void init(int32_t nl, int32_t ne, int32_t nu, int n_sets) {
        n_layer = nl; n_expert = ne; n_used = nu;
        sets.resize(n_sets);
        for (auto & s : sets) {
            s.assign(nl, layer_stats());
            for (auto & l : s) { l.init(ne); }
        }
    }

    // copies a tensor's bytes (host or device); the result is indexed with the tensor's strides
    const uint8_t * data_of(const ggml_tensor * t) {
        if (ggml_backend_buffer_is_host(t->buffer)) {
            return (const uint8_t *) t->data;
        }
        tmp.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, tmp.data(), 0, tmp.size());
        return tmp.data();
    }

    static bool parse(const char * name, const char * base, int & il) {
        const size_t n = strlen(base);
        if (strncmp(name, base, n) != 0 || name[n] != '-') { return false; }
        char * end = nullptr;
        il = (int) strtol(name + n + 1, &end, 10);
        return end != name + n + 1 && *end == '\0';
    }

    // the router weights as they are multiplied into the expert outputs: the last of these names in build_moe_ffn
    static int weights_prio(const char * name, int & il) {
        static const char * names[] = { "ffn_moe_weights", "ffn_moe_weights_softmax", "ffn_moe_weights_norm", "ffn_moe_weights_scaled" };
        for (int p = 3; p >= 0; --p) {
            if (parse(name, names[p], il)) { return p; }
        }
        return -1;
    }

    static bool is_down(const ggml_tensor * t, int & il) {
        return t->op == GGML_OP_MUL_MAT_ID && (parse(t->name, "ffn_moe_down", il) || parse(t->name, "ffn_moe_down_scaled", il));
    }

    bool collect(ggml_tensor * t, bool ask) {
        int il = -1;
        if (ask) {
            return weights_prio(t->name, il) >= 0 || is_down(t, il);
        }

        std::lock_guard<std::mutex> lock(mtx);

        const int prio = weights_prio(t->name, il);
        if (prio >= 0) {
            GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t));
            const int64_t nt = ggml_nelements(t) / n_used;
            GGML_ASSERT(nt * n_used == ggml_nelements(t));
            auto & wb = weights[il];
            if (prio >= wb.prio || wb.n_tokens != nt) {
                const float * w = (const float *) data_of(t);
                wb.prio = prio;
                wb.n_tokens = nt;
                wb.w.assign(w, w + ggml_nelements(t));
            }
            return true;
        }

        if (!is_down(t, il)) { return true; }
        GGML_ASSERT(il >= 0 && il < n_layer);

        // experts: [n_embd, n_used, n_tokens], ids (src[2]): [n_used, n_tokens] I32
        const ggml_tensor * ids = t->src[2];
        GGML_ASSERT(t->type == GGML_TYPE_F32 && ids && ids->type == GGML_TYPE_I32);
        GGML_ASSERT(t->ne[1] == n_used && ids->ne[0] == n_used && ids->ne[1] == t->ne[2]);
        const int64_t n_embd = t->ne[0];
        const int64_t nt     = t->ne[2];

        auto it = weights.find(il);
        GGML_ASSERT(it != weights.end() && it->second.n_tokens == nt && "router weights not seen before the expert outputs");
        const std::vector<float> w = std::move(it->second.w);
        weights.erase(it);

        std::vector<uint8_t> ids_copy;
        {
            const uint8_t * p = data_of(ids);
            ids_copy.assign(p, p + ggml_nbytes(ids));
        }
        const uint8_t * out = data_of(t);

        layer_stats & ls = sets[cur_set][il];
        ls.total_tokens += nt;
        const int64_t E = n_expert;
        std::vector<int32_t> e(n_used);
        std::vector<double>  g(n_used), nrm(n_used);

        for (int64_t i = 0; i < nt; ++i) {
            double gsum = 0.0;
            for (int k = 0; k < n_used; ++k) {
                e[k] = *(const int32_t *) (ids_copy.data() + k*ids->nb[0] + i*ids->nb[1]);
                GGML_ASSERT(e[k] >= 0 && e[k] < n_expert);
                g[k] = w[i*n_used + k];
                gsum += g[k];
                const float * f = (const float *) (out + k*t->nb[1] + i*t->nb[2]);
                double s = 0.0;
                for (int64_t d = 0; d < n_embd; ++d) { s += (double) f[d] * f[d]; }
                nrm[k] = sqrt(s);
            }
            max_gate_sum_err = std::max(max_gate_sum_err, fabs(gsum - 1.0));
            for (int a = 0; a < n_used; ++a) {
                const double ga = g[a], na = nrm[a];
                ls.gate_sums[e[a]]             += ga;
                ls.norm_sums[e[a]]             += na;
                ls.norm_square_sums[e[a]]      += na*na;
                ls.gate_norm_sums[e[a]]        += ga*na;
                ls.gate_norm_square_sums[e[a]] += ga*ga*na*na;
                for (int b = 0; b < n_used; ++b) {
                    ls.norm_prod_sums [e[a]*E + e[b]] += ga*na*g[b]*nrm[b];
                    ls.coselect_counts[e[a]*E + e[b]] += 1;
                }
            }
        }
        n_obs += nt;
        return true;
    }

    bool save(const std::string & path) const {
        const std::string tmp_path = path + ".tmp";
        FILE * f = fopen(tmp_path.c_str(), "wb");
        if (!f) { LOG_ERR("%s: cannot open %s\n", __func__, tmp_path.c_str()); return false; }
        const int32_t hdr[4] = { n_layer, n_expert, n_used, (int32_t) sets.size() };
        fwrite(HOPE_MAGIC, 1, 8, f);
        fwrite(hdr, sizeof(hdr), 1, f);
        fwrite(&n_docs_done, sizeof(n_docs_done), 1, f);
        fwrite(&n_tokens_done, sizeof(n_tokens_done), 1, f);
        for (const auto & s : sets) {
            for (const auto & l : s) {
                fwrite(&l.total_tokens, sizeof(int64_t), 1, f);
                fwrite(l.norm_prod_sums.data(),  sizeof(double),  l.norm_prod_sums.size(),  f);
                fwrite(l.coselect_counts.data(), sizeof(int64_t), l.coselect_counts.size(), f);
                for (const auto * v : { &l.gate_sums, &l.norm_sums, &l.norm_square_sums, &l.gate_norm_sums, &l.gate_norm_square_sums }) {
                    fwrite(v->data(), sizeof(double), v->size(), f);
                }
            }
        }
        const bool ok = !ferror(f);
        fclose(f);
        if (!ok) { LOG_ERR("%s: write error on %s\n", __func__, tmp_path.c_str()); return false; }
        std::remove(path.c_str());
        if (std::rename(tmp_path.c_str(), path.c_str()) != 0) { LOG_ERR("%s: cannot rename to %s\n", __func__, path.c_str()); return false; }

        nlohmann::json side = { { "docs_done", n_docs_done }, { "tokens_done", n_tokens_done }, { "src_tokens", src_tokens } };
        std::vector<int64_t> set_tokens;
        for (const auto & s : sets) { set_tokens.push_back(s.empty() ? 0 : s[0].total_tokens); }
        side["set_tokens"] = set_tokens;
        FILE * fj = fopen((path + ".json").c_str(), "wb");
        if (fj) { const std::string js = side.dump(1); fwrite(js.data(), 1, js.size(), fj); fclose(fj); }
        return true;
    }

    bool load(const std::string & path) {
        FILE * f = fopen(path.c_str(), "rb");
        if (!f) { LOG_ERR("%s: cannot open %s\n", __func__, path.c_str()); return false; }
        char magic[8];
        int32_t hdr[4];
        bool ok = fread(magic, 1, 8, f) == 8 && memcmp(magic, HOPE_MAGIC, 8) == 0 && fread(hdr, sizeof(hdr), 1, f) == 1;
        ok = ok && hdr[0] == n_layer && hdr[1] == n_expert && hdr[2] == n_used && hdr[3] == (int32_t) sets.size();
        ok = ok && fread(&n_docs_done, sizeof(n_docs_done), 1, f) == 1 && fread(&n_tokens_done, sizeof(n_tokens_done), 1, f) == 1;
        for (auto & s : sets) {
            for (auto & l : s) {
                if (!ok) { break; }
                ok = fread(&l.total_tokens, sizeof(int64_t), 1, f) == 1;
                ok = ok && fread(l.norm_prod_sums.data(),  sizeof(double),  l.norm_prod_sums.size(),  f) == l.norm_prod_sums.size();
                ok = ok && fread(l.coselect_counts.data(), sizeof(int64_t), l.coselect_counts.size(), f) == l.coselect_counts.size();
                for (auto * v : { &l.gate_sums, &l.norm_sums, &l.norm_square_sums, &l.gate_norm_sums, &l.gate_norm_square_sums }) {
                    ok = ok && fread(v->data(), sizeof(double), v->size(), f) == v->size();
                }
            }
        }
        fclose(f);
        if (!ok) { LOG_ERR("%s: %s is not a matching hope-collect file\n", __func__, path.c_str()); return false; }
        try {
            std::ifstream fj(path + ".json");
            const auto side = nlohmann::json::parse(fj);
            src_tokens = side.at("src_tokens").get<std::map<std::string, int64_t>>();
        } catch (const std::exception & e) {
            LOG_WRN("%s: no usable sidecar %s.json (%s), per-source counts restart\n", __func__, path.c_str(), e.what());
        }
        return true;
    }
};

static hope_collector g_collector;

// integer hyperparameter `<arch>.<key>` from the model metadata, -1 if missing
static int32_t model_arch_int(const llama_model * model, const char * key) {
    char arch[64], val[64];
    if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) < 0) { return -1; }
    const std::string full = std::string(arch) + "." + key;
    if (llama_model_meta_val_str(model, full.c_str(), val, sizeof(val)) < 0) { return -1; }
    return (int32_t) atoi(val);
}

static bool hope_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    GGML_UNUSED(user_data);
    return g_collector.collect(t, ask);
}

struct piece {
    int64_t doc;   // index of the document it belongs to
    bool    last;  // last piece of its document
    std::vector<llama_token> toks;
};

// decodes pieces of one set: all in one batch, or one long piece over several batches
static bool decode_pieces(llama_context * ctx, common_batch & batch, const std::vector<piece> & ps, int n_batch) {
    llama_memory_clear(llama_get_memory(ctx), true);
    if (ps.size() == 1 && (int) ps[0].toks.size() > n_batch) {
        const auto & t = ps[0].toks;
        for (size_t off = 0; off < t.size(); off += n_batch) {
            batch.clear();
            const size_t end = std::min(t.size(), off + n_batch);
            for (size_t k = off; k < end; ++k) { batch.add(t[k], (llama_pos) k, 0, k + 1 == t.size()); }
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) { return false; }
        }
        return true;
    }
    batch.clear();
    for (size_t s = 0; s < ps.size(); ++s) {
        const auto & t = ps[s].toks;
        for (size_t k = 0; k < t.size(); ++k) { batch.add(t[k], (llama_pos) k, (llama_seq_id) s, k + 1 == t.size()); }
    }
    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
}

static bool run(llama_context * ctx, const common_params & params, int32_t n_piece, int n_seq_max, int64_t snapshot_every) {
    const int n_sets  = (int) g_collector.sets.size();
    const int n_batch = params.n_batch;

    std::vector<std::string> lines;
    {
        size_t pos = 0;
        const std::string & in = params.prompt;
        while (pos < in.size()) {
            size_t nl = in.find('\n', pos);
            if (nl == std::string::npos) { nl = in.size(); }
            if (nl > pos) { lines.emplace_back(in, pos, nl - pos); }
            pos = nl + 1;
        }
    }
    const int64_t n_docs = (int64_t) lines.size();
    LOG_INF("%s: %lld documents, resuming after %lld; pieces <= %d tokens, batches <= %d tokens / %d sequences\n", __func__,
            (long long) n_docs, (long long) g_collector.n_docs_done, n_piece, n_batch, n_seq_max);

    common_batch batch(ctx);
    std::vector<std::deque<piece>> queue(n_sets);
    std::vector<int64_t> queued(n_sets, 0);
    std::map<int64_t, int> open_pieces;   // doc -> pieces not decoded yet
    int64_t next_snapshot = snapshot_every > 0 ? (g_collector.n_tokens_done / snapshot_every + 1) * snapshot_every : INT64_MAX;
    int64_t n_iter = 0, t_run_tokens = 0;
    const auto t_run = std::chrono::steady_clock::now();

    // decodes the front of queue[s]: as many whole pieces as fit in one batch
    auto flush_one = [&](int s) -> bool {
        std::vector<piece> ps;
        int64_t n = 0;
        while (!queue[s].empty() && (int) ps.size() < n_seq_max &&
               (ps.empty() || n + (int64_t) queue[s].front().toks.size() <= n_batch)) {
            n += queue[s].front().toks.size();
            ps.push_back(std::move(queue[s].front()));
            queue[s].pop_front();
            if (n >= n_batch) { break; }
        }
        queued[s] -= n;
        g_collector.cur_set = s;
        if (!decode_pieces(ctx, batch, ps, n_batch)) { LOG_ERR("%s: failed to eval\n", __func__); return false; }
        llama_synchronize(ctx);
        g_collector.n_tokens_done += n;
        t_run_tokens += n;
        for (const auto & p : ps) {
            if (--open_pieces[p.doc] == 0) { open_pieces.erase(p.doc); }
        }
        // documents before the oldest one still open are complete
        g_collector.n_docs_done = open_pieces.empty() ? g_collector.n_docs_done : open_pieces.begin()->first;
        if (++n_iter % 10 == 0) {
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_run).count();
            LOG_INF("%s: %lld tokens, doc %lld/%lld, %.1f tok/s, max |sum g - 1| = %.2e\n", __func__,
                    (long long) g_collector.n_tokens_done, (long long) g_collector.n_docs_done, (long long) n_docs,
                    t_run_tokens / secs, g_collector.max_gate_sum_err);
        }
        return true;
    };
    auto flush_all = [&]() -> bool {
        for (int s = 0; s < n_sets; ++s) {
            while (!queue[s].empty()) { if (!flush_one(s)) { return false; } }
        }
        return true;
    };
    auto save_points = [&](bool force) -> bool {
        const bool snap = g_collector.n_tokens_done >= next_snapshot;
        if (!snap && !force) { return true; }
        if (!flush_all()) { return false; }
        if (snap) {
            const std::string p = params.out_file + "." + std::to_string(g_collector.n_tokens_done) + ".bin";
            if (!g_collector.save(p)) { return false; }
            LOG_INF("%s: snapshot %s\n", __func__, p.c_str());
            while (next_snapshot <= g_collector.n_tokens_done) { next_snapshot += snapshot_every; }
        }
        return g_collector.save(params.out_file);
    };

    for (int64_t d = g_collector.n_docs_done; d < n_docs; ++d) {
        nlohmann::json doc;
        try {
            doc = nlohmann::json::parse(lines[d]);
        } catch (const std::exception & e) {
            LOG_ERR("%s: line %lld is not JSON: %s\n", __func__, (long long) d + 1, e.what());
            return false;
        }
        const std::string text = doc.value("text", "");
        const int set = doc.value("set", 0);
        const std::string src = doc.value("src", "");
        if (set < 0 || set >= n_sets) {
            LOG_ERR("%s: line %lld has set %d, --n-sets is %d\n", __func__, (long long) d + 1, set, n_sets);
            return false;
        }
        const std::vector<llama_token> toks = common_tokenize(ctx, text, false, /*parse_special*/ true);
        if (toks.empty()) { continue; }
        g_collector.src_tokens[src] += toks.size();

        int np = 0;
        for (size_t off = 0; off < toks.size(); off += n_piece, ++np) {
            const size_t end = std::min(toks.size(), off + n_piece);
            queue[set].push_back({ d, end == toks.size(), std::vector<llama_token>(toks.begin() + off, toks.begin() + end) });
            queued[set] += end - off;
        }
        open_pieces[d] = np;

        while (queued[set] >= n_batch) {
            if (!flush_one(set)) { return false; }
        }
        if (!save_points(false)) { return false; }
        if (params.n_out_freq > 0 && n_iter >= params.n_out_freq) {
            n_iter = 0;
            if (!save_points(true)) { return false; }
        }
    }
    if (!flush_all()) { return false; }
    g_collector.n_docs_done = n_docs;
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.out_file = "hope-obs.bin";
    params.n_ctx    = 2048;
    params.escape   = false;

    common_init();

    // own options, removed before the common parser sees them
    int     n_sets         = 4;
    int     n_seq_max      = 16;
    int64_t snapshot_every = 0;
    std::vector<char *> args;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        if (i + 1 < argc && (a == "--n-sets" || a == "--max-seqs" || a == "--snapshot-every")) {
            const long long v = atoll(argv[++i]);
            if (a == "--n-sets") { n_sets = (int) v; } else if (a == "--max-seqs") { n_seq_max = (int) v; } else { snapshot_every = v; }
            continue;
        }
        args.push_back(argv[i]);
    }
    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_IMATRIX, print_usage)) {
        return 1;
    }
    if (n_sets < 1 || n_sets > 64 || n_seq_max < 1) {
        LOG_ERR("%s: bad --n-sets / --max-seqs\n", __func__);
        return 1;
    }

    const int32_t n_piece = params.n_ctx;  // -c: the longest piece a document is cut into
    if (n_piece <= 0) {
        LOG_ERR("%s: requires '--ctx-size' > 0\n", __func__);
        return 1;
    }
    params.n_parallel = n_seq_max;
    params.kv_unified = true;
    params.n_ctx      = std::max(n_piece, params.n_batch);
    params.n_batch    = std::min(params.n_batch, params.n_ctx);

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = hope_cb_eval;
    params.cb_eval_user_data = nullptr;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }

    const int32_t n_layer  = llama_model_n_layer(model);
    const int32_t n_expert = model_arch_int(model, "expert_count");
    const int32_t n_used   = model_arch_int(model, "expert_used_count");
    if (n_expert <= 1 || n_used <= 0) {
        LOG_ERR("%s: not a MoE model (expert_count %d, expert_used_count %d)\n", __func__, n_expert, n_used);
        return 1;
    }
    g_collector.init(n_layer, n_expert, n_used, n_sets);
    // unmasked MTP rows: the last layer runs its FFN on every row although only one token per piece is an output
    llama_set_embeddings_nextn(ctx, true, /*masked*/ false);
    LOG_INF("%s: %d layers, %d experts, %d used per token\n", __func__, n_layer, n_expert, n_used);

    for (const auto & in_file : params.in_files) {
        if (!g_collector.load(in_file)) { return 1; }
        LOG_INF("%s: resuming from %s after %lld documents (%lld tokens)\n", __func__, in_file.c_str(),
                (long long) g_collector.n_docs_done, (long long) g_collector.n_tokens_done);
    }

    LOG_INF("\n%s\n", common_params_get_system_info(params).c_str());

    if (!run(ctx, params, n_piece, n_seq_max, snapshot_every)) { return 1; }
    if (!g_collector.save(params.out_file)) { return 1; }
    LOG_INF("%s: saved %s (%lld documents, %lld tokens, %lld layer-token observations)\n", __func__, params.out_file.c_str(),
            (long long) g_collector.n_docs_done, (long long) g_collector.n_tokens_done, (long long) g_collector.n_obs);

    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}
