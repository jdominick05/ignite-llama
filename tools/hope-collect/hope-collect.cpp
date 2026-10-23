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
// Statistics are kept in two sets by batch parity (even / odd batches), so the convergence check (solve each
// half, compare the keep-sets) needs no second pass. The solver input is their sum.
//
// Run it on the CPU backend (-dev none -ngl 0): fused or megakernel GPU paths may not expose the named tensors.

#include "llama.h"

#include "arg.h"
#include "common.h"
#include "log.h"

#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -f calib.txt -o hope-obs.bin -c 2048 -b 4096 -dev none -ngl 0 \\\n"
        "       [--chunks N] [--output-frequency 10] [--in-file hope-obs.bin (resume)]\n", argv[0]);
    LOG("\n");
}

static const char   HOPE_MAGIC[8] = { 'H', 'O', 'P', 'E', 'O', 'B', 'S', '1' };
static const int    N_SETS        = 2;

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
    int64_t n_chunks_done = 0;   // chunks fully processed, for resuming
    int     cur_set  = 0;        // set the running batch accumulates into

    std::vector<layer_stats> sets[N_SETS]; // [set][layer]

    // final router weights of the current ubatch per layer, by priority of the name they were seen under
    struct weights_buf { int prio = -1; int64_t n_tokens = 0; std::vector<float> w; };
    std::map<int, weights_buf> weights;

    double  max_gate_sum_err = 0.0;
    int64_t n_obs = 0;
    std::mutex mtx;
    std::vector<uint8_t> tmp;

    void init(int32_t nl, int32_t ne, int32_t nu) {
        n_layer = nl; n_expert = ne; n_used = nu;
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
        const int32_t hdr[4] = { n_layer, n_expert, n_used, N_SETS };
        fwrite(HOPE_MAGIC, 1, 8, f);
        fwrite(hdr, sizeof(hdr), 1, f);
        fwrite(&n_chunks_done, sizeof(n_chunks_done), 1, f);
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
        return true;
    }

    bool load(const std::string & path) {
        FILE * f = fopen(path.c_str(), "rb");
        if (!f) { LOG_ERR("%s: cannot open %s\n", __func__, path.c_str()); return false; }
        char magic[8];
        int32_t hdr[4];
        bool ok = fread(magic, 1, 8, f) == 8 && memcmp(magic, HOPE_MAGIC, 8) == 0 && fread(hdr, sizeof(hdr), 1, f) == 1;
        ok = ok && hdr[0] == n_layer && hdr[1] == n_expert && hdr[2] == n_used && hdr[3] == N_SETS;
        ok = ok && fread(&n_chunks_done, sizeof(n_chunks_done), 1, f) == 1;
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
        if (!ok) { LOG_ERR("%s: %s is not a matching hope-collect file\n", __func__, path.c_str()); }
        return ok;
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

static bool run(llama_context * ctx, const common_params & params, int32_t n_ctx) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    LOG_INF("%s: tokenizing the input ..\n", __func__);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true, params.parse_special);

    const int64_t n_chunk_max = tokens.size() / n_ctx;
    const int64_t n_chunk     = params.n_chunks < 0 ? n_chunk_max : std::min<int64_t>(params.n_chunks, n_chunk_max);
    const int64_t first       = params.i_chunk > 0 ? params.i_chunk : g_collector.n_chunks_done;
    const int     n_batch     = params.n_batch;
    const int     n_seq       = std::max(1, n_batch / n_ctx);
    const int     num_batches = (n_ctx + n_batch - 1) / n_batch;
    GGML_ASSERT(n_batch < n_ctx || n_batch % n_ctx == 0);

    if (first >= n_chunk) {
        LOG_ERR("%s: nothing to do: start chunk %lld, %lld chunks of %d tokens in the input\n", __func__, (long long) first, (long long) n_chunk, n_ctx);
        return false;
    }
    LOG_INF("%s: %zu tokens, chunks %lld..%lld of %d tokens, batch %d, %d sequences per batch\n",
            __func__, tokens.size(), (long long) first, (long long) n_chunk - 1, n_ctx, n_batch, n_seq);

    common_batch batch(ctx);
    const auto t_run = std::chrono::steady_clock::now();
    int64_t n_done_run = 0;
    int     n_iter     = 0;

    for (int64_t i = first; i < n_chunk; i += n_seq) {
        const int n_seq_batch = (int) std::min<int64_t>(n_seq, n_chunk - i);
        g_collector.cur_set = (int) ((i / n_seq) % N_SETS);

        llama_memory_clear(llama_get_memory(ctx), true);

        for (int j = 0; j < num_batches; ++j) {
            const int64_t batch_start = i*n_ctx + (int64_t) j*n_batch;
            const int     batch_size  = std::min<int>(n_ctx - j*n_batch, n_batch);

            batch.clear();
            for (int seq = 0; seq < n_seq_batch; seq++) {
                const int64_t seq_start = batch_start + (int64_t) seq*n_ctx;
                for (int k = 0; k < batch_size; ++k) {
                    const llama_token tok = (add_bos && j == 0 && k == 0) ? llama_vocab_bos(vocab) : tokens[seq_start + k];
                    // every token is an output: otherwise the last layer only runs its FFN for the output rows
                    batch.add(tok, j*n_batch + k, seq, true);
                }
            }
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
                LOG_ERR("%s: failed to eval\n", __func__);
                return false;
            }
        }
        llama_synchronize(ctx);

        g_collector.n_chunks_done = i + n_seq_batch;
        n_done_run += n_seq_batch;
        ++n_iter;

        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_run).count();
        const double tps  = n_done_run * n_ctx / secs;
        LOG_INF("%s: chunk %lld/%lld, %.1f tok/s, ETA %.1f h, max |sum g - 1| = %.2e\n", __func__,
                (long long) g_collector.n_chunks_done, (long long) n_chunk, tps,
                (n_chunk - g_collector.n_chunks_done) * n_ctx / tps / 3600.0, g_collector.max_gate_sum_err);

        if (params.n_out_freq > 0 && n_iter % params.n_out_freq == 0) {
            if (!g_collector.save(params.out_file)) { return false; }
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.out_file = "hope-obs.bin";
    params.n_ctx    = 2048;
    params.escape   = false;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_IMATRIX, print_usage)) {
        return 1;
    }

    const int32_t n_ctx = params.n_ctx;
    if (n_ctx <= 0) {
        LOG_ERR("%s: requires '--ctx-size' > 0\n", __func__);
        return 1;
    }
    {
        const int32_t n_seq = std::max(1, params.n_batch / n_ctx);
        params.n_parallel = n_seq;
        params.n_ctx      = n_seq * n_ctx;
        params.n_batch    = std::min(params.n_batch, params.n_ctx);
    }

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
    g_collector.init(n_layer, n_expert, n_used);
    LOG_INF("%s: %d layers, %d experts, %d used per token\n", __func__, n_layer, n_expert, n_used);

    for (const auto & in_file : params.in_files) {
        if (!g_collector.load(in_file)) { return 1; }
        LOG_INF("%s: resuming from %s after %lld chunks\n", __func__, in_file.c_str(), (long long) g_collector.n_chunks_done);
    }

    LOG_INF("\n%s\n", common_params_get_system_info(params).c_str());

    if (!run(ctx, params, n_ctx)) { return 1; }
    if (!g_collector.save(params.out_file)) { return 1; }
    LOG_INF("%s: saved %s (%lld chunks, %lld layer-token observations)\n", __func__, params.out_file.c_str(),
            (long long) g_collector.n_chunks_done, (long long) g_collector.n_obs);

    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}
