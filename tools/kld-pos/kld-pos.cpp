// llama-kld-pos: per-position KL divergence of a model against a reference model's top-K, along fixed token sequences
// (teacher forcing). Used to find WHERE a pruned/quantized model departs from the full model (reply start, </think>,
// tool-call start, ...) instead of one average over all tokens, which hides single-token behaviour decisions.
//
//   llama-kld-pos -m MODEL.gguf -f IN.kldp -o OUT.kldo [usual placement args: -ngl -dev -ts -sm -c -b -ub -fa ...]
//
// IN  (written by ignite/perf/kld_pos.py from a rare_tokens.py --dump of the reference model):
//     "KLDP" u32 n_samples u32 K, then per sample: u32 n_prompt u32 n_gen, i32 prompt[n_prompt], i32 gen[n_gen],
//     i32 top_ids[n_gen][K], f32 top_lp[n_gen][K]   (reference log-probs of its top-K tokens at each gen position)
// OUT "KLDO" u32 n_samples, then per sample: u32 n_gen, then per gen position 5 x f32:
//     kld        KL(ref || model) over the reference top-K plus one tail bucket
//     lp_gen     model log-prob of the reference's generated token
//     lp_end     model log-prob of </think>
//     top1       model's top-1 token id (stored as f32 bits of an i32)
//     mass_topk  model probability mass on the reference top-K
#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

static bool rd(std::ifstream & f, void * p, size_t n) { f.read((char *) p, n); return (size_t) f.gcount() == n; }

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_IMATRIX)) {   // -f input, -o output
        return 1;
    }
    std::ifstream in(params.prompt_file, std::ios::binary);
    char magic[4]; uint32_t n_samples = 0, K = 0;
    if (!in || !rd(in, magic, 4) || memcmp(magic, "KLDP", 4) != 0 || !rd(in, &n_samples, 4) || !rd(in, &K, 4)) {
        fprintf(stderr, "%s: cannot read a KLDP file from '%s' (-f)\n", __func__, params.prompt_file.c_str());
        return 1;
    }
    common_init();
    llama_backend_init();
    auto init = common_init_from_params(params);
    llama_context * ctx = init->context();
    const llama_model * model = init->model();
    if (!ctx || !model) { fprintf(stderr, "%s: failed to load the model\n", __func__); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_ctx = llama_n_ctx(ctx), n_batch = llama_n_batch(ctx);
    const auto end_tok = common_tokenize(ctx, "</think>", false, true);
    const llama_token think_end = end_tok.size() == 1 ? end_tok[0] : -1;
    if (think_end < 0) fprintf(stderr, "%s: </think> is not a single token; lp_end will be NaN\n", __func__);

    std::ofstream out(params.out_file, std::ios::binary);
    out.write("KLDO", 4); out.write((const char *) &n_samples, 4);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<float> ls(n_vocab);
    double kld_sum = 0; size_t kld_n = 0;

    for (uint32_t s = 0; s < n_samples; s++) {
        uint32_t np = 0, ng = 0;
        if (!rd(in, &np, 4) || !rd(in, &ng, 4)) { fprintf(stderr, "truncated input at sample %u\n", s); return 1; }
        std::vector<llama_token> toks(np + ng);
        std::vector<int32_t> top_ids((size_t) ng * K);
        std::vector<float> top_lp((size_t) ng * K);
        if (!rd(in, toks.data(), 4 * (size_t) (np + ng)) || !rd(in, top_ids.data(), 4 * top_ids.size()) ||
            !rd(in, top_lp.data(), 4 * top_lp.size())) { fprintf(stderr, "truncated sample %u\n", s); return 1; }
        if ((int) (np + ng) > n_ctx || np == 0 || ng == 0) {
            fprintf(stderr, "sample %u: %u tokens does not fit -c %d (or empty)\n", s, np + ng, n_ctx); return 1;
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        std::vector<float> rec((size_t) ng * 5);
        // the logits at position p predict token p+1: gen token j is predicted at position np - 1 + j
        const int n_in = (int) (np + ng - 1);
        for (int b0 = 0; b0 < n_in; b0 += n_batch) {
            const int b1 = std::min(n_in, b0 + n_batch);
            batch.n_tokens = b1 - b0;
            for (int p = b0; p < b1; p++) {
                const int i = p - b0;
                batch.token[i] = toks[p]; batch.pos[i] = p; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
                batch.logits[i] = p >= (int) np - 1;
            }
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed at sample %u\n", s); return 1; }
            for (int p = std::max(b0, (int) np - 1); p < b1; p++) {
                const int j = p - ((int) np - 1);
                const float * lg = llama_get_logits_ith(ctx, p - b0);
                float mx = lg[0]; int am = 0;
                for (int v = 1; v < n_vocab; v++) if (lg[v] > mx) { mx = lg[v]; am = v; }
                double se = 0; for (int v = 0; v < n_vocab; v++) se += std::exp((double) lg[v] - mx);
                const double lse = mx + std::log(se);
                for (int v = 0; v < n_vocab; v++) ls[v] = (float) (lg[v] - lse);
                double kl = 0, pb_top = 0, ps_top = 0;
                for (uint32_t k = 0; k < K; k++) {
                    const int id = top_ids[(size_t) j * K + k]; const double lb = top_lp[(size_t) j * K + k];
                    const double pb = std::exp(lb);
                    pb_top += pb; ps_top += std::exp((double) ls[id]);
                    kl += pb * (lb - ls[id]);
                }
                const double tb = std::max(1.0 - pb_top, 1e-10), ts = std::max(1.0 - ps_top, 1e-10);
                kl += tb * (std::log(tb) - std::log(ts));
                float * r = &rec[(size_t) j * 5];
                r[0] = (float) kl; r[1] = ls[toks[np + j]]; r[2] = think_end >= 0 ? ls[think_end] : NAN;
                int32_t t1 = am; memcpy(&r[3], &t1, 4); r[4] = (float) ps_top;
                kld_sum += kl; kld_n++;
            }
        }
        out.write((const char *) &ng, 4); out.write((const char *) rec.data(), 4 * rec.size());
        if ((s + 1) % 50 == 0 || s + 1 == n_samples)
            fprintf(stderr, "%s: %u/%u samples, mean KLD so far %.4f\n", __func__, s + 1, n_samples, kld_sum / kld_n);
    }
    llama_batch_free(batch);
    llama_backend_free();
    return 0;
}
