// llama-topk-cache: a model's top-k next-token log-probs at every position of a token shard (teacher forcing), for
// distilling a drafter from the model it will draft for.
//
//   llama-topk-cache -m MODEL.gguf -f SHARD.bin -o OUT.topk -c SEQ [--chunks N] [-t THREADS] [placement args]
//
// SHARD: uint32 token ids (the drafter's training shards). Window w = tokens [w*SEQ, (w+1)*SEQ), decoded from an
// empty context; position i's entry predicts token i+1 of the shard, so a window's last entry predicts the first
// token of the next window. --chunks limits the number of windows. TOPK_K (env, default 8) entries per position.
// OUT: uint32 per entry, id in the low 18 bits, round(-lp * 512) clipped to 16383 in the high 14 bits (any other
// teacher, e.g. a small HF model, can write the same layout). Resumable: an existing OUT is continued at its last
// whole window.
#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

// top-k of log_softmax(lg) for one row: one pass for max and the top-k (min at the front), one for the sum
static void row_topk(const float * lg, int n_vocab, int K, uint32_t * out) {
    std::vector<std::pair<float, int>> top(K, {-INFINITY, 0});
    float mx = -INFINITY;
    for (int v = 0; v < n_vocab; v++) {
        const float x = lg[v];
        mx = std::max(mx, x);
        if (x > top[0].first) {
            top[0] = {x, v};
            for (int k = 0; k + 1 < K && top[k].first > top[k + 1].first; k++) std::swap(top[k], top[k + 1]);
        }
    }
    double se = 0;
    for (int v = 0; v < n_vocab; v++) se += std::exp((double) (lg[v] - mx));
    const double lse = mx + std::log(se);
    for (int k = 0; k < K; k++) {
        const auto & e = top[K - 1 - k];   // highest first
        const double q = std::min(16383.0, std::max(0.0, std::round(-(e.first - lse) * 512.0)));
        out[k] = (uint32_t) e.second | ((uint32_t) q << 18);
    }
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_IMATRIX)) {   // -f shard, -o output, --chunks
        return 1;
    }
    const int K = getenv("TOPK_K") ? atoi(getenv("TOPK_K")) : 8;
    const size_t n_tok = std::filesystem::file_size(params.prompt_file) / 4;
    std::ifstream in(params.prompt_file, std::ios::binary);
    if (!in) { fprintf(stderr, "%s: cannot open '%s' (-f)\n", __func__, params.prompt_file.c_str()); return 1; }

    common_init();
    llama_backend_init();
    auto init = common_init_from_params(params);
    llama_context * ctx = init->context();
    const llama_model * model = init->model();
    if (!ctx || !model) { fprintf(stderr, "%s: failed to load the model\n", __func__); return 1; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int seq = llama_n_ctx(ctx), n_batch = llama_n_batch(ctx);
    const int n_thr = std::max(1, params.cpuparams.n_threads);
    if (n_vocab > (1 << 18)) { fprintf(stderr, "%s: vocab %d does not fit 18 bits\n", __func__, n_vocab); return 1; }

    size_t n_win = (n_tok - 1) / seq;
    if (params.n_chunks > 0) n_win = std::min(n_win, (size_t) params.n_chunks);
    const size_t win_bytes = (size_t) seq * K * 4;
    size_t done = 0;
    if (std::filesystem::exists(params.out_file)) {
        done = std::filesystem::file_size(params.out_file) / win_bytes;
        std::filesystem::resize_file(params.out_file, done * win_bytes);
    }
    std::ofstream out(params.out_file, std::ios::binary | std::ios::app);
    fprintf(stderr, "%s: %zu windows of %d, K=%d, starting at %zu, %d top-k threads\n", __func__, n_win, seq, K, done, n_thr);

    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token> toks(seq);
    std::vector<uint32_t> rec((size_t) seq * K);
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t w = done; w < n_win; w++) {
        in.seekg((std::streamoff) (w * seq * 4));
        in.read((char *) toks.data(), (std::streamsize) seq * 4);
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int b0 = 0; b0 < seq; b0 += n_batch) {
            const int nb = std::min(n_batch, seq - b0);
            batch.n_tokens = nb;
            for (int i = 0; i < nb; i++) {
                batch.token[i] = toks[b0 + i]; batch.pos[i] = b0 + i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
                batch.logits[i] = true;
            }
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed at window %zu\n", w); return 1; }
            std::vector<std::thread> th;
            for (int t = 0; t < n_thr; t++) th.emplace_back([&, t] {
                for (int i = t; i < nb; i += n_thr) row_topk(llama_get_logits_ith(ctx, i), n_vocab, K, &rec[(size_t) (b0 + i) * K]);
            });
            for (auto & x : th) x.join();
        }
        out.write((const char *) rec.data(), (std::streamsize) rec.size() * 4);
        if ((w + 1) % 20 == 0 || w + 1 == n_win) {
            out.flush();
            const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            fprintf(stderr, "%s: %zu/%zu windows, %.0f tok/s\n", __func__, w + 1, n_win, (w + 1 - done) * (double) seq / dt);
        }
    }
    llama_batch_free(batch);
    llama_backend_free();
    return 0;
}
