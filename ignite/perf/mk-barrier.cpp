// Megakernel Stage 0: grid barrier cost vs kernel launch cost on RDNA3, and graph capture of a persistent kernel.
// build: hipcc -O3 --offload-arch=gfx1100 mk-barrier.cpp -o mk-barrier
// usage: mk-barrier [device]
//
// Barrier: one 64-bit arrival counter that is never reset. Barrier k of a grid of n blocks waits until the counter
// reaches (k+1)*n; a block gets k from its own atomicAdd, so launches can follow each other without a reset.
// Every spin is bounded: on timeout the block sets an error flag and leaves (no hang).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); exit(1); } } while (0)

static constexpr long long SPIN_LIMIT = 1ll << 26;

__device__ __forceinline__ void grid_barrier(unsigned long long * arrive, unsigned n, int * err) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        const unsigned long long old    = atomicAdd(arrive, 1ull);
        const unsigned long long target = (old / n + 1) * n;
        long long spins = 0;
        while (__hip_atomic_load(arrive, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < target) {
            __builtin_amdgcn_s_sleep(1);
            if (++spins > SPIN_LIMIT) {
                atomicExch(err, 1);
                break;
            }
        }
    }
    __syncthreads();
}

// nbar barriers; between them every block does a little work so the compiler keeps the loop
__global__ void persistent(unsigned long long * arrive, int * err, float * sink, int nbar) {
    float acc = threadIdx.x;
    for (int i = 0; i < nbar; ++i) {
        acc = acc * 1.0001f + 1.0f;
        grid_barrier(arrive, gridDim.x, err);
    }
    if (acc == -1.0f) {
        sink[blockIdx.x] = acc;
    }
}

__global__ void tiny(float * sink) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && sink[0] == -1.0f) {
        sink[1] = 1.0f;
    }
}

int main(int argc, char ** argv) {
    const int dev = argc > 1 ? atoi(argv[1]) : 0;
    CHECK(hipSetDevice(dev));
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, dev));
    int occ = 0;
    CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&occ, persistent, 256, 0));
    printf("device %d %s: %d CUs, %d blocks/CU of 256 threads co-resident\n", dev, prop.name, prop.multiProcessorCount, occ);

    unsigned long long * arrive;
    int * err;
    float * sink;
    CHECK(hipMalloc(&arrive, sizeof(*arrive)));
    CHECK(hipMalloc(&err, sizeof(*err)));
    CHECK(hipMalloc(&sink, 4096 * sizeof(float)));
    CHECK(hipMemset(arrive, 0, sizeof(*arrive)));
    CHECK(hipMemset(err, 0, sizeof(*err)));
    CHECK(hipMemset(sink, 0, 4096 * sizeof(float)));

    hipStream_t s;
    CHECK(hipStreamCreate(&s));
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0));
    CHECK(hipEventCreate(&e1));

    auto time_ms = [&](auto && body) {
        body();  // warmup
        CHECK(hipStreamSynchronize(s));
        CHECK(hipEventRecord(e0, s));
        body();
        CHECK(hipEventRecord(e1, s));
        CHECK(hipEventSynchronize(e1));
        float ms;
        CHECK(hipEventElapsedTime(&ms, e0, e1));
        return ms;
    };

    // 1. barrier cost for a few grid sizes: (t(N) - t(0)) / N
    const int N = 2000;
    for (int g : { prop.multiProcessorCount, prop.multiProcessorCount * occ / 2, prop.multiProcessorCount * occ }) {
        if (g < 1) {
            continue;
        }
        const float t0 = time_ms([&] { persistent<<<g, 256, 0, s>>>(arrive, err, sink, 0); });
        const float tn = time_ms([&] { persistent<<<g, 256, 0, s>>>(arrive, err, sink, N); });
        int h_err = 0;
        CHECK(hipMemcpy(&h_err, err, sizeof(int), hipMemcpyDeviceToHost));
        printf("grid %4d: %.3f us per barrier (kernel alone %.1f us)%s\n", g, 1000.0f * (tn - t0) / N, 1000.0f * t0,
               h_err ? "  BARRIER TIMEOUT" : "");
    }

    // 2. launch cost: back-to-back tiny kernels in a captured graph (what ggml-cuda runs), per launch
    {
        const int L = 2000;
        hipGraph_t graph;
        hipGraphExec_t exec;
        CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeRelaxed));
        for (int i = 0; i < L; ++i) {
            tiny<<<1, 64, 0, s>>>(sink);
        }
        CHECK(hipStreamEndCapture(s, &graph));
        CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        const float t = time_ms([&] { CHECK(hipGraphLaunch(exec, s)); });
        printf("graph of %d tiny kernels: %.3f us per launch\n", L, 1000.0f * t / L);
        CHECK(hipGraphExecDestroy(exec));
        CHECK(hipGraphDestroy(graph));
    }

    // 3. a persistent barrier kernel inside a captured graph, replayed: correct (no timeout) and same cost
    {
        const int g = prop.multiProcessorCount * occ;
        hipGraph_t graph;
        hipGraphExec_t exec;
        CHECK(hipMemset(err, 0, sizeof(int)));
        CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeRelaxed));
        for (int i = 0; i < 48; ++i) {
            tiny<<<1, 64, 0, s>>>(sink);
            persistent<<<g, 256, 0, s>>>(arrive, err, sink, 20);
        }
        CHECK(hipStreamEndCapture(s, &graph));
        CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        float total = 0.0f;
        for (int r = 0; r < 20; ++r) {
            total += time_ms([&] { CHECK(hipGraphLaunch(exec, s)); });
        }
        int h_err = 0;
        CHECK(hipMemcpy(&h_err, err, sizeof(int), hipMemcpyDeviceToHost));
        printf("graph replay: 48 x (tiny + persistent with 20 barriers), 40 replays: %.1f us per replay, %s\n",
               1000.0f * total / 20, h_err ? "BARRIER TIMEOUT" : "ok");
        CHECK(hipGraphExecDestroy(exec));
        CHECK(hipGraphDestroy(graph));
    }
    // 4. fixed cost of a persistent kernel inside a graph, by grid size: 200 launches with 0 and with 10 barriers
    for (int g : { prop.multiProcessorCount, 2 * prop.multiProcessorCount, prop.multiProcessorCount * occ / 2, prop.multiProcessorCount * occ }) {
        float per[2];
        for (int k = 0; k < 2; ++k) {
            hipGraph_t graph;
            hipGraphExec_t exec;
            CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeRelaxed));
            for (int i = 0; i < 200; ++i) {
                persistent<<<g, 256, 0, s>>>(arrive, err, sink, k ? 10 : 0);
            }
            CHECK(hipStreamEndCapture(s, &graph));
            CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
            per[k] = 1000.0f * time_ms([&] { CHECK(hipGraphLaunch(exec, s)); }) / 200;
            CHECK(hipGraphExecDestroy(exec));
            CHECK(hipGraphDestroy(graph));
        }
        printf("in graph, grid %4d: %.2f us per launch with 0 barriers, %.2f us with 10 (%.2f us/barrier)\n",
               g, per[0], per[1], (per[1] - per[0]) / 10);
    }
    return 0;
}
