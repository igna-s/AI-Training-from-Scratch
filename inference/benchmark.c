/*
 * benchmark.c — Standalone C benchmark for CPU-1 170M inference
 *
 * Compile (Windows):
 *   g++ -O3 -mavx2 -mfma -fopenmp -falign-loops=32 -funroll-loops -ffast-math
 *       -x c++ -o benchmark.exe benchmark.c kernels.cpp -lm
 *
 * Compile (Linux/WSL):
 *   g++ -O3 -mavx2 -mfma -fopenmp -falign-loops=32 -funroll-loops -ffast-math
 *       -x c++ -o benchmark benchmark.c kernels.cpp -lm
 *
 * Run:
 *   ./benchmark --patches 200 --warmup 20 --mode 0
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>

#ifdef _WIN32
#include <windows.h>
static double get_time_sec(void) {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart;
}
#else
#include <time.h>
static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

/* ── Struct must match kernels.cpp LayerDataMAD exactly ─────────────── */
struct LayerDataMAD {
    const float*   norm1_w;  const float* norm2_w; float norm_eps;
    const uint8_t* W_forget; float s_forget;
    const uint8_t* W_cand;   float s_cand;
    const uint8_t* W_out;    float s_out;
    const uint8_t* W_gate;   float s_gate;
    const uint8_t* W_up;     float s_up;
    const uint8_t* W_down;   float s_down;
    const float*   fp_A_mix; const float* fp_B_mix;
    const float*   fp_A_down; const float* fp_B_down;
    const float*   pfnet_norm_w;
    const uint8_t* pfnet_w1; float pfnet_s1;
    const uint8_t* pfnet_w2; float pfnet_s2;
};

/* ── Extern declarations from kernels.cpp ───────────────────────────── */
#ifdef __cplusplus
extern "C" {
#endif

void pack_ternary_to_2bit_mad(const int8_t* W, uint8_t* packed, int d_out, int d_in);

void forward_full_mad(
    const float* x_in, float* x_out, float* h_all,
    const struct LayerDataMAD* layers,
    int n_layers, int d_model, int d_ff, int fp_rank, int pfnet_hidden);

void decode_layer_fused_2bit_mad(
    const float* x_in, float* x_out, float* h,
    const float* norm1_w, const float* norm2_w, float norm_eps,
    const uint8_t* W_forget, float s_forget,
    const uint8_t* W_cand, float s_cand,
    const uint8_t* W_out, float s_out,
    const uint8_t* W_gate, float s_gate,
    const uint8_t* W_up, float s_up,
    const uint8_t* W_down, float s_down,
    int d_model, int d_ff);

#ifdef __cplusplus
}
#endif

/* ── Memory helpers ─────────────────────────────────────────────────── */

static void* alloc_aligned(size_t align, size_t size) {
#ifdef _WIN32
    void* p = _aligned_malloc(size, align);
#else
    void* p = aligned_alloc(align, ((size + align - 1) / align) * align);
#endif
    if (!p) { fprintf(stderr, "OOM %zu bytes\n", size); exit(1); }
    memset(p, 0, size);
    return p;
}

static void free_aligned(void* p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

static float*   alloc_f32(int n) { return (float*)  alloc_aligned(64, n * sizeof(float)); }
static int8_t*  alloc_i8(int n)  { return (int8_t*) alloc_aligned(64, n); }
static uint8_t* alloc_u8(int n)  { return (uint8_t*)alloc_aligned(64, n); }

/* ── Weight creation ───────────────────────────────────────────────── */

static uint8_t* make_random_mad_weight(int d_out, int d_in) {
    size_t raw_size = (size_t)d_out * d_in;
    int8_t* raw = (int8_t*)alloc_aligned(64, raw_size);
    for (size_t i = 0; i < raw_size; i++)
        raw[i] = (int8_t)((rand() % 3) - 1);  /* {-1, 0, 1} */
    int ncb = (d_in + 31) / 32;
    int mad_size = ((d_out + 3) / 4) * ncb * 32;
    uint8_t* packed = alloc_u8(mad_size);
    pack_ternary_to_2bit_mad(raw, packed, d_out, d_in);
    free_aligned(raw);
    return packed;
}

static float* make_random_f32(int n, float scale) {
    float* v = alloc_f32(n);
    for (int i = 0; i < n; i++)
        v[i] = ((float)(rand() % 1000) / 500.0f - 1.0f) * scale;
    return v;
}

/* ── Sorting ───────────────────────────────────────────────────────── */

static int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

/* ── Main ──────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    int d_model = 1024, n_layers = 14, d_ff = 2688;
    int fp_rank = 8, pfnet_hidden = 64;
    int n_patches = 200, n_warmup = 20;
    int mode = 0; /* 0=forward_full_mad, 1=per-layer */
    int n_threads = 0; /* 0 = auto-detect */

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--patches")  && i+1 < argc) n_patches    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup")   && i+1 < argc) n_warmup     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--d_model")  && i+1 < argc) d_model      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n_layers") && i+1 < argc) n_layers     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mode")     && i+1 < argc) mode         = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads")  && i+1 < argc) n_threads    = atoi(argv[++i]);
    }

    /* ── Thread configuration ───────────────────────────────────── */
    if (n_threads <= 0) {
        n_threads = omp_get_num_procs();
        /* On HT systems, halve the count (heuristic) */
        if (n_threads > 4) n_threads = n_threads / 2;
        if (n_threads < 1) n_threads = 1;
    }
    omp_set_num_threads(n_threads);

    srand(42);
    printf("=================================================================\n");
    printf("CPU-1 Benchmark (C) v3 — d_model=%d n_layers=%d d_ff=%d\n", d_model, n_layers, d_ff);
    printf("OMP threads: %d (of %d logical procs)\n", n_threads, omp_get_num_procs());
    printf("=================================================================\n");

    /* ── Allocate model weights ──────────────────────────────────── */
    struct LayerDataMAD* layers = (struct LayerDataMAD*)calloc(n_layers, sizeof(struct LayerDataMAD));
    size_t total_bytes = 0;

    for (int L = 0; L < n_layers; L++) {
        struct LayerDataMAD* ld = &layers[L];
        ld->norm1_w = make_random_f32(d_model, 1.0f);
        ld->norm2_w = make_random_f32(d_model, 1.0f);
        ld->norm_eps = 1e-6f;
        ld->W_forget = make_random_mad_weight(d_model, d_model); ld->s_forget = 0.01f;
        ld->W_cand   = make_random_mad_weight(d_model, d_model); ld->s_cand   = 0.01f;
        ld->W_out    = make_random_mad_weight(d_model, d_model); ld->s_out    = 0.01f;
        ld->W_gate   = make_random_mad_weight(d_ff, d_model);    ld->s_gate   = 0.01f;
        ld->W_up     = make_random_mad_weight(d_ff, d_model);    ld->s_up     = 0.01f;
        ld->W_down   = make_random_mad_weight(d_model, d_ff);    ld->s_down   = 0.01f;
        ld->fp_A_mix = make_random_f32(d_model * fp_rank, 0.01f);
        ld->fp_B_mix = make_random_f32(fp_rank * d_model, 0.01f);
        ld->fp_A_down = NULL; ld->fp_B_down = NULL;
        ld->pfnet_norm_w = make_random_f32(d_model, 1.0f);
        ld->pfnet_w1 = make_random_mad_weight(pfnet_hidden, d_model); ld->pfnet_s1 = 0.01f;
        ld->pfnet_w2 = make_random_mad_weight(d_model, pfnet_hidden); ld->pfnet_s2 = 0.01f;

        int ncb_m = (d_model+31)/32, ncb_ff = (d_ff+31)/32, ncb_pf = (pfnet_hidden+31)/32;
        total_bytes += 3*(size_t)((d_model+3)/4)*ncb_m*32;
        total_bytes += 2*(size_t)((d_ff+3)/4)*ncb_m*32;
        total_bytes += (size_t)((d_model+3)/4)*ncb_ff*32;
        total_bytes += (size_t)((pfnet_hidden+3)/4)*ncb_m*32;
        total_bytes += (size_t)((d_model+3)/4)*ncb_pf*32;
    }
    printf("Weights: %.1f MB (%d layers)\n", total_bytes / 1e6, n_layers);

    /* ── I/O buffers ─────────────────────────────────────────────── */
    float* x_in  = make_random_f32(d_model, 1.0f);
    float* x_out = alloc_f32(d_model);
    float* h_all = alloc_f32(n_layers * d_model);

    /* ── Warmup ──────────────────────────────────────────────────── */
    printf("Warmup %d patches...", n_warmup); fflush(stdout);
    for (int w = 0; w < n_warmup; w++) {
        memset(h_all, 0, n_layers * d_model * sizeof(float));
        if (mode == 0) {
            forward_full_mad(x_in, x_out, h_all, layers,
                n_layers, d_model, d_ff, fp_rank, pfnet_hidden);
        } else {
            float *cur = x_in, *nxt = x_out;
            for (int L = 0; L < n_layers; L++) {
                struct LayerDataMAD* ld = &layers[L];
                decode_layer_fused_2bit_mad(
                    cur, nxt, h_all + L*d_model,
                    ld->norm1_w, ld->norm2_w, ld->norm_eps,
                    ld->W_forget, ld->s_forget, ld->W_cand, ld->s_cand,
                    ld->W_out, ld->s_out, ld->W_gate, ld->s_gate,
                    ld->W_up, ld->s_up, ld->W_down, ld->s_down,
                    d_model, d_ff);
                float* tmp = cur; cur = nxt; nxt = tmp;
            }
        }
    }
    printf(" OK\n");

    /* ── Timed run ───────────────────────────────────────────────── */
    double* times = (double*)malloc(n_patches * sizeof(double));
    for (int p = 0; p < n_patches; p++) {
        memset(h_all, 0, n_layers * d_model * sizeof(float));
        double t0 = get_time_sec();
        if (mode == 0) {
            forward_full_mad(x_in, x_out, h_all, layers,
                n_layers, d_model, d_ff, fp_rank, pfnet_hidden);
        } else {
            float *cur = x_in, *nxt = x_out;
            for (int L = 0; L < n_layers; L++) {
                struct LayerDataMAD* ld = &layers[L];
                decode_layer_fused_2bit_mad(
                    cur, nxt, h_all + L*d_model,
                    ld->norm1_w, ld->norm2_w, ld->norm_eps,
                    ld->W_forget, ld->s_forget, ld->W_cand, ld->s_cand,
                    ld->W_out, ld->s_out, ld->W_gate, ld->s_gate,
                    ld->W_up, ld->s_up, ld->W_down, ld->s_down,
                    d_model, d_ff);
                float* tmp = cur; cur = nxt; nxt = tmp;
            }
        }
        times[p] = get_time_sec() - t0;
    }

    /* ── Stats ───────────────────────────────────────────────────── */
    qsort(times, n_patches, sizeof(double), cmp_double);
    double total = 0;
    for (int i = 0; i < n_patches; i++) total += times[i];
    double p50 = times[n_patches / 2];
    double p95 = times[(int)(n_patches * 0.95)];
    double pps = n_patches / total;
    double bw  = (total_bytes / 1e6) * pps;

    printf("\n=================================================================\n");
    printf("Mode: %s\n", mode == 0 ? "forward_full_mad" : "per-layer");
    printf("Patches: %d  Warmup: %d\n", n_patches, n_warmup);
    printf("-----------------------------------------------------------------\n");
    printf("Throughput:  %.1f patches/sec  (%.1f bytes/sec)\n", pps, pps*4);
    printf("Latency:     p50=%.2f ms  p95=%.2f ms\n", p50*1000, p95*1000);
    printf("Bandwidth:   %.0f MB/s (%.1f%% of ~30 GB/s)\n", bw, bw/30000*100);
    printf("Weight size: %.1f MB\n", total_bytes / 1e6);
    printf("=================================================================\n");

    free(times);
    free_aligned(x_in); free_aligned(x_out); free_aligned(h_all);
    return 0;
}
