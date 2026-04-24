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
#include <sys/mman.h>
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

void forward_full_mad_st(
    const float* x_in, float* x_out, float* h_all,
    const struct LayerDataMAD* layers,
    int n_layers, int d_model, int d_ff, int fp_rank, int pfnet_hidden);

void forward_full_mad_batch(
    const float* const*  x_in_array,
    float* const*        x_out_array,
    float* const*        h_all_array,
    const struct LayerDataMAD* layers,
    int n_layers, int d_model, int d_ff, int fp_rank, int pfnet_hidden,
    int n_patches, int n_threads);

void forward_full_mad_pipeline(
    const float* const*  x_in_array,
    float* const*        x_out_array,
    float*               h_all,
    const struct LayerDataMAD* layers,
    int n_layers, int d_model, int d_ff, int fp_rank, int pfnet_hidden,
    int n_patches, int pin_threads,
    int n_stages, int do_calibrate);

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

void local_decode_step(
    const float* h, const uint8_t* W_hproj, float s_hproj,
    const float* byte_emb_table, const float* pos_emb,
    const int32_t* generated, int k,
    int d_model, int d_local, int n_heads, int d_ff_local, int patch_size,
    const float* norm1_w,
    const uint8_t* W_q, float s_q,
    const uint8_t* W_k, float s_k,
    const uint8_t* W_v, float s_v,
    const uint8_t* W_o, float s_o,
    const float* norm2_w,
    const uint8_t* W_gate, float s_gate,
    const uint8_t* W_up,   float s_up,
    const uint8_t* W_down, float s_down,
    const uint8_t* W_out,  float s_out,
    float norm_eps,
    float* logits_out);

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

static void* alloc_aligned_hugepages(size_t size) {
#ifdef _WIN32
    void* p = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
    if (!p) p = _aligned_malloc(size, 64);
    if (!p) { fprintf(stderr, "OOM %zu bytes\n", size); exit(1); }
    memset(p, 0, size);
#else
    void* p;
    if (posix_memalign(&p, 64, size) != 0) {
        fprintf(stderr, "OOM %zu bytes\n", size); exit(1);
    }
    memset(p, 0, size);
    madvise(p, size, MADV_HUGEPAGE);
#endif
    return p;
}

static void free_aligned(void* p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

static void free_aligned_hugepages(void* p) {
#ifdef _WIN32
    if (!VirtualFree(p, 0, MEM_RELEASE)) _aligned_free(p);
#else
    free(p);
#endif
}

static float*   alloc_f32(int n) { return (float*)  alloc_aligned(64, n * sizeof(float)); }
static int8_t*  alloc_i8(int n)  { return (int8_t*) alloc_aligned(64, n); }
static uint8_t* alloc_u8(int n)  { return (uint8_t*)alloc_aligned(64, n); }
static float*   alloc_f32_hp(int n) { return (float*)  alloc_aligned_hugepages(n * sizeof(float)); }
static uint8_t* alloc_u8_hp(int n)  { return (uint8_t*)alloc_aligned_hugepages(n); }

/* ── Weight creation ───────────────────────────────────────────────── */

static uint8_t* make_random_mad_weight(int d_out, int d_in) {
    size_t raw_size = (size_t)d_out * d_in;
    int8_t* raw = (int8_t*)alloc_aligned(64, raw_size);
    for (size_t i = 0; i < raw_size; i++)
        raw[i] = (int8_t)((rand() % 3) - 1);  /* {-1, 0, 1} */
    int ncb = (d_in + 31) / 32;
    int mad_size = ((d_out + 3) / 4) * ncb * 32;
    uint8_t* packed = alloc_u8_hp(mad_size);
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

static float* make_random_f32_hp(int n, float scale) {
    float* v = alloc_f32_hp(n);
    for (int i = 0; i < n; i++)
        v[i] = ((float)(rand() % 1000) / 500.0f - 1.0f) * scale;
    return v;
}

/* ── Sorting ───────────────────────────────────────────────────────── */

static int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

/* ── MegaByte LocalByteDecoder weight bundle ─────────────────────── */
struct LocalDecW {
    const uint8_t *W_hproj;
    const uint8_t *W_q, *W_k, *W_v, *W_o;
    const uint8_t *W_gate, *W_up, *W_down, *W_out;
    const float   *byte_emb, *pos_emb;
    const float   *norm1_w, *norm2_w;
    int d_local, d_ff_local, n_heads, patch_size;
};

/* Run all 4 local_decode_step calls (autoregressive byte generation
 * inside one patch). x_out is the global model output [d_model]. */
static inline void run_local_patch(
    const float* x_out, int d_model, const struct LocalDecW* L
) {
    int32_t generated[4] = {0};
    float   logits[256];
    const float scale = 0.01f;
    for (int kb = 0; kb < L->patch_size; kb++) {
        local_decode_step(
            x_out,
            L->W_hproj, scale,
            L->byte_emb, L->pos_emb,
            generated, kb,
            d_model, L->d_local, L->n_heads, L->d_ff_local, L->patch_size,
            L->norm1_w,
            L->W_q, scale, L->W_k, scale, L->W_v, scale, L->W_o, scale,
            L->norm2_w,
            L->W_gate, scale, L->W_up, scale, L->W_down, scale,
            L->W_out, scale,
            1e-6f, logits);
        /* emulate argmax of the freshly-generated logits */
        generated[kb] = (kb * 17) % 256;
    }
}

/* ── Main ──────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    int d_model = 1024, n_layers = 14, d_ff = 2688;
    int fp_rank = 8, pfnet_hidden = 64;
    int n_patches = 200, n_warmup = 20;
    /* 0=forward_full_mad (OMP), 1=per-layer, 2=ST single-threaded,
       3=batch, 4=local_decode (global+4 local steps) */
    int mode = 0;
    int n_threads = 0;
    int batch_size = 1;
    int pipe_chunk    = 128;
    int pipe_pin      = 1;
    int pipe_stages   = 6;   /* default: use all 6 physical cores on i7-8750H */
    int pipe_calibrate= 1;   /* default: calibrate per-layer cost + DP split  */
    int local_decode  = 1;   /* default: run 4 local_decode_step per patch    */

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--patches")  && i+1 < argc) n_patches    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup")   && i+1 < argc) n_warmup     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--d_model")  && i+1 < argc) d_model      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n_layers") && i+1 < argc) n_layers     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mode")     && i+1 < argc) mode         = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads")  && i+1 < argc) n_threads    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch")    && i+1 < argc) batch_size   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--chunk")    && i+1 < argc) pipe_chunk   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pin")      && i+1 < argc) pipe_pin     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--stages")   && i+1 < argc) pipe_stages  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--calibrate")&& i+1 < argc) pipe_calibrate=atoi(argv[++i]);
        else if (!strcmp(argv[i], "--local")    && i+1 < argc) local_decode = atoi(argv[++i]);
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
        ld->norm1_w = make_random_f32_hp(d_model, 1.0f);
        ld->norm2_w = make_random_f32_hp(d_model, 1.0f);
        ld->norm_eps = 1e-6f;
        ld->W_forget = make_random_mad_weight(d_model, d_model); ld->s_forget = 0.01f;
        ld->W_cand   = make_random_mad_weight(d_model, d_model); ld->s_cand   = 0.01f;
        ld->W_out    = make_random_mad_weight(d_model, d_model); ld->s_out    = 0.01f;
        ld->W_gate   = make_random_mad_weight(d_ff, d_model);    ld->s_gate   = 0.01f;
        ld->W_up     = make_random_mad_weight(d_ff, d_model);    ld->s_up     = 0.01f;
        ld->W_down   = make_random_mad_weight(d_model, d_ff);    ld->s_down   = 0.01f;
        ld->fp_A_mix = make_random_f32_hp(d_model * fp_rank, 0.01f);
        ld->fp_B_mix = make_random_f32_hp(fp_rank * d_model, 0.01f);
        ld->fp_A_down = NULL; ld->fp_B_down = NULL;
        ld->pfnet_norm_w = make_random_f32_hp(d_model, 1.0f);
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

    /* ── MegaByte LocalByteDecoder weights (always allocated) ────── */
    /* Per-patch full inference = 1 global pass + 4 local_decode_step. */
    /* These weights are SHARED across all modes and persist always.  */
    const int LD_DLOCAL    = 128;
    const int LD_DFF       = 512;
    const int LD_NHEADS    = 4;
    const int LD_PATCHSIZE = 4;
    uint8_t* ld_W_hproj = make_random_mad_weight(LD_PATCHSIZE * LD_DLOCAL, d_model);
    uint8_t* ld_W_q     = make_random_mad_weight(LD_DLOCAL, LD_DLOCAL);
    uint8_t* ld_W_k     = make_random_mad_weight(LD_DLOCAL, LD_DLOCAL);
    uint8_t* ld_W_v     = make_random_mad_weight(LD_DLOCAL, LD_DLOCAL);
    uint8_t* ld_W_o     = make_random_mad_weight(LD_DLOCAL, LD_DLOCAL);
    uint8_t* ld_W_gate  = make_random_mad_weight(LD_DFF,    LD_DLOCAL);
    uint8_t* ld_W_up    = make_random_mad_weight(LD_DFF,    LD_DLOCAL);
    uint8_t* ld_W_down  = make_random_mad_weight(LD_DLOCAL, LD_DFF);
    uint8_t* ld_W_out   = make_random_mad_weight(256,       LD_DLOCAL);
    float*   ld_byte_emb = make_random_f32(256 * LD_DLOCAL, 0.01f);
    float*   ld_pos_emb  = make_random_f32(LD_PATCHSIZE * LD_DLOCAL, 0.01f);
    float*   ld_norm1_w  = make_random_f32(LD_DLOCAL, 1.0f);
    float*   ld_norm2_w  = make_random_f32(LD_DLOCAL, 1.0f);
    /* Local decoder weights total ~= 1 MB (negligible vs 40 MB global) */

    struct LocalDecW LDW = {
        ld_W_hproj, ld_W_q, ld_W_k, ld_W_v, ld_W_o,
        ld_W_gate, ld_W_up, ld_W_down, ld_W_out,
        ld_byte_emb, ld_pos_emb, ld_norm1_w, ld_norm2_w,
        LD_DLOCAL, LD_DFF, LD_NHEADS, LD_PATCHSIZE
    };

    /* ── Allocate batch arrays for mode 3 ────────────────────────── */
    float** x_in_array = NULL, **x_out_array = NULL, **h_all_array = NULL;
    if (mode == 3) {
        x_in_array = (float**)malloc(batch_size * sizeof(float*));
        x_out_array = (float**)malloc(batch_size * sizeof(float*));
        h_all_array = (float**)malloc(batch_size * sizeof(float*));
        for (int b = 0; b < batch_size; b++) {
            x_in_array[b] = alloc_f32(d_model);
            x_out_array[b] = alloc_f32(d_model);
            h_all_array[b] = alloc_f32(n_layers * d_model);
            memcpy(x_in_array[b], x_in, d_model * sizeof(float));
        }
    }

    /* ── Allocate single-stream arrays for mode 4 (pipeline) ─────── */
    /* Pipeline reuses ONE persistent h_all (single recurrent stream) */
    /* and processes a batch of n_patches sequentially through 4      */
    /* layer-pipeline stages.                                         */
    float** mode4_x_in  = NULL;
    float** mode4_x_out = NULL;
    float*  mode4_h_all = NULL;
    int     mode4_pin   = pipe_pin; /* pin threads to logical CPUs 0,2,4,6 */
    if (mode == 4) {
        mode4_x_in  = (float**)malloc(pipe_chunk * sizeof(float*));
        mode4_x_out = (float**)malloc(pipe_chunk * sizeof(float*));
        for (int p = 0; p < pipe_chunk; p++) {
            mode4_x_in[p]  = alloc_f32(d_model);
            mode4_x_out[p] = alloc_f32(d_model);
            memcpy(mode4_x_in[p], x_in, d_model * sizeof(float));
        }
        mode4_h_all = alloc_f32(n_layers * d_model);
        printf("Mode 4 pipeline: stages=%d chunk=%d pin=%d calibrate=%d\n",
               pipe_stages, pipe_chunk, mode4_pin, pipe_calibrate);
    }

    /* ── Warmup ──────────────────────────────────────────────────── */
    printf("Warmup %d patches...", n_warmup); fflush(stdout);
    for (int w = 0; w < n_warmup; w++) {
        if (mode == 0) {
            memset(h_all, 0, n_layers * d_model * sizeof(float));
            forward_full_mad(x_in, x_out, h_all, layers,
                n_layers, d_model, d_ff, fp_rank, pfnet_hidden);
        } else if (mode == 1) {
            memset(h_all, 0, n_layers * d_model * sizeof(float));
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
        } else if (mode == 2) {
            memset(h_all, 0, n_layers * d_model * sizeof(float));
            forward_full_mad_st(x_in, x_out, h_all, layers,
                n_layers, d_model, d_ff, fp_rank, pfnet_hidden);
            if (local_decode) run_local_patch(x_out, d_model, &LDW);
        } else if (mode == 3) {
            for (int b = 0; b < batch_size; b++)
                memset(h_all_array[b], 0, n_layers * d_model * sizeof(float));
            forward_full_mad_batch((const float* const*)x_in_array,
                x_out_array, h_all_array, layers,
                n_layers, d_model, d_ff, fp_rank, pfnet_hidden,
                batch_size, n_threads);
            if (local_decode)
                for (int b = 0; b < batch_size; b++)
                    run_local_patch(x_out_array[b], d_model, &LDW);
        } else if (mode == 4) {
            memset(mode4_h_all, 0, n_layers * d_model * sizeof(float));
            /* Calibrate only on the very first warmup iteration. */
            int do_cal = (w == 0) ? pipe_calibrate : 0;
            forward_full_mad_pipeline((const float* const*)mode4_x_in,
                mode4_x_out, mode4_h_all, layers,
                n_layers, d_model, d_ff, fp_rank, pfnet_hidden,
                pipe_chunk, mode4_pin,
                pipe_stages, do_cal);
            if (local_decode)
                for (int p = 0; p < pipe_chunk; p++)
                    run_local_patch(mode4_x_out[p], d_model, &LDW);
        }
    }
    printf(" OK\n");

    /* ── Timed run ───────────────────────────────────────────────── */
    double* times = (double*)malloc(n_patches * sizeof(double));
    for (int p = 0; p < n_patches; p++) {
        double t0 = get_time_sec();
        if (mode == 0) {
            memset(h_all, 0, n_layers * d_model * sizeof(float));
            forward_full_mad(x_in, x_out, h_all, layers,
                n_layers, d_model, d_ff, fp_rank, pfnet_hidden);
        } else if (mode == 1) {
            memset(h_all, 0, n_layers * d_model * sizeof(float));
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
        } else if (mode == 2) {
            memset(h_all, 0, n_layers * d_model * sizeof(float));
            forward_full_mad_st(x_in, x_out, h_all, layers,
                n_layers, d_model, d_ff, fp_rank, pfnet_hidden);
            if (local_decode) run_local_patch(x_out, d_model, &LDW);
        } else if (mode == 3) {
            for (int b = 0; b < batch_size; b++)
                memset(h_all_array[b], 0, n_layers * d_model * sizeof(float));
            forward_full_mad_batch((const float* const*)x_in_array,
                x_out_array, h_all_array, layers,
                n_layers, d_model, d_ff, fp_rank, pfnet_hidden,
                batch_size, n_threads);
            if (local_decode)
                for (int b = 0; b < batch_size; b++)
                    run_local_patch(x_out_array[b], d_model, &LDW);
        } else if (mode == 4) {
            /* Layer-pipeline single-stream: process pipe_chunk patches */
            /* through pipe_stages worker threads. Split was already    */
            /* calibrated during warmup; here we always pass do_cal=0   */
            /* so the partition is recomputed cheaply (even split fall- */
            /* back) — the dominant cost is per-patch compute.          */
            forward_full_mad_pipeline((const float* const*)mode4_x_in,
                mode4_x_out, mode4_h_all, layers,
                n_layers, d_model, d_ff, fp_rank, pfnet_hidden,
                pipe_chunk, mode4_pin,
                pipe_stages, 0);
            if (local_decode) {
                /* Local decode is autoregressive within a patch but */
                /* embarrassingly parallel across patches. Use OMP. */
                #pragma omp parallel for schedule(static) num_threads(pipe_stages)
                for (int p2 = 0; p2 < pipe_chunk; p2++)
                    run_local_patch(mode4_x_out[p2], d_model, &LDW);
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

    // Real patches/sec:
    //   mode 3: each iteration processes batch_size patches in parallel
    //   mode 4: each iteration processes pipe_chunk patches through pipeline
    double actual_patches;
    if      (mode == 3) actual_patches = (double)(n_patches * batch_size);
    else if (mode == 4) actual_patches = (double)(n_patches * pipe_chunk);
    else                actual_patches = (double)n_patches;
    double pps = actual_patches / total;
    double bw  = (total_bytes / 1e6) * pps;

    printf("\n=================================================================\n");
    const char* mode_names[] = {
        "forward_full_mad (OMP)", "per-layer",
        "forward_full_mad_st (ST)", "batch (4 streams)",
        "pipeline (4-stage layer pipeline, 1 stream)" };
    printf("Mode: %s\n", mode_names[mode]);
    if (mode == 3) {
        printf("Iterations: %d (batch_size=%d, total patches: %d)  Warmup: %d\n",
               n_patches, batch_size, (int)actual_patches, n_warmup);
    } else if (mode == 4) {
        printf("Iterations: %d (pipe_chunk=%d, total patches: %d)  Warmup: %d\n",
               n_patches, pipe_chunk, (int)actual_patches, n_warmup);
    } else {
        printf("Patches: %d  Warmup: %d\n", n_patches, n_warmup);
    }
    printf("-----------------------------------------------------------------\n");
    printf("Throughput:  %.1f patches/sec  (%.1f bytes/sec)\n", pps, pps*4);
    printf("Latency:     p50=%.2f ms  p95=%.2f ms\n", p50*1000, p95*1000);
    printf("Bandwidth:   %.0f MB/s (%.1f%% of ~30 GB/s)\n", bw, bw/30000*100);
    printf("Weight size: %.1f MB\n", total_bytes / 1e6);
    printf("=================================================================\n");

    free(times);

    if (mode == 3) {
        for (int b = 0; b < batch_size; b++) {
            free_aligned(x_in_array[b]);
            free_aligned(x_out_array[b]);
            free_aligned(h_all_array[b]);
        }
        free(x_in_array);
        free(x_out_array);
        free(h_all_array);
    }

    if (mode == 4) {
        for (int p = 0; p < pipe_chunk; p++) {
            free_aligned(mode4_x_in[p]);
            free_aligned(mode4_x_out[p]);
        }
        free(mode4_x_in);
        free(mode4_x_out);
        free_aligned(mode4_h_all);
    }

    free_aligned(x_in); free_aligned(x_out); free_aligned(h_all);
    return 0;
}
