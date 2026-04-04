/*
 * Super-kernel CPU optimizado — Multi-arquitectura.
 *
 * Estrategias por tipo de modelo:
 *   ternary_lut_matmul_omp  — modelos ternarios (CPU1ModelV2, Ablation runs 03-15)
 *   fp32_gemv_avx2          — modelos FP16/FP32 (Ablation runs 01, 02, 05, 08)
 *   mlgru_scan_fusion       — update recurrente (todos los modelos MLGRU)
 *
 * Optimizaciones L3 cache:
 *   - LUT Tiling L2-aware: TILE_G=64 grupos → 64×256×4=64KB por tile (cabe en L2)
 *   - Cada thread calcula su propio tile LUT independientemente (sin coherencia cruzada)
 *   - AVX2 SIMD gather con _mm256_i32gather_ps (8 lookups en paralelo)
 *   - Prefetch explicit _MM_HINT_T0 para siguiente bloque de tile
 *   - AVX2 FMA explicito en mlgru_scan_fusion (fuerza vectorizacion en Windows/MinGW)
 *   - fp32_gemv_avx2: 4×unroll FMA con reduccion horizontal AVX
 */

#include <stdint.h>
#include <string.h>
#include <omp.h>
#include <immintrin.h>

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif

// Tile size: 64 grupos × 256 entradas × 4 bytes = 64KB → cabe en L2 por core
#define TILE_G 64

extern "C" {

// ---------------------------------------------------------------------------
// Utilidad: reduccion horizontal de __m256 a float escalar
// ---------------------------------------------------------------------------
static inline float _hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

// ---------------------------------------------------------------------------
// ternary_lut_matmul_omp
//
// Matmul ternario via Lookup Tables — optimizado para L2 cache.
//
// Input:
//   x          [M, d_in]             float32 activaciones
//   w_packed   [d_out, n_groups]     uint8 pesos empaquetados  (Row-Major)
//   w_scale    float                 escalar de cuantizacion
//   decode_table [128, group_size]   tabla constante de decodificacion
//   output     [M, d_out]            float32 salida
//
// Estrategia Tiling:
//   1. Procesar grupos en tiles de TILE_G (default 64).
//      Cada tile genera LUT_tile[TILE_G, 256] = 64KB → cabe en L2.
//   2. Cada thread calcula Phase 1 (LUT) y Phase 2 (gather) de su tile.
//      Phase 1 es replicada por thread (barata, elimina sincronizacion).
//      Phase 2 es distribuida sobre d_out (omp for, sin race condition).
//   3. AVX2 gather (_mm256_i32gather_ps) para 8 lookups simultaneos.
//   4. Prefetch _MM_HINT_T0 para siguiente bloque dentro del tile.
// ---------------------------------------------------------------------------
// Force AVX2+FMA codegen (VEX prefix) — prevents GCC from emitting
// EVEX-encoded AVX-512 auto-vectorization in Phase 1 scalar loops,
// which causes SIGILL on CPUs/OS without full AVX-512 support.
#ifdef __GNUC__
__attribute__((target("avx2,fma")))
#endif
EXPORT void ternary_lut_matmul_omp(
    const float*   x,            // [M, d_in]
    const uint8_t* w_packed,     // [d_out, n_groups]
    float          w_scale,
    const float*   decode_table, // [128, group_size]
    float*         output,       // [M, d_out]
    int M, int d_in, int d_out, int group_size
) {
    int n_groups = d_in / group_size;

    for (int m = 0; m < M; m++) {
        const float* xm  = x      + m * d_in;
        float*       outm = output + m * d_out;

        // Inicializar output a cero (se acumula sobre tiles)
        memset(outm, 0, d_out * sizeof(float));

        // ---------------------------------------------------------------
        // Bucle de tiles sobre grupos (serie) — cada tile cabe en L2
        // ---------------------------------------------------------------
        #pragma omp parallel
        {
            // Thread-local tile LUT: TILE_G*256 floats = 64KB maximo
            // 'static thread_local' → una instancia por thread, reutilizada entre llamadas
            static thread_local float tile_luts[TILE_G * 256];

            for (int g_base = 0; g_base < n_groups; g_base += TILE_G) {
                int g_end     = g_base + TILE_G < n_groups ? g_base + TILE_G : n_groups;
                int tile_size = g_end - g_base;

                // -----------------------------------------------------------
                // FASE 1: Cada thread calcula su propio tile LUT
                // (replicado por thread, sin sincronizacion, muy barato)
                // -----------------------------------------------------------
                for (int g_local = 0; g_local < tile_size; g_local++) {
                    int          g     = g_base + g_local;
                    const float* x_grp = xm + g * group_size;
                    float*       lut_g = tile_luts + g_local * 256;

                    if (group_size == 4) {
                        // Unrolled group_size=4, compilador vectoriza con -O3 -mavx2
                        for (int c = 0; c < 128; c++) {
                            const float* dr = decode_table + c * 4;
                            float s = x_grp[0]*dr[0] + x_grp[1]*dr[1]
                                    + x_grp[2]*dr[2] + x_grp[3]*dr[3];
                            lut_g[c]       =  s;
                            lut_g[128 + c] = -s;
                        }
                    } else {
                        for (int c = 0; c < 128; c++) {
                            const float* dr = decode_table + c * group_size;
                            float s = 0.0f;
                            for (int i = 0; i < group_size; i++) s += x_grp[i] * dr[i];
                            lut_g[c]       =  s;
                            lut_g[128 + c] = -s;
                        }
                    }
                }

                // -----------------------------------------------------------
                // FASE 2: Gather reduction — distribuida sobre d_out
                //
                // Invariante de scheduling: schedule(static) garantiza que
                // el mismo thread siempre maneja el mismo rango de j.
                // → outm[j] es escrito por exactamente un thread → sin race.
                // -----------------------------------------------------------
                #pragma omp for schedule(static)
                for (int j = 0; j < d_out; j++) {
                    const uint8_t* w_row = w_packed + j * n_groups;

                    // Vectores AVX2 para acumulacion y offset de base
                    // vbase[k] = (g_local + k) * 256  para k=0..7
                    __m256 vsum  = _mm256_setzero_ps();
                    __m256i vbase = _mm256_set_epi32(
                        7*256, 6*256, 5*256, 4*256, 3*256, 2*256, 1*256, 0
                    );
                    const __m256i vstep = _mm256_set1_epi32(8 * 256);

                    int g_local = 0;

                    // Loop principal: 8 grupos a la vez via AVX2 gather
                    for (; g_local <= tile_size - 8; g_local += 8) {
                        int g_abs = g_base + g_local;

                        // Prefetch siguiente bloque del tile a L1
                        if (g_local + 16 < tile_size) {
                            _mm_prefetch(
                                (const char*)(tile_luts + (g_local + 16) * 256),
                                _MM_HINT_T0
                            );
                        }

                        // Cargar 8 indices uint8 de w_row y extender a int32
                        __m128i w8  = _mm_loadl_epi64((const __m128i*)(w_row + g_abs));
                        __m256i w32 = _mm256_cvtepu8_epi32(w8);

                        // vindex[k] = (g_local + k) * 256 + w_row[g_abs + k]
                        __m256i vindex = _mm256_add_epi32(vbase, w32);

                        // Gather: 8 floats de tile_luts en posiciones vindex[k]
                        __m256 vval = _mm256_i32gather_ps(tile_luts, vindex, 4);
                        vsum = _mm256_add_ps(vsum, vval);

                        vbase = _mm256_add_epi32(vbase, vstep);
                    }

                    // Reduccion horizontal: sum(vsum[0..7])
                    float sum = _hsum256_ps(vsum);

                    // Tail escalar para grupos restantes (< 8)
                    for (; g_local < tile_size; g_local++) {
                        sum += tile_luts[g_local * 256 + w_row[g_base + g_local]];
                    }

                    // Acumular sobre tiles (el mismo thread siempre escribe este j)
                    outm[j] += sum;
                }
                // Barrera implicita al final de '#pragma omp for':
                // ningun thread empieza el siguiente tile hasta que todos
                // terminen Phase 2 del tile actual. Esto es correcto.
            }

            // Aplicar escala global (una vez, paralelo sobre d_out)
            #pragma omp for schedule(static) nowait
            for (int j = 0; j < d_out; j++) {
                outm[j] *= w_scale;
            }
        } // fin omp parallel
    } // fin bucle M
}


// ---------------------------------------------------------------------------
// fp32_gemv_avx2
//
// GEMV FP32 optimizado para modelos FP16/FP32 (Ablation runs sin BitLinear).
// W layout [d_out, d_in] Row-Major (igual que nn.Linear.weight en PyTorch
// cuando se llama con weight.T).
//
// Para llamar desde Python:
//   lib.fp32_gemv_avx2(x_ptr, W_ptr, out_ptr, M, d_in, d_out)
//
// Optimizaciones:
//   - 4x unroll en j (4 filas de W simultaneas → 4x ILP)
//   - Inner AVX2 FMA (8 FP32 por instruccion)
//   - Reduccion horizontal con _mm256_hadd_ps
// ---------------------------------------------------------------------------
EXPORT void fp32_gemv_avx2(
    const float* x,       // [M, d_in]
    const float* W,       // [d_out, d_in] Row-Major
    float*       output,  // [M, d_out]
    int M, int d_in, int d_out
) {
    for (int m = 0; m < M; m++) {
        const float* xm   = x      + m * d_in;
        float*       outm = output + m * d_out;

        #pragma omp parallel for schedule(static)
        for (int j = 0; j < d_out - 3; j += 4) {
            const float* w0 = W + j * d_in;
            const float* w1 = w0 + d_in;
            const float* w2 = w1 + d_in;
            const float* w3 = w2 + d_in;

            __m256 vs0 = _mm256_setzero_ps();
            __m256 vs1 = _mm256_setzero_ps();
            __m256 vs2 = _mm256_setzero_ps();
            __m256 vs3 = _mm256_setzero_ps();

            int i = 0;
            for (; i <= d_in - 8; i += 8) {
                __m256 vx = _mm256_loadu_ps(xm + i);
                vs0 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(w0 + i), vs0);
                vs1 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(w1 + i), vs1);
                vs2 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(w2 + i), vs2);
                vs3 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(w3 + i), vs3);
            }

            float s0 = _hsum256_ps(vs0);
            float s1 = _hsum256_ps(vs1);
            float s2 = _hsum256_ps(vs2);
            float s3 = _hsum256_ps(vs3);

            for (; i < d_in; i++) {
                s0 += xm[i] * w0[i];
                s1 += xm[i] * w1[i];
                s2 += xm[i] * w2[i];
                s3 += xm[i] * w3[i];
            }

            outm[j    ] = s0;
            outm[j + 1] = s1;
            outm[j + 2] = s2;
            outm[j + 3] = s3;
        }

        // Tail: salidas restantes (cuando d_out no es multiplo de 4)
        int j_tail = (d_out / 4) * 4;
        for (int j = j_tail; j < d_out; j++) {
            const float* wj = W + j * d_in;
            __m256 vs = _mm256_setzero_ps();
            int i = 0;
            for (; i <= d_in - 8; i += 8) {
                vs = _mm256_fmadd_ps(
                    _mm256_loadu_ps(xm + i),
                    _mm256_loadu_ps(wj + i), vs
                );
            }
            float s = _hsum256_ps(vs);
            for (; i < d_in; i++) s += xm[i] * wj[i];
            outm[j] = s;
        }
    }
}


// ---------------------------------------------------------------------------
// mlgru_scan_fusion
//
// MLGRU state update con AVX2 FMA explicito.
// Reemplaza '#pragma omp parallel for simd' que no vectoriza en MinGW/MSVC.
//
// Operacion: h_new = c + f * (h - c)   [lerp]
//            o     = g * h_new
//
// Input:
//   c  [B, d_model]  candidate
//   f  [B, d_model]  forget gate (0-1)
//   g  [B, d_model]  output gate
//   h  [B, d_model]  estado previo (modificado IN-PLACE)
//   o  [B, d_model]  output final
// ---------------------------------------------------------------------------
EXPORT void mlgru_scan_fusion(
    const float* c,   // [B, d_model] candidate
    const float* f,   // [B, d_model] forget gate
    const float* g_gate,  // [B, d_model] output gate (renombrado para no colisionar con g anterior)
    float*       h,   // [B, d_model] estado IN/OUT
    float*       o,   // [B, d_model] output
    int B, int d_model
) {
    int total = B * d_model;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i <= total - 8; i += 8) {
        __m256 vc  = _mm256_loadu_ps(c      + i);
        __m256 vf  = _mm256_loadu_ps(f      + i);
        __m256 vh  = _mm256_loadu_ps(h      + i);
        __m256 vg  = _mm256_loadu_ps(g_gate + i);

        // lerp: c + f * (h - c)  == fmadd(f, h-c, c)
        __m256 h_new = _mm256_fmadd_ps(vf, _mm256_sub_ps(vh, vc), vc);

        _mm256_storeu_ps(h + i, h_new);
        _mm256_storeu_ps(o + i, _mm256_mul_ps(vg, h_new));
    }

    // Tail escalar para elementos restantes (si total no es multiplo de 8)
    int tail_start = (total / 8) * 8;
    for (int i = tail_start; i < total; i++) {
        float h_new = c[i] + f[i] * (h[i] - c[i]);
        h[i] = h_new;
        o[i] = g_gate[i] * h_new;
    }
}

} // extern "C"
