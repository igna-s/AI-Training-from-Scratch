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
 *   - Phase 1 AVX2 SAXPY: decode_T transpuesto [4,128] → 4 pasadas FMA, acceso contiguo
 *   - Phase 2 2x j-unroll: 2 filas de w_packed simultaneas → 2x ILP en gather loop
 *   - AVX2 SIMD gather con _mm256_i32gather_ps (8 lookups en paralelo)
 *   - Prefetch explicit _MM_HINT_T0 para siguiente bloque de tile
 *   - AVX2 FMA explicito en mlgru_scan_fusion (fuerza vectorizacion en Windows/MinGW)
 *   - fp32_gemv_avx2: 8×unroll FMA con reduccion horizontal AVX
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
//   3. Phase 1 SAXPY: decode_T[group_size,128] transpuesto — acceso secuencial
//      sin stride, 4 pasadas FMA sobre 128 floats → ~9x vs scatter escalar.
//   4. Phase 2 2x j-unroll: dos filas de w_packed procesadas por iteracion
//      → dos gather independientes, ILP maxima en OOO hardware.
//   5. Prefetch _MM_HINT_T0 para siguiente bloque dentro del tile.
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

    // Fast path para M==1 (comun en generacion): evita overhead de thread team
    if (M == 1) {
        const float* xm  = x;
        float*       outm = output;
        memset(outm, 0, d_out * sizeof(float));

        // Single-threaded computation
        static thread_local float tile_luts[TILE_G * 256];
        static thread_local float decode_T[8 * 128];

        for (int g_base = 0; g_base < n_groups; g_base += TILE_G) {
            int g_end     = g_base + TILE_G < n_groups ? g_base + TILE_G : n_groups;
            int tile_size = g_end - g_base;

            // FASE 1: LUT computation (same as parallel path)
            for (int g_local = 0; g_local < tile_size; g_local++) {
                int          g     = g_base + g_local;
                const float* x_grp = xm + g * group_size;
                float*       lut_g = tile_luts + g_local * 256;

                if (group_size == 4) {
                    for (int i = 0; i < 4; i++) {
                        for (int c = 0; c < 128; c++) {
                            decode_T[i * 128 + c] = decode_table[c * 4 + i];
                        }
                    }
                    memset(lut_g, 0, 128 * sizeof(float));
                    for (int i = 0; i < 4; i++) {
                        __m256 vxi     = _mm256_broadcast_ss(x_grp + i);
                        const float* col = decode_T + i * 128;
                        for (int c = 0; c < 128; c += 8) {
                            __m256 vlut = _mm256_loadu_ps(lut_g + c);
                            __m256 vcol = _mm256_loadu_ps(col   + c);
                            _mm256_storeu_ps(lut_g + c,
                                _mm256_fmadd_ps(vxi, vcol, vlut));
                        }
                    }
                    const __m256 vneg = _mm256_set1_ps(-1.0f);
                    for (int c = 0; c < 128; c += 8) {
                        _mm256_storeu_ps(lut_g + 128 + c,
                            _mm256_mul_ps(vneg, _mm256_loadu_ps(lut_g + c)));
                    }
                }
            }

            // FASE 2: Gather reduction (single-threaded, no #pragma omp for)
            for (int j = 0; j < d_out - 1; j += 2) {
                const uint8_t* w_row0 = w_packed + j       * n_groups;
                const uint8_t* w_row1 = w_packed + (j + 1) * n_groups;

                __m256 vsum0 = _mm256_setzero_ps();
                __m256 vsum1 = _mm256_setzero_ps();
                __m256i vbase = _mm256_set_epi32(
                    7*256, 6*256, 5*256, 4*256, 3*256, 2*256, 1*256, 0
                );
                const __m256i vstep = _mm256_set1_epi32(8 * 256);

                int g_local = 0;
                for (; g_local <= tile_size - 8; g_local += 8) {
                    int g_abs = g_base + g_local;
                    __m128i w8_0  = _mm_loadl_epi64((const __m128i*)(w_row0 + g_abs));
                    __m128i w8_1  = _mm_loadl_epi64((const __m128i*)(w_row1 + g_abs));
                    __m256i w32_0 = _mm256_cvtepu8_epi32(w8_0);
                    __m256i w32_1 = _mm256_cvtepu8_epi32(w8_1);
                    __m256i vindex0 = _mm256_add_epi32(vbase, w32_0);
                    __m256i vindex1 = _mm256_add_epi32(vbase, w32_1);
                    __m256 vval0 = _mm256_i32gather_ps(tile_luts, vindex0, 4);
                    __m256 vval1 = _mm256_i32gather_ps(tile_luts, vindex1, 4);
                    vsum0 = _mm256_add_ps(vsum0, vval0);
                    vsum1 = _mm256_add_ps(vsum1, vval1);
                    vbase = _mm256_add_epi32(vbase, vstep);
                }

                float sum0 = _hsum256_ps(vsum0);
                float sum1 = _hsum256_ps(vsum1);

                for (; g_local < tile_size; g_local++) {
                    int g_abs = g_base + g_local;
                    sum0 += tile_luts[g_local * 256 + w_row0[g_abs]];
                    sum1 += tile_luts[g_local * 256 + w_row1[g_abs]];
                }

                outm[j    ] += sum0;
                outm[j + 1] += sum1;
            }

            if (d_out & 1) {
                int j = d_out - 1;
                const uint8_t* w_row = w_packed + j * n_groups;
                __m256 vsum  = _mm256_setzero_ps();
                __m256i vbase = _mm256_set_epi32(
                    7*256, 6*256, 5*256, 4*256, 3*256, 2*256, 1*256, 0
                );
                const __m256i vstep = _mm256_set1_epi32(8 * 256);
                int g_local = 0;
                for (; g_local <= tile_size - 8; g_local += 8) {
                    int g_abs = g_base + g_local;
                    __m128i w8  = _mm_loadl_epi64((const __m128i*)(w_row + g_abs));
                    __m256i w32 = _mm256_cvtepu8_epi32(w8);
                    __m256i vindex = _mm256_add_epi32(vbase, w32);
                    __m256 vval = _mm256_i32gather_ps(tile_luts, vindex, 4);
                    vsum = _mm256_add_ps(vsum, vval);
                    vbase = _mm256_add_epi32(vbase, vstep);
                }
                float sum = _hsum256_ps(vsum);
                for (; g_local < tile_size; g_local++)
                    sum += tile_luts[g_local * 256 + w_row[g_base + g_local]];
                outm[j] += sum;
            }
        }

        // Apply scale
        for (int j = 0; j < d_out; j++) {
            outm[j] *= w_scale;
        }

        return;  // Fast path done
    }

    // Slow path para M > 1: mantener codigo original con parallelization
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

            // Thread-local decode_T transpuesto para Phase 1 SAXPY
            // decode_T[i * 128 + c] = decode_table[c * group_size + i]
            // Max group_size soportado en este buffer: 8 (ampliable)
            static thread_local float decode_T[8 * 128];

            for (int g_base = 0; g_base < n_groups; g_base += TILE_G) {
                int g_end     = g_base + TILE_G < n_groups ? g_base + TILE_G : n_groups;
                int tile_size = g_end - g_base;

                // -----------------------------------------------------------
                // FASE 1: Cada thread calcula su propio tile LUT
                //
                // AVX2 SAXPY para group_size==4:
                //   a) Transponer decode_table[128,4] → decode_T[4,128] una vez
                //      por tile (barato: 512 ops, acceso contiguo al escribir).
                //   b) 4 pasadas SAXPY sobre 128 floats con FMA vectorial.
                //      lut_g += x_grp[i] * decode_T[i*128 .. i*128+127]
                //   c) Mirror negado: lut_g[128..255] = -lut_g[0..127]
                //
                // Ventaja frente a scatter escalar: acceso secuencial a decode_T
                // (L1 friendly, sin stride) vs acceso con stride group_size a
                // decode_table. ~9x menos cache miss en Phase 1.
                // -----------------------------------------------------------
                for (int g_local = 0; g_local < tile_size; g_local++) {
                    int          g     = g_base + g_local;
                    const float* x_grp = xm + g * group_size;
                    float*       lut_g = tile_luts + g_local * 256;

                    if (group_size == 4) {
                        // Transponer decode_table[128,4] → decode_T[4,128]
                        // Stride original: group_size=4 → decode_table[c*4+i]
                        // Stride transpuesto: 1 → decode_T[i*128+c] (contiguo)
                        for (int i = 0; i < 4; i++) {
                            for (int c = 0; c < 128; c++) {
                                decode_T[i * 128 + c] = decode_table[c * 4 + i];
                            }
                        }

                        // Inicializar lut_g[0..127] a cero
                        memset(lut_g, 0, 128 * sizeof(float));

                        // 4 pasadas SAXPY: lut_g += x_grp[i] * decode_T[i,:]
                        for (int i = 0; i < 4; i++) {
                            __m256 vxi     = _mm256_broadcast_ss(x_grp + i);
                            const float* col = decode_T + i * 128;
                            for (int c = 0; c < 128; c += 8) {
                                __m256 vlut = _mm256_loadu_ps(lut_g + c);
                                __m256 vcol = _mm256_loadu_ps(col   + c);
                                _mm256_storeu_ps(lut_g + c,
                                    _mm256_fmadd_ps(vxi, vcol, vlut));
                            }
                        }

                        // Mirror negado: lut_g[128..255] = -lut_g[0..127]
                        const __m256 vneg = _mm256_set1_ps(-1.0f);
                        for (int c = 0; c < 128; c += 8) {
                            _mm256_storeu_ps(lut_g + 128 + c,
                                _mm256_mul_ps(vneg, _mm256_loadu_ps(lut_g + c)));
                        }
                    } else {
                        // Camino generico: transponer para group_size arbitrario
                        int gs = group_size < 8 ? group_size : 8;
                        for (int i = 0; i < gs; i++)
                            for (int c = 0; c < 128; c++)
                                decode_T[i * 128 + c] = decode_table[c * group_size + i];

                        memset(lut_g, 0, 128 * sizeof(float));
                        for (int i = 0; i < gs; i++) {
                            __m256 vxi     = _mm256_broadcast_ss(x_grp + i);
                            const float* col = decode_T + i * 128;
                            for (int c = 0; c < 128; c += 8) {
                                __m256 vlut = _mm256_loadu_ps(lut_g + c);
                                _mm256_storeu_ps(lut_g + c,
                                    _mm256_fmadd_ps(vxi, _mm256_loadu_ps(col + c), vlut));
                            }
                        }
                        // Elementos restantes si group_size > 8 (camino escalar)
                        for (int i = gs; i < group_size; i++) {
                            float xi = x_grp[i];
                            for (int c = 0; c < 128; c++)
                                lut_g[c] += xi * decode_table[c * group_size + i];
                        }
                        const __m256 vneg = _mm256_set1_ps(-1.0f);
                        for (int c = 0; c < 128; c += 8)
                            _mm256_storeu_ps(lut_g + 128 + c,
                                _mm256_mul_ps(vneg, _mm256_loadu_ps(lut_g + c)));
                    }
                }

                // -----------------------------------------------------------
                // FASE 2: Gather reduction — 2x j-unroll para ILP maxima
                //
                // Invariante de scheduling: schedule(static) garantiza que
                // el mismo thread siempre maneja el mismo rango de j.
                // → outm[j] es escrito por exactamente un thread → sin race.
                //
                // 2x unroll: j y j+1 comparten vbase/vstep → mitad de
                // instrucciones de control de loop, dos gather por ciclo.
                // -----------------------------------------------------------
                #pragma omp for schedule(static)
                for (int j = 0; j < d_out - 1; j += 2) {
                    const uint8_t* w_row0 = w_packed + j       * n_groups;
                    const uint8_t* w_row1 = w_packed + (j + 1) * n_groups;

                    __m256 vsum0 = _mm256_setzero_ps();
                    __m256 vsum1 = _mm256_setzero_ps();
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

                        // Cargar 8 indices uint8 de cada fila y extender a int32
                        __m128i w8_0  = _mm_loadl_epi64((const __m128i*)(w_row0 + g_abs));
                        __m128i w8_1  = _mm_loadl_epi64((const __m128i*)(w_row1 + g_abs));
                        __m256i w32_0 = _mm256_cvtepu8_epi32(w8_0);
                        __m256i w32_1 = _mm256_cvtepu8_epi32(w8_1);

                        // vindex[k] = (g_local + k) * 256 + w_row[g_abs + k]
                        __m256i vindex0 = _mm256_add_epi32(vbase, w32_0);
                        __m256i vindex1 = _mm256_add_epi32(vbase, w32_1);

                        // Gather: 8 floats de tile_luts para cada fila
                        __m256 vval0 = _mm256_i32gather_ps(tile_luts, vindex0, 4);
                        __m256 vval1 = _mm256_i32gather_ps(tile_luts, vindex1, 4);
                        vsum0 = _mm256_add_ps(vsum0, vval0);
                        vsum1 = _mm256_add_ps(vsum1, vval1);

                        vbase = _mm256_add_epi32(vbase, vstep);
                    }

                    // Reduccion horizontal
                    float sum0 = _hsum256_ps(vsum0);
                    float sum1 = _hsum256_ps(vsum1);

                    // Tail escalar para grupos restantes (< 8)
                    for (; g_local < tile_size; g_local++) {
                        int g_abs = g_base + g_local;
                        sum0 += tile_luts[g_local * 256 + w_row0[g_abs]];
                        sum1 += tile_luts[g_local * 256 + w_row1[g_abs]];
                    }

                    // Acumular (el mismo thread siempre escribe estos j)
                    outm[j    ] += sum0;
                    outm[j + 1] += sum1;
                }

                // Tail: si d_out es impar, procesar ultimo j en solitario
                #pragma omp single nowait
                if (d_out & 1) {
                    int j = d_out - 1;
                    const uint8_t* w_row = w_packed + j * n_groups;
                    __m256 vsum  = _mm256_setzero_ps();
                    __m256i vbase = _mm256_set_epi32(
                        7*256, 6*256, 5*256, 4*256, 3*256, 2*256, 1*256, 0
                    );
                    const __m256i vstep = _mm256_set1_epi32(8 * 256);
                    int g_local = 0;
                    for (; g_local <= tile_size - 8; g_local += 8) {
                        int g_abs = g_base + g_local;
                        __m128i w8  = _mm_loadl_epi64((const __m128i*)(w_row + g_abs));
                        __m256i w32 = _mm256_cvtepu8_epi32(w8);
                        __m256i vindex = _mm256_add_epi32(vbase, w32);
                        __m256 vval = _mm256_i32gather_ps(tile_luts, vindex, 4);
                        vsum = _mm256_add_ps(vsum, vval);
                        vbase = _mm256_add_epi32(vbase, vstep);
                    }
                    float sum = _hsum256_ps(vsum);
                    for (; g_local < tile_size; g_local++)
                        sum += tile_luts[g_local * 256 + w_row[g_base + g_local]];
                    outm[j] += sum;
                }
                // Barrera implicita al final de '#pragma omp for':
                // ningun thread empieza el siguiente tile hasta que todos
                // terminen Phase 2 del tile actual.
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
//   - 8x unroll en j (8 filas de W simultaneas → 8x ILP)
//   - Inner AVX2 FMA (8 FP32 por instruccion)
//   - Reduccion horizontal con _hsum256_ps
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

        // 8x unroll: procesar 8 filas de W por iteracion
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < d_out - 7; j += 8) {
            const float* w0 = W + j * d_in;
            const float* w1 = w0 + d_in;
            const float* w2 = w1 + d_in;
            const float* w3 = w2 + d_in;
            const float* w4 = w3 + d_in;
            const float* w5 = w4 + d_in;
            const float* w6 = w5 + d_in;
            const float* w7 = w6 + d_in;

            __m256 vs0 = _mm256_setzero_ps();
            __m256 vs1 = _mm256_setzero_ps();
            __m256 vs2 = _mm256_setzero_ps();
            __m256 vs3 = _mm256_setzero_ps();
            __m256 vs4 = _mm256_setzero_ps();
            __m256 vs5 = _mm256_setzero_ps();
            __m256 vs6 = _mm256_setzero_ps();
            __m256 vs7 = _mm256_setzero_ps();

            int i = 0;
            for (; i <= d_in - 8; i += 8) {
                __m256 vx = _mm256_loadu_ps(xm + i);
                vs0 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(w0 + i), vs0);
                vs1 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(w1 + i), vs1);
                vs2 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(w2 + i), vs2);
                vs3 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(w3 + i), vs3);
                vs4 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(w4 + i), vs4);
                vs5 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(w5 + i), vs5);
                vs6 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(w6 + i), vs6);
                vs7 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(w7 + i), vs7);
            }

            float s0 = _hsum256_ps(vs0);
            float s1 = _hsum256_ps(vs1);
            float s2 = _hsum256_ps(vs2);
            float s3 = _hsum256_ps(vs3);
            float s4 = _hsum256_ps(vs4);
            float s5 = _hsum256_ps(vs5);
            float s6 = _hsum256_ps(vs6);
            float s7 = _hsum256_ps(vs7);

            for (; i < d_in; i++) {
                float xi = xm[i];
                s0 += xi * w0[i];
                s1 += xi * w1[i];
                s2 += xi * w2[i];
                s3 += xi * w3[i];
                s4 += xi * w4[i];
                s5 += xi * w5[i];
                s6 += xi * w6[i];
                s7 += xi * w7[i];
            }

            outm[j    ] = s0;
            outm[j + 1] = s1;
            outm[j + 2] = s2;
            outm[j + 3] = s3;
            outm[j + 4] = s4;
            outm[j + 5] = s5;
            outm[j + 6] = s6;
            outm[j + 7] = s7;
        }

        // Tail: salidas restantes (cuando d_out no es multiplo de 8)
        int j_tail = (d_out / 8) * 8;
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
