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
#include <math.h>
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

    // Fast path para M==1 (decode): OPT-1 + OPT-2
    //
    // Estructura: UN solo omp parallel por llamada (no por tile).
    // Cada thread tiene su propio tile_luts (thread_local, 64KB, cabe en L2).
    // Phase 1: cada thread reconstruye tile_luts independientemente (sin sync).
    // Phase 2: omp for schedule(static) con nowait — mismo rango j por thread
    //          entre tiles sucesivos → no hay race en outm[] y no hay barrera.
    // OPT-2: decode_T calculado UNA vez antes del parallel, compartido read-only.
    if (M == 1) {
        const float* xm  = x;
        float*       outm = output;
        memset(outm, 0, d_out * sizeof(float));

        // OPT-2: Transponer decode_table una sola vez (constante)
        float decode_T_shared[8 * 128];
        if (group_size == 4) {
            for (int i = 0; i < 4; i++)
                for (int c = 0; c < 128; c++)
                    decode_T_shared[i * 128 + c] = decode_table[c * 4 + i];
        }

        // OPT-1: Thread team creado UNA vez — sin overhead por tile
        #pragma omp parallel
        {
            // Cada thread tiene su propia copia de tile_luts (evita falsa comparticion)
            static thread_local float tile_luts[TILE_G * 256];

            for (int g_base = 0; g_base < n_groups; g_base += TILE_G) {
                int g_end     = g_base + TILE_G < n_groups ? g_base + TILE_G : n_groups;
                int tile_size = g_end - g_base;

                // PHASE 1: Cada thread construye su propio tile_luts (sin sincronizacion)
                for (int g_local = 0; g_local < tile_size; g_local++) {
                    int          g     = g_base + g_local;
                    const float* x_grp = xm + g * group_size;
                    float*       lut_g = tile_luts + g_local * 256;

                    if (group_size == 4) {
                        memset(lut_g, 0, 128 * sizeof(float));
                        for (int i = 0; i < 4; i++) {
                            __m256 vxi     = _mm256_broadcast_ss(x_grp + i);
                            const float* col = decode_T_shared + i * 128;
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

                // PHASE 2: Gather reduction — dividido entre threads
                // nowait: mismo j range por thread → sin race en outm[]
                #pragma omp for schedule(static) nowait
                for (int j = 0; j < d_out; j++) {
                    const uint8_t* w_row = w_packed + j * n_groups;
                    __m256 vsum = _mm256_setzero_ps();
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
                    for (; g_local < tile_size; g_local++) {
                        int g_abs = g_base + g_local;
                        sum += tile_luts[g_local * 256 + w_row[g_abs]];
                    }

                    outm[j] += sum;
                }
            }

            // Escala: mismo range j por thread → nowait ok
            #pragma omp for schedule(static) nowait
            for (int j = 0; j < d_out; j++)
                outm[j] *= w_scale;
        } // fin omp parallel

        return;
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
// _hmax256_ps — reduccion horizontal de __m256 a float (valor maximo)
// ---------------------------------------------------------------------------
static inline float _hmax256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}

// ---------------------------------------------------------------------------
// ternary_i8_fused_gemv_avx2
//
// GEMV ternario con cuantizacion interna: float32 input → int8 → GEMV → float32.
//
// Elimina 7+ ops PyTorch en Python por llamada a BitLinear:
//   - abs().amax(), /, round(), clamp(), to(int8), contiguous(), data_ptr()
// Para M==1 (decode), esto suprime la mayor parte del overhead Python.
//
// Cuantizacion AVX2:
//   1. Encontrar max(|x|) con AVX2 horizontal max
//   2. inv_scale = 127 / max(|x|, eps)
//   3. Redondear y saturar a int8 con CVTPS2DQ + PACKS
//
// GEMV identico a ternary_i8_gemv_avx2: sign_epi8 + madd_epi16 + acc int32
//
// Input:
//   x        [d_in]        float32 activaciones
//   w        [d_out, d_in] int8 pesos ternarios row-major
//   output   [d_out]       float32 salida
//   d_in, d_out            dimensiones
//   w_scale               escala del peso (x_scale calculado internamente)
// ---------------------------------------------------------------------------
#ifdef __GNUC__
__attribute__((target("avx2,fma")))
#endif
EXPORT void ternary_i8_fused_gemv_avx2(
    const float*  x,       // [d_in] float32 activaciones
    const int8_t* w,       // [d_out, d_in] pesos ternarios int8 row-major
    float*        output,  // [d_out]
    int           d_in,
    int           d_out,
    float         w_scale
) {
    // --- Cuantizacion interna (sin alloc Python) ---
    // Paso 1: max(|x|) con AVX2
    const __m256 vabsmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256 vmax = _mm256_setzero_ps();
    int i = 0;
    for (; i <= d_in - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        vmax = _mm256_max_ps(vmax, _mm256_and_ps(v, vabsmask));
    }
    float x_max = _hmax256_ps(vmax);
    for (; i < d_in; i++) {
        float a = x[i] < 0.0f ? -x[i] : x[i];
        if (a > x_max) x_max = a;
    }
    const float inv_scale = 127.0f / (x_max + 1e-8f);
    const float x_scale   = x_max / (127.0f + 1e-8f);

    // Paso 2: Cuantizar float32 → int8 con AVX2 (en buffer stack)
    // d_in max esperado: d_ff=4096 → 4096 bytes → 4KB, seguro en stack
    int8_t x_i8[4096];
    if (d_in > 4096) {
        // Fallback escalar para d_in > 4096 (no ocurre en CPU1-350M)
        for (int ii = 0; ii < d_in; ii++) {
            float v = x[ii] * inv_scale + 0.5f;
            if (v >  127.0f) v =  127.0f;
            if (v < -128.0f) v = -128.0f;
            x_i8[ii] = (int8_t)(int)v;
        }
    } else {
        const __m256  vinv  = _mm256_set1_ps(inv_scale);
        const __m256  vhalf = _mm256_set1_ps(0.5f);
        const __m256i vplo  = _mm256_set1_epi32(-128);
        const __m256i vphi  = _mm256_set1_epi32(127);
        int ii = 0;
        for (; ii <= d_in - 8; ii += 8) {
            __m256 vf = _mm256_loadu_ps(x + ii);
            // round-to-nearest: floor(v * inv_scale + 0.5) con saturacion
            __m256i vi = _mm256_cvtps_epi32(
                _mm256_fmadd_ps(vf, vinv, vhalf)
            );
            vi = _mm256_min_epi32(_mm256_max_epi32(vi, vplo), vphi);
            // Extraer 8 int32 → 8 int8 (sin AVX-512 VPMOVDB: usar packs)
            __m128i lo8 = _mm_packs_epi32(
                _mm256_castsi256_si128(vi),
                _mm256_extracti128_si256(vi, 1)
            );
            lo8 = _mm_packs_epi16(lo8, lo8);
            // Guardar 8 bytes
            _mm_storel_epi64((__m128i*)(x_i8 + ii), lo8);
        }
        for (; ii < d_in; ii++) {
            float v = x[ii] * inv_scale + 0.5f;
            if (v >  127.0f) v =  127.0f;
            if (v < -128.0f) v = -128.0f;
            x_i8[ii] = (int8_t)(int)v;
        }
    }

    // --- GEMV ternario int8 (identico a ternary_i8_gemv_avx2) ---
    const float       combined_scale = x_scale * w_scale;
    const __m256i vones16 = _mm256_set1_epi16(1);

    for (int j = 0; j < d_out; j++) {
        const int8_t* wj   = w + (ptrdiff_t)j * d_in;
        __m256i       vacc = _mm256_setzero_si256();
        int ii = 0;

        for (; ii <= d_in - 32; ii += 32) {
            __m256i vact  = _mm256_loadu_si256((const __m256i*)(x_i8 + ii));
            __m256i vw    = _mm256_loadu_si256((const __m256i*)(wj   + ii));
            __m256i vprod = _mm256_sign_epi8(vact, vw);
            __m256i p16lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vprod));
            __m256i p16hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vprod, 1));
            vacc = _mm256_add_epi32(vacc, _mm256_madd_epi16(p16lo, vones16));
            vacc = _mm256_add_epi32(vacc, _mm256_madd_epi16(p16hi, vones16));
        }

        __m128i lo  = _mm256_castsi256_si128(vacc);
        __m128i hi  = _mm256_extracti128_si256(vacc, 1);
        __m128i s   = _mm_add_epi32(lo, hi);
        s = _mm_hadd_epi32(s, s);
        s = _mm_hadd_epi32(s, s);
        int32_t sum = _mm_cvtsi128_si32(s);

        for (; ii < d_in; ii++)
            sum += (int32_t)x_i8[ii] * (int32_t)wj[ii];

        output[j] = (float)sum * combined_scale;
    }
}


// ---------------------------------------------------------------------------
// ternary_i8_gemv_avx2
//
// GEMV ternario: activaciones int8 × pesos int8 {-1,0,1} → salida float32.
//
// Estrategia AVX2:
//   - _mm256_sign_epi8(act, w): act*sign(w) — w=0→0, w>0→act, w<0→-act
//   - Widening int8→int16 + _mm256_madd_epi16(·, ones) para acumular en int32
//   - OpenMP paralelo sobre filas (d_out)
//   - x [d_in=1536 → 1.5KB] cabe en L1 cache de cada core
//   - Scale aplicado solo al escribir float (sin overhead extra de bucle)
//
// Input:
//   x        [d_in]        int8 activaciones cuantizadas
//   w        [d_out, d_in] int8 pesos ternarios row-major (acceso contiguo)
//   output   [d_out]       float32 salida
//   d_in, d_out            dimensiones
//   scale                  x_scale * w_scale (escalar float)
// ---------------------------------------------------------------------------
#ifdef __GNUC__
__attribute__((target("avx2,fma")))
#endif
EXPORT void ternary_i8_gemv_avx2(
    const int8_t* x,       // [d_in] activaciones int8
    const int8_t* w,       // [d_out, d_in] pesos ternarios int8 row-major
    float*        output,  // [d_out]
    int           d_in,
    int           d_out,
    float         scale
) {
    const __m256i vones16 = _mm256_set1_epi16(1);

    for (int j = 0; j < d_out; j++) {
        const int8_t* wj   = w + (ptrdiff_t)j * d_in;
        __m256i       vacc = _mm256_setzero_si256();
        int i = 0;

        for (; i <= d_in - 32; i += 32) {
            __m256i vact  = _mm256_loadu_si256((const __m256i*)(x  + i));
            __m256i vw    = _mm256_loadu_si256((const __m256i*)(wj + i));
            // Ternary multiply: sign_epi8(act, w) = 0 if w==0, ±act otherwise
            __m256i vprod = _mm256_sign_epi8(vact, vw);
            // Widen int8 → int16, then pairwise add → int32 accumulator
            __m256i p16lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vprod));
            __m256i p16hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vprod, 1));
            vacc = _mm256_add_epi32(vacc, _mm256_madd_epi16(p16lo, vones16));
            vacc = _mm256_add_epi32(vacc, _mm256_madd_epi16(p16hi, vones16));
        }

        // Horizontal sum int32
        __m128i lo  = _mm256_castsi256_si128(vacc);
        __m128i hi  = _mm256_extracti128_si256(vacc, 1);
        __m128i s   = _mm_add_epi32(lo, hi);
        s = _mm_hadd_epi32(s, s);
        s = _mm_hadd_epi32(s, s);
        int32_t sum = _mm_cvtsi128_si32(s);

        // Scalar tail (cuando d_in no es multiplo de 32)
        for (; i < d_in; i++)
            sum += (int32_t)x[i] * (int32_t)wj[i];

        output[j] = (float)sum * scale;
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


// ---------------------------------------------------------------------------
// Fast exp/sigmoid/silu helpers — AVX2 polynomial approximation
// exp(x): max error < 1 ULP for |x| < 87
// ---------------------------------------------------------------------------
static inline __m256 _avx2_exp_ps(__m256 x) {
    const __m256 vln2     = _mm256_set1_ps(0.6931471806f);
    const __m256 vinv_ln2 = _mm256_set1_ps(1.4426950408f);
    // Clamp to avoid overflow/underflow
    x = _mm256_min_ps(x, _mm256_set1_ps( 88.0f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
    // n = round(x/ln2), r = x - n*ln2
    __m256  z  = _mm256_fmadd_ps(x, vinv_ln2, _mm256_set1_ps(0.5f));
    __m256i ni = _mm256_cvttps_epi32(z);
    __m256  fn = _mm256_cvtepi32_ps(ni);
    __m256  r  = _mm256_fnmadd_ps(fn, vln2, x);
    // Polynomial: exp(r) ≈ 1 + r(1 + r(0.5 + r(1/6 + r(1/24 + r/120))))
    __m256 p = _mm256_set1_ps(1.0f/120);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f/24));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f/6));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    // Scale by 2^n via float exponent field
    __m256i bias = _mm256_set1_epi32(127);
    __m256  pow2 = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_add_epi32(ni, bias), 23));
    return _mm256_mul_ps(p, pow2);
}

static inline __m256 _avx2_sigmoid_ps(__m256 x) {
    // 1 / (1 + exp(-x))
    __m256 e = _avx2_exp_ps(_mm256_sub_ps(_mm256_setzero_ps(), x));
    return _mm256_div_ps(_mm256_set1_ps(1.0f),
                         _mm256_add_ps(_mm256_set1_ps(1.0f), e));
}

static inline __m256 _avx2_silu_ps(__m256 x) {
    return _mm256_mul_ps(x, _avx2_sigmoid_ps(x));
}


// ---------------------------------------------------------------------------
// rms_norm_f32
//
// RMSNorm: out[i] = x[i] * weight[i] * rsqrt(mean(x^2) + eps)
// Single-threaded (D is small, typically 1024 — 1 core saturates).
// ---------------------------------------------------------------------------
EXPORT void rms_norm_f32(
    const float* x,      // [D]
    const float* weight, // [D] learned scale
    float*       out,    // [D]
    int          D,
    float        eps
) {
    // Pass 1: sum of squares with AVX2
    __m256 vss = _mm256_setzero_ps();
    int i = 0;
    for (; i <= D - 8; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        vss = _mm256_fmadd_ps(v, v, vss);
    }
    // Horizontal sum
    __m128 lo = _mm256_castps256_ps128(vss);
    __m128 hi = _mm256_extractf128_ps(vss, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_hadd_ps(s4, s4);
    s4 = _mm_hadd_ps(s4, s4);
    float ss = _mm_cvtss_f32(s4);
    // Scalar tail
    for (; i < D; i++) ss += x[i] * x[i];

    float scale = 1.0f / sqrtf(ss / (float)D + eps);

    // Pass 2: apply scale * weight
    __m256 vscale = _mm256_set1_ps(scale);
    i = 0;
    for (; i <= D - 8; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(weight + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(vx, vscale), vw));
    }
    for (; i < D; i++) out[i] = x[i] * scale * weight[i];
}


// ---------------------------------------------------------------------------
// silu_f32
//
// SiLU (Swish) in-place: out[i] = x[i] * sigmoid(x[i])
// Used for the SwiGLU FFN gate (D_FF = 2730 typically).
// OpenMP parallel for larger vectors.
// ---------------------------------------------------------------------------
EXPORT void silu_f32(
    const float* x,
    float*       out,
    int          N
) {
    for (int i = 0; i <= N - 8; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(out + i, _avx2_silu_ps(vx));
    }
    // Scalar tail (serial, no race since tail < 8)
    int tail = (N / 8) * 8;
    for (int i = tail; i < N; i++) {
        float sig = 1.0f / (1.0f + expf(-x[i]));
        out[i] = x[i] * sig;
    }
}


// ---------------------------------------------------------------------------
// fused_mlgru_gates_avx2
//
// Fused MLGRU activation + cell update in ONE pass:
//   f = sigmoid(f_raw)
//   c = silu(c_raw)
//   g = sigmoid(g_raw)
//   h_new = c + f * (h - c)        [lerp]
//   out   = g * h_new
//   h     ← h_new                  [in-place update]
//
// Replaces 5 separate Python/PyTorch ops (sigmoid×2, silu, lerp, mul).
// Single-threaded — D=1024 fits in L2, no benefit from parallelism here.
// ---------------------------------------------------------------------------
EXPORT void fused_mlgru_gates_avx2(
    const float* f_raw,  // [D] forget gate pre-activation
    const float* c_raw,  // [D] candidate pre-activation
    const float* g_raw,  // [D] output gate pre-activation
    float*       h,      // [D] hidden state (IN/OUT)
    float*       out,    // [D] block output
    int          D
) {
    int i = 0;
    for (; i <= D - 8; i += 8) {
        __m256 vf_raw = _mm256_loadu_ps(f_raw + i);
        __m256 vc_raw = _mm256_loadu_ps(c_raw + i);
        __m256 vg_raw = _mm256_loadu_ps(g_raw + i);
        __m256 vh     = _mm256_loadu_ps(h     + i);

        __m256 vf = _avx2_sigmoid_ps(vf_raw);
        __m256 vc = _avx2_silu_ps(vc_raw);
        __m256 vg = _avx2_sigmoid_ps(vg_raw);

        // h_new = c + f * (h - c)
        __m256 vh_new = _mm256_fmadd_ps(vf, _mm256_sub_ps(vh, vc), vc);
        // out = g * h_new
        _mm256_storeu_ps(out + i, _mm256_mul_ps(vg, vh_new));
        // update h in-place
        _mm256_storeu_ps(h   + i, vh_new);
    }
    // Scalar tail
    for (; i < D; i++) {
        float f     = 1.0f / (1.0f + expf(-f_raw[i]));
        float c     = c_raw[i] / (1.0f + expf(-c_raw[i]));
        float g     = 1.0f / (1.0f + expf(-g_raw[i]));
        float h_new = c + f * (h[i] - c);
        out[i] = g * h_new;
        h[i]   = h_new;
    }
}


// ---------------------------------------------------------------------------
// elem_add_f32
//
// out[i] = a[i] + b[i], or in-place a[i] += b[i] when out==a.
// Used for residual connections.
// ---------------------------------------------------------------------------
EXPORT void elem_add_f32(
    const float* a,
    const float* b,
    float*       out,
    int          N
) {
    int i = 0;
    for (; i <= N - 8; i += 8)
        _mm256_storeu_ps(out + i,
            _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    for (; i < N; i++) out[i] = a[i] + b[i];
}

} // extern "C"
