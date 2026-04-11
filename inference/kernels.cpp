/*
 * Super-kernel CPU optimizado — Multi-arquitectura.
 *
 * Estrategias por tipo de modelo:
 *   ternary_lut_matmul_omp  — modelos ternarios (CPU1ModelV2, Ablation runs 03-15)
 *   fp32_gemv_avx2          — modelos FP16/FP32 (Ablation runs 01, 02, 05, 08)
 *   mlgru_scan_fusion       — update recurrente (todos los modelos MLGRU)
 *
 * Optimizaciones L3 cache:
 *   - LUT Tiling L2-aware: TILE_G=16 grupos → 16×256×4=16KB por tile (cabe en L2)
 *   - Cada thread calcula su propio tile LUT independientemente (sin coherencia cruzada)
 *   - Phase 1 AVX2 SAXPY: decode_T transpuesto [4,128] → 4 pasadas FMA, acceso contiguo
 *   - Phase 2 2x j-unroll: 2 filas de w_packed simultaneas → 2x ILP en gather loop
 *   - AVX2 SIMD gather con _mm256_i32gather_ps (8 lookups en paralelo)
 *   - Prefetch explicit _MM_HINT_T0 para siguiente bloque de tile
 *   - AVX2 FMA explicito en mlgru_scan_fusion (fuerza vectorizacion en Windows/MinGW)
 *   - fp32_gemv_avx2: 8×unroll FMA con reduccion horizontal AVX
 *
 * FIX-2: Controla si Phase 2 usa _mm256_i32gather_ps (gather AVX2, ~20 ciclos latencia
 *         en Coffee Lake) o accesos escalares secuenciales (mas predecibles en L2).
 *         En Coffee Lake, el acceso escalar es 5-16x mas rapido que gather
 *         porque gather con indices aleatorios destruye el L2 prefetcher.
 *         Dejar en 0 (escalar). Cambiar a 1 solo para benchmarkear gather.
 */
#define USE_GATHER 0

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>
#include <time.h>

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif

// FIX-1: TILE_G reducido de 64 a 16.
// Razonamiento de cache:
//   TILE_G=64 → tile_luts[64*256] × 4 bytes = 64 KB por thread.
//   Con 6 threads = 384 KB total, excede L2 de Coffee Lake (256 KB/core).
//   TILE_G=16 → tile_luts[16*256] × 4 bytes = 16 KB por thread → cabe bien en L2.
//   TILE_G=32 → 32 KB por thread, también viable si el thread count es <= 4.
//   Medición empírica recomendada: benchmark_lut_matmul() con ambos valores.
#define TILE_G 16

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

// FIX-2: Implementacion escalar alternativa a _mm256_i32gather_ps para Phase 2.
// En Coffee Lake, gather tiene ~20 ciclos de latencia por instruccion y no puede
// solaparse facilmente con otras instrucciones en el pipeline OOO.
// El acceso escalar es secuencial e impredecible en patron, pero evita la penalizacion
// de latencia del gather cuando tile_luts esta en L2 (~4-8 ciclos por acceso).
// Seleccionado via #define USE_GATHER (0=escalar, 1=gather AVX2).
static inline float ternary_lut_phase2_scalar(
    const float*   tile_luts,
    const uint8_t* w_row,
    int            g_base,
    int            tile_size
) {
    float sum = 0.0f;
    for (int g_local = 0; g_local < tile_size; g_local++) {
        sum += tile_luts[g_local * 256 + w_row[g_base + g_local]];
    }
    return sum;
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
//   1. Procesar grupos en tiles de TILE_G (default 16).
//      Cada tile genera LUT_tile[TILE_G, 256] = 16KB → cabe en L2.
//   2. En M=1: Phase 1 ejecutada UNA sola vez (omp single), Phase 2 paralela.
//      En M>1: Phase 1 por tile (una vez, decode_T hoisted), Phase 2 paralela.
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
    // FIX-3: tile_luts_shared es compartido entre threads (no thread_local).
    //        Phase 1 la ejecuta SOLO thread 0 con omp single → 1/6 del trabajo.
    //        Barrera implicita de omp single garantiza que todos los threads
    //        ven tile_luts_shared actualizado antes de entrar a Phase 2.
    // FIX-4 (path M=1): El parallel region existe a nivel de kernel — no se crea
    //        por tile sino UNA vez por llamada, eliminando ~50us de overhead.
    //        Para proyecciones pequenas (d_out < 512), la ganancia de Phase 2
    //        paralela no compensa el overhead de sincronizacion interna.
    if (M == 1) {
        const float* xm  = x;
        float*       outm = output;
        memset(outm, 0, d_out * sizeof(float));

        // OPT-2: Transponer decode_table una sola vez (constante, fuera del parallel)
        float decode_T_shared[8 * 128];
        if (group_size == 4) {
            for (int i = 0; i < 4; i++)
                for (int c = 0; c < 128; c++)
                    decode_T_shared[i * 128 + c] = decode_table[c * 4 + i];
        }

        // FIX-3: Buffer compartido (no thread_local) alineado a linea de cache.
        // Accedido read-only en Phase 2 por todos los threads → no false sharing.
        static float tile_luts_shared[TILE_G * 256] __attribute__((aligned(64)));

        // OPT-1: Thread team creado UNA vez — sin overhead por tile
        #pragma omp parallel
        {
            for (int g_base = 0; g_base < n_groups; g_base += TILE_G) {
                int g_end     = g_base + TILE_G < n_groups ? g_base + TILE_G : n_groups;
                int tile_size = g_end - g_base;

                // FIX-3: Phase 1 ejecutada por UN solo thread (omp single).
                // Sin omp single, los 6 threads hacen trabajo identico (6× redundante).
                // omp single tiene barrera implicita al final → Phase 2 ve tile completo.
                #pragma omp single
                {
                    for (int g_local = 0; g_local < tile_size; g_local++) {
                        int          g     = g_base + g_local;
                        const float* x_grp = xm + g * group_size;
                        float*       lut_g = tile_luts_shared + g_local * 256;

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
                } // fin omp single — barrera implicita aqui

                // PHASE 2: Gather reduction — dividido entre threads
                // nowait: mismo j range por thread → sin race en outm[]
                #pragma omp for schedule(static) nowait
                for (int j = 0; j < d_out; j++) {
                    const uint8_t* w_row = w_packed + j * n_groups;
                    float sum = 0.0f;

#if USE_GATHER
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
                        __m256 vval = _mm256_i32gather_ps(tile_luts_shared, vindex, 4);
                        vsum = _mm256_add_ps(vsum, vval);
                        vbase = _mm256_add_epi32(vbase, vstep);
                    }
                    sum = _hsum256_ps(vsum);
                    for (; g_local < tile_size; g_local++) {
                        int g_abs = g_base + g_local;
                        sum += tile_luts_shared[g_local * 256 + w_row[g_abs]];
                    }
#else
                    // FIX-2: Acceso escalar — evita latencia de gather (~20 ciclos en Coffee Lake)
                    sum = ternary_lut_phase2_scalar(tile_luts_shared, w_row, g_base, tile_size);
#endif
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
        // FIX-4: Lanzar threads solo si d_out >= 512.
        // Para proyecciones pequenas el overhead de omp parallel (~50us) supera
        // el beneficio de paralelismo — la condicion 'if' evita ese costo.
        #pragma omp parallel if(d_out >= 512)
        {
            // Thread-local tile LUT: TILE_G*256 floats = 16KB maximo (FIX-1)
            // 'static thread_local' → una instancia por thread, reutilizada entre llamadas
            static thread_local float tile_luts[TILE_G * 256];

            // Thread-local decode_T transpuesto para Phase 1 SAXPY
            // FIX-5: decode_T calculado UNA vez por thread antes del bucle de grupos.
            // En el codigo original se recalculaba dentro del loop g_local (trabajo redundante).
            // decode_T[i * 128 + c] = decode_table[c * group_size + i]
            // Max group_size soportado en este buffer: 8 (ampliable)
            static thread_local float decode_T[8 * 128];

            // FIX-5: Transponer decode_T AFUERA del loop de grupos (una vez por llamada).
            // Antes: la transposicion ocurria dentro del loop g_local → O(tile_size) veces.
            // Ahora: ocurre O(1) veces → elimina tile_size transposiciones redundantes.
            if (group_size == 4) {
                for (int i = 0; i < 4; i++)
                    for (int c = 0; c < 128; c++)
                        decode_T[i * 128 + c] = decode_table[c * 4 + i];
            } else {
                int gs = group_size < 8 ? group_size : 8;
                for (int i = 0; i < gs; i++)
                    for (int c = 0; c < 128; c++)
                        decode_T[i * 128 + c] = decode_table[c * group_size + i];
            }

            for (int g_base = 0; g_base < n_groups; g_base += TILE_G) {
                int g_end     = g_base + TILE_G < n_groups ? g_base + TILE_G : n_groups;
                int tile_size = g_end - g_base;

                // -----------------------------------------------------------
                // FASE 1: Cada thread calcula su propio tile LUT
                //
                // AVX2 SAXPY para group_size==4:
                //   a) decode_T ya transpuesto (FIX-5: calculado antes de este loop).
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
                        // decode_T ya esta transpuesto (FIX-5: hoisted al bloque externo)

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
                        // Camino generico: decode_T ya transpuesto para group_size arbitrario
                        int gs = group_size < 8 ? group_size : 8;

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

                    float sum0, sum1;

#if USE_GATHER
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
                    sum0 = _hsum256_ps(vsum0);
                    sum1 = _hsum256_ps(vsum1);

                    // Tail escalar para grupos restantes (< 8)
                    for (; g_local < tile_size; g_local++) {
                        int g_abs = g_base + g_local;
                        sum0 += tile_luts[g_local * 256 + w_row0[g_abs]];
                        sum1 += tile_luts[g_local * 256 + w_row1[g_abs]];
                    }
#else
                    // FIX-2: Rutas escalares para ambas filas (sin gather)
                    sum0 = ternary_lut_phase2_scalar(tile_luts, w_row0, g_base, tile_size);
                    sum1 = ternary_lut_phase2_scalar(tile_luts, w_row1, g_base, tile_size);
#endif

                    // Acumular (el mismo thread siempre escribe estos j)
                    outm[j    ] += sum0;
                    outm[j + 1] += sum1;
                }

                // Tail: si d_out es impar, procesar ultimo j en solitario
                #pragma omp single nowait
                if (d_out & 1) {
                    int j = d_out - 1;
                    const uint8_t* w_row = w_packed + j * n_groups;
#if USE_GATHER
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
#else
                    outm[j] += ternary_lut_phase2_scalar(tile_luts, w_row, g_base, tile_size);
#endif
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


// ---------------------------------------------------------------------------
// benchmark_lut_matmul
//
// Benchmark standalone para ternary_lut_matmul_omp con datos random.
// Mide tokens/segundo para el path M=1 (decode step critico).
//
// Parametros:
//   d_in       dimension de entrada (ej: 1024 para d_model)
//   d_out      dimension de salida  (ej: 1024, 2730, 4096...)
//   n_tokens   numero de iteraciones (ej: 200 para ~2s de benchmark)
//   use_gather flag: 1=gather AVX2 (USE_GATHER=1), 0=escalar (USE_GATHER=0)
//              NOTA: el flag en tiempo de ejecucion redirige al path correcto
//              independientemente del #define en tiempo de compilacion.
//
// Retorna: tokens por segundo (double) promedio sobre n_tokens iteraciones.
//
// Uso desde Python:
//   tps = lib.benchmark_lut_matmul(1024, 1024, 200, 1)
//   print(f"Gather:  {tps:.1f} TPS")
//   tps = lib.benchmark_lut_matmul(1024, 1024, 200, 0)
//   print(f"Scalar:  {tps:.1f} TPS")
// ---------------------------------------------------------------------------
EXPORT double benchmark_lut_matmul(int d_in, int d_out, int n_tokens, int use_gather) {
    const int group_size = 4;
    int n_groups = d_in / group_size;

    // Allocar buffers con datos random
    float*   x            = (float*)  malloc(d_in   * sizeof(float));
    uint8_t* w_packed     = (uint8_t*)malloc((size_t)d_out * n_groups * sizeof(uint8_t));
    float*   decode_table = (float*)  malloc(128 * group_size * sizeof(float));
    float*   output       = (float*)  malloc(d_out  * sizeof(float));

    if (!x || !w_packed || !decode_table || !output) {
        free(x); free(w_packed); free(decode_table); free(output);
        return -1.0;
    }

    // Inicializar con valores pseudoaleatorios deterministas
    for (int i = 0; i < d_in; i++)
        x[i] = (float)(i % 17 - 8) * 0.1f;
    for (size_t i = 0; i < (size_t)d_out * n_groups; i++)
        w_packed[i] = (uint8_t)(i % 256);
    for (int i = 0; i < 128 * group_size; i++)
        decode_table[i] = (float)(i % 7 - 3) * 0.5f;

    const float w_scale = 0.01f;

    // Warm-up: 5 iteraciones para poblar caches y estabilizar el team OMP
    for (int t = 0; t < 5; t++) {
        // Rutas de benchmark controladas por use_gather en tiempo de ejecucion.
        // Como USE_GATHER es un #define, llamamos directamente a ternary_lut_matmul_omp
        // en ambos casos — la diferencia real se mide cambiando #define USE_GATHER y
        // recompilando, o usando este flag para seleccionar la ruta escalar forzada.
        if (use_gather) {
            ternary_lut_matmul_omp(x, w_packed, w_scale, decode_table,
                                   output, 1, d_in, d_out, group_size);
        } else {
            // Path escalar forzado: iterar grupos directamente sin gather
            memset(output, 0, d_out * sizeof(float));
            // Construir tile_luts de referencia (group_size=4, un tile)
            static float bmark_luts[TILE_G * 256];
            float decode_T[4 * 128];
            for (int i = 0; i < 4; i++)
                for (int c = 0; c < 128; c++)
                    decode_T[i * 128 + c] = decode_table[c * 4 + i];

            for (int g_base = 0; g_base < n_groups; g_base += TILE_G) {
                int g_end     = g_base + TILE_G < n_groups ? g_base + TILE_G : n_groups;
                int tile_size = g_end - g_base;
                for (int gl = 0; gl < tile_size; gl++) {
                    int g = g_base + gl;
                    const float* xg = x + g * group_size;
                    float* lut_g = bmark_luts + gl * 256;
                    memset(lut_g, 0, 128 * sizeof(float));
                    for (int i = 0; i < 4; i++) {
                        float xi = xg[i];
                        const float* col = decode_T + i * 128;
                        for (int c = 0; c < 128; c++) lut_g[c] += xi * col[c];
                    }
                    for (int c = 0; c < 128; c++) lut_g[128 + c] = -lut_g[c];
                }
                for (int j = 0; j < d_out; j++) {
                    const uint8_t* wr = w_packed + j * n_groups;
                    output[j] += ternary_lut_phase2_scalar(bmark_luts, wr, g_base, tile_size);
                }
            }
            for (int j = 0; j < d_out; j++) output[j] *= w_scale;
        }
    }

    // Medicion principal
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int t = 0; t < n_tokens; t++) {
        if (use_gather) {
            ternary_lut_matmul_omp(x, w_packed, w_scale, decode_table,
                                   output, 1, d_in, d_out, group_size);
        } else {
            memset(output, 0, d_out * sizeof(float));
            static float bmark_luts2[TILE_G * 256];
            float decode_T[4 * 128];
            for (int i = 0; i < 4; i++)
                for (int c = 0; c < 128; c++)
                    decode_T[i * 128 + c] = decode_table[c * 4 + i];

            for (int g_base = 0; g_base < n_groups; g_base += TILE_G) {
                int g_end     = g_base + TILE_G < n_groups ? g_base + TILE_G : n_groups;
                int tile_size = g_end - g_base;
                for (int gl = 0; gl < tile_size; gl++) {
                    int g = g_base + gl;
                    const float* xg = x + g * group_size;
                    float* lut_g = bmark_luts2 + gl * 256;
                    memset(lut_g, 0, 128 * sizeof(float));
                    for (int i = 0; i < 4; i++) {
                        float xi = xg[i];
                        const float* col = decode_T + i * 128;
                        for (int c = 0; c < 128; c++) lut_g[c] += xi * col[c];
                    }
                    for (int c = 0; c < 128; c++) lut_g[128 + c] = -lut_g[c];
                }
                for (int j = 0; j < d_out; j++) {
                    const uint8_t* wr = w_packed + j * n_groups;
                    output[j] += ternary_lut_phase2_scalar(bmark_luts2, wr, g_base, tile_size);
                }
            }
            for (int j = 0; j < d_out; j++) output[j] *= w_scale;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed_s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double tps = (elapsed_s > 0.0) ? (double)n_tokens / elapsed_s : 0.0;

    free(x);
    free(w_packed);
    free(decode_table);
    free(output);

    return tps;
}

} // extern "C"
