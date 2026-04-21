/*
 * Super-kernel CPU optimizado — Multi-arquitectura. v2.1
 *
 * NUEVAS OPTIMIZACIONES para 150 batches/seg:
 *
 *   ternary_i8_fused_gemv_avx2_omp  — INT8 GEMV paralelo OMP
 *     Paraleliza sobre d_out filas. Threshold: d_out >= 2048 (macroblock).
 *     Para d_out < 2048 es mas rapido single-thread (overhead OMP > ganancia).
 *
 *   decode_layer_fused  — 1 llamada C por capa completa
 *     Hace norm1→[forget|cand|out GEMV en 1 OMP parallel]→MLGRU→norm2→
 *     [gate|up GEMV en 1 OMP parallel]→SiLU→down GEMV→residual en 1 call.
 *
 *     Estrategia OMP: JITADO, un solo #pragma omp parallel por grupo de GEMVs.
 *     Dentro del parallel: cada thread toma su rango de filas de los 3 GEMVs
 *     concatenados (range [0, 3*d_model) para mixer, [0, 2*d_ff) para ffn).
 *     Esto elimina los 6 fork/join de teams separados y los 6 ctypes dispatches.
 *
 *   quantize_x_avx2  — cuantiza float32→int8 reutilizable entre GEMVs
 *     llamado UNA vez por grupo de proyecciones que comparten el mismo x.
 *
 * Estrategias anteriores conservadas sin cambios:
 *   ternary_lut_matmul_omp, fp32_gemv_avx2, mlgru_scan_fusion,
 *   fused_mlgru_gates_avx2, rms_norm_f32, silu_f32, elem_add_f32,
 *   ternary_i8_gemv_avx2, benchmark_lut_matmul.
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

#define TILE_G      16
#define MAX_D_MODEL 2048
#define MAX_D_FF    5120   // d_ff hasta 5120

extern "C" {

// ---------------------------------------------------------------------------
// Utilidades SIMD
// ---------------------------------------------------------------------------
static inline float _hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

static inline float _hmax256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}

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
// NUEVO: quantize_x_avx2
// Cuantizar float32 → int8, retorna x_scale = x_max/127.
// Reutilizable para multiples GEMVs con el mismo input x.
// ---------------------------------------------------------------------------
#ifdef __GNUC__
__attribute__((target("avx2,fma")))
#endif
static float quantize_x_avx2(const float* x, int8_t* out, int d_in) {
    const __m256 vabsmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256 vmax = _mm256_setzero_ps();
    int i = 0;
    for (; i <= d_in - 8; i += 8)
        vmax = _mm256_max_ps(vmax, _mm256_and_ps(_mm256_loadu_ps(x+i), vabsmask));
    float x_max = _hmax256_ps(vmax);
    for (; i < d_in; i++) { float a = fabsf(x[i]); if (a > x_max) x_max = a; }

    const float inv_s = 127.0f / (x_max + 1e-8f);
    const __m256  vinv = _mm256_set1_ps(inv_s);
    const __m256  vh   = _mm256_set1_ps(0.5f);
    const __m256i vlo  = _mm256_set1_epi32(-128);
    const __m256i vhi  = _mm256_set1_epi32(127);

    int ii = 0;
    for (; ii <= d_in - 8; ii += 8) {
        __m256i vi = _mm256_cvtps_epi32(_mm256_fmadd_ps(_mm256_loadu_ps(x+ii), vinv, vh));
        vi = _mm256_min_epi32(_mm256_max_epi32(vi, vlo), vhi);
        __m128i r = _mm_packs_epi32(_mm256_castsi256_si128(vi), _mm256_extracti128_si256(vi,1));
        r = _mm_packs_epi16(r, r);
        _mm_storel_epi64((__m128i*)(out+ii), r);
    }
    for (; ii < d_in; ii++) {
        float v = x[ii]*inv_s + 0.5f;
        v = v > 127.f ? 127.f : (v < -128.f ? -128.f : v);
        out[ii] = (int8_t)(int)v;
    }
    return x_max / (127.0f + 1e-8f);
}

// ---------------------------------------------------------------------------
// NUEVO: gemv_i8_row — 1 fila del GEMV ternario, retorna suma entera
// Usada dentro de loops OMP (inline-able, sin overhead de llamada).
// ---------------------------------------------------------------------------
#ifdef __GNUC__
__attribute__((target("avx2,fma")))
#endif
static inline int32_t gemv_i8_row(const int8_t* x_i8, const int8_t* wj, int d_in) {
    const __m256i vones16 = _mm256_set1_epi16(1);
    __m256i vacc = _mm256_setzero_si256();
    int ii = 0;
    for (; ii <= d_in - 32; ii += 32) {
        __m256i vact  = _mm256_loadu_si256((const __m256i*)(x_i8+ii));
        __m256i vw    = _mm256_loadu_si256((const __m256i*)(wj  +ii));
        __m256i vprod = _mm256_sign_epi8(vact, vw);
        __m256i p16lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vprod));
        __m256i p16hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vprod,1));
        vacc = _mm256_add_epi32(vacc, _mm256_madd_epi16(p16lo, vones16));
        vacc = _mm256_add_epi32(vacc, _mm256_madd_epi16(p16hi, vones16));
    }
    __m128i lo = _mm256_castsi256_si128(vacc);
    __m128i hi = _mm256_extracti128_si256(vacc,1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    int32_t sum = _mm_cvtsi128_si32(s);
    for (; ii < d_in; ii++) sum += (int32_t)x_i8[ii] * (int32_t)wj[ii];
    return sum;
}

// ---------------------------------------------------------------------------
// NUEVO: gemv_i8_block — GEMV de j_start a j_end (sin alloc, sin OMP interno)
// ---------------------------------------------------------------------------
#ifdef __GNUC__
__attribute__((target("avx2,fma")))
#endif
static void gemv_i8_block(
    const int8_t* x_i8, const int8_t* W, float* out,
    int d_in, int j_start, int j_end, float cs
) {
    for (int j = j_start; j < j_end; j++)
        out[j] = (float)gemv_i8_row(x_i8, W + (ptrdiff_t)j*d_in, d_in) * cs;
}

// ===========================================================================
// NUEVO: ternary_2bit_gemv_avx2
//
// Kernel GEMV ternario con pesos empaquetados a 2 bits.
//
// Encoding: 2 bits por peso, 4 pesos por byte, d_in pesos por fila.
//   00 = 0
//   01 = +1
//   10 = -1
//   11 = (no usado — el rango es ternario)
//
// Tamano de fila: d_in/4 bytes (vs d_in bytes en INT8) → 4x menos lectura.
//
// Para d_model=1536, d_out=1536:
//   INT8:  1536*1536 = 2.36MB por matriz
//   2-bit: 1536*384  = 590KB por matriz → cabe en L2 (tipicamente 256-512KB per core)
//   Resultado: 2-4x speedup por cache bandwidth
//
// Algoritmo:
//   Para cada fila j y cada chunk de 16 elementos d_in:
//     1. Leer 4 bytes (16 pesos de 2 bits cada uno)
//     2. Extraer bits con shift+AND
//     3. Convertir 00→0, 01→+1, 10→-1
//     4. Multiplicar por x_i8 y acumular
//
// Firma compatible con la API existente de ternary_i8_fused_gemv_avx2.
// x se cuantiza internamente (igual que ternary_i8_fused_gemv_avx2).
// w_2bit: [d_out, (d_in+3)/4] uint8 — pesos 2-bit packed, row-major.
// ===========================================================================

// ===========================================================================
// gemv_2bit_row — AVX2 con shuffle-LUT (sin stack alloc en el inner loop)
//
// Decodifica 2-bit pesos {00→0, 01→+1, 10→-1} a int8 y multiplica con x_i8.
//
// Técnica: usar _mm256_shuffle_epi8 como LUT de 16 entradas para convertir
// nibbles de 4 bits (2 pesos cada uno) a pares de int8 directamente.
// Procesa 16 pesos por iteración (4 bytes de w_2bit = 8 nibbles × 2 pesos).
// Cero allocaciones en stack, cero branches dentro del loop.
// ===========================================================================
#ifdef __GNUC__
__attribute__((target("avx2,fma")))
#endif
static inline int32_t gemv_2bit_row(
    const int8_t*  x_i8,   // [d_in] cuantizado int8
    const uint8_t* wj_2b,  // [(d_in+3)/4] pesos 2-bit packed
    int d_in
) {
    // Tabla LUT para 16 nibbles posibles (4 bits = 2 pesos de 2 bits cada uno)
    // nibble = bits[3:2] | bits[1:0]  → {peso1, peso0}
    // peso = 00→0, 01→+1, 10→-1, 11→0
    // LUT[nibble] = (int8_t){peso0, peso1} como int16 little-endian
    // Para shuffle: índice 0..15, resultado 2 bytes (lo=peso0, hi=peso1)
    // LUT entradas (lo byte = peso del bit[1:0], hi byte = peso del bit[3:2]):
    //  0b0000=0 → (0, 0)=0x0000
    //  0b0001=1 → (+1,0)=0x0001
    //  0b0010=2 → (-1,0)=0x00FF
    //  0b0011=3 → (0, 0)=0x0000 [inválido, tratado como 0]
    //  0b0100=4 → (0,+1)=0x0100
    //  0b0101=5 → (+1,+1)=0x0101
    //  0b0110=6 → (-1,+1)=0x01FF
    //  0b0111=7 → (0,+1)=0x0100 [inválido lo, sólo hi]
    //  0b1000=8 → (0,-1)=0xFF00
    //  0b1001=9 → (+1,-1)=0xFF01
    //  0b1010=10→ (-1,-1)=0xFFFF
    //  0b1011=11→ (0,-1)=0xFF00
    //  0b1100=12→ (0, 0)=0x0000 [inválido]
    //  ... etc.
    // Nota: int8_t -1 = 0xFF en complemento a 2
    // Tabla de 32 bytes (lo de cada par, luego hi de cada par) para shuffle_epi8:
    const __m128i lut_lo = _mm_set_epi8(
        // índices 8..15 (hi nibble)
        (int8_t)0,  (int8_t)1,  (int8_t)-1, (int8_t)0,
        (int8_t)0,  (int8_t)1,  (int8_t)-1, (int8_t)0,
        // índices 0..7 (lo nibble)
        (int8_t)0,  (int8_t)1,  (int8_t)-1, (int8_t)0,
        (int8_t)0,  (int8_t)1,  (int8_t)-1, (int8_t)0
    );
    const __m128i lut_hi = _mm_set_epi8(
        (int8_t)-1, (int8_t)-1, (int8_t)-1, (int8_t)-1,
        (int8_t)1,  (int8_t)1,  (int8_t)1,  (int8_t)1,
        (int8_t)0,  (int8_t)0,  (int8_t)0,  (int8_t)0,
        (int8_t)0,  (int8_t)0,  (int8_t)0,  (int8_t)0
    );
    const __m128i mask_lo_nibble = _mm_set1_epi8(0x0F);
    const __m256i vones16 = _mm256_set1_epi16(1);
    __m256i vacc = _mm256_setzero_si256();

    int i = 0;
    // Procesar 16 pesos por iteración (4 bytes = 8 nibbles × 2 pesos)
    for (; i <= d_in - 16; i += 16, wj_2b += 4) {
        // 4 bytes de pesos → 8 nibbles (un nibble = 2 pesos de 2 bits)
        __m128i raw4 = _mm_cvtsi32_si128(*(const uint32_t*)wj_2b);

        // Separar nibble bajo (bits[3:0]) y nibble alto (bits[7:4]) de cada byte
        __m128i lo_nibble = _mm_and_si128(raw4, mask_lo_nibble);          // 4 nibbles bajos
        __m128i hi_nibble = _mm_and_si128(_mm_srli_epi16(raw4, 4), mask_lo_nibble); // 4 nibbles altos

        // Intercalar: [lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3]
        __m128i interleaved = _mm_unpacklo_epi8(lo_nibble, hi_nibble);    // 8 nibbles ordenados

        // LUT lookup usando shuffle: convierte cada nibble al peso del bit bajo (pos 0 del par)
        __m128i w_lo = _mm_shuffle_epi8(lut_lo, interleaved);  // peso de bits[1:0] de cada nibble
        // LUT lookup para el bit alto (pos 1 del par)
        __m128i w_hi = _mm_shuffle_epi8(lut_hi, interleaved);  // peso de bits[3:2] de cada nibble

        // Intercalar para obtener [w0,w1,w2,...,w15] en orden
        __m128i weights_lo = _mm_unpacklo_epi8(w_lo, w_hi);    // pesos 0..7
        __m128i weights_hi = _mm_unpackhi_epi8(w_lo, w_hi);    // pesos 8..15

        // Unir en un __m256i: 16 pesos int8 ordenados
        __m256i vw = _mm256_set_m128i(weights_hi, weights_lo);

        // Cargar 16 activaciones int8
        __m256i vact = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(x_i8+i)));

        // Multiplicar: vact (int16) * vw (int16) → acum int32
        // Convertir vw de int8 a int16 para madd
        __m256i vw16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vw));
        vacc = _mm256_add_epi32(vacc, _mm256_madd_epi16(vact, vw16));
    }

    // Reducir
    __m128i lo128 = _mm256_castsi256_si128(vacc);
    __m128i hi128 = _mm256_extracti128_si256(vacc, 1);
    __m128i s4    = _mm_add_epi32(lo128, hi128);
    s4 = _mm_hadd_epi32(s4, s4);
    s4 = _mm_hadd_epi32(s4, s4);
    int32_t sum = _mm_cvtsi128_si32(s4);

    // Tail escalar
    for (; i < d_in; i++) {
        int byte_i  = i >> 2;
        int shift   = (i & 3) << 1;
        uint8_t code = (wj_2b[byte_i - (d_in/16*4)] >> shift) & 3;
        sum += (int32_t)x_i8[i] * ((code==1) ? 1 : (code==2) ? -1 : 0);
    }
    return sum;
}


#ifdef __GNUC__
__attribute__((target("avx2,fma")))
#endif
EXPORT void ternary_2bit_fused_gemv_avx2(
    const float*   x,      // [d_in] float32
    const uint8_t* w_2bit, // [d_out, (d_in+3)/4] pesos 2-bit packed
    float*         output, // [d_out]
    int            d_in,
    int            d_out,
    float          w_scale
) {
    static thread_local int8_t _xi[MAX_D_FF];
    float xs = quantize_x_avx2(x, _xi, d_in);
    float cs = xs * w_scale;
    int bytes_per_row = (d_in + 3) / 4;

    for (int j = 0; j < d_out; j++) {
        const uint8_t* wj = w_2bit + (ptrdiff_t)j * bytes_per_row;
        output[j] = (float)gemv_2bit_row(_xi, wj, d_in) * cs;
    }
}

// pack_ternary_to_2bit: convertir int8 ternario [-1,0,1] a 2-bit packed
// Input:  W [d_out, d_in] int8  (valores -1, 0, 1)
// Output: packed [d_out, (d_in+3)/4] uint8
// Llamar desde Python OFFLINE, antes de usar ternary_2bit_fused_gemv_avx2.
EXPORT void pack_ternary_to_2bit(
    const int8_t* W,       // [d_out, d_in] int8 ternario
    uint8_t*      packed,  // [d_out, (d_in+3)/4] output — pre-asignado por caller
    int           d_in,
    int           d_out
) {
    int bytes_per_row = (d_in + 3) / 4;
    for (int j = 0; j < d_out; j++) {
        const int8_t* wj  = W      + (ptrdiff_t)j * d_in;
        uint8_t*      pj  = packed + (ptrdiff_t)j * bytes_per_row;
        memset(pj, 0, bytes_per_row);
        for (int i = 0; i < d_in; i++) {
            int8_t v = wj[i];
            uint8_t code = (v == 1) ? 1 : (v == -1) ? 2 : 0;
            pj[i >> 2] |= (code << ((i & 3) * 2));
        }
    }
}



// ---------------------------------------------------------------------------
// Fast exp/sigmoid/silu AVX2
// ---------------------------------------------------------------------------
static inline __m256 _avx2_exp_ps(__m256 x) {
    const __m256 vln2 = _mm256_set1_ps(0.6931471806f);
    const __m256 vilg = _mm256_set1_ps(1.4426950408f);
    x = _mm256_min_ps(x, _mm256_set1_ps(88.f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.f));
    __m256  z  = _mm256_fmadd_ps(x, vilg, _mm256_set1_ps(0.5f));
    __m256i ni = _mm256_cvttps_epi32(z);
    __m256  fn = _mm256_cvtepi32_ps(ni);
    __m256  r  = _mm256_fnmadd_ps(fn, vln2, x);
    __m256 p = _mm256_fmadd_ps(_mm256_set1_ps(1.f/120), r, _mm256_set1_ps(1.f/24));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.f/6));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(.5f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.f));
    __m256 pw2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(ni,_mm256_set1_epi32(127)),23));
    return _mm256_mul_ps(p, pw2);
}
static inline __m256 _avx2_sigmoid_ps(__m256 x) {
    return _mm256_div_ps(_mm256_set1_ps(1.f),
        _mm256_add_ps(_mm256_set1_ps(1.f), _avx2_exp_ps(_mm256_sub_ps(_mm256_setzero_ps(), x))));
}
static inline __m256 _avx2_silu_ps(__m256 x) {
    return _mm256_mul_ps(x, _avx2_sigmoid_ps(x));
}

// ---------------------------------------------------------------------------
// NUEVO: rms_norm_inline — RMSNorm interna (sin llamada de funcion)
// ---------------------------------------------------------------------------
#ifdef __GNUC__
__attribute__((target("avx2,fma")))
#endif
static void rms_norm_inline(const float* x, const float* w, float* out, int D, float eps) {
    __m256 vss = _mm256_setzero_ps();
    int i = 0;
    for (; i <= D-8; i+=8) { __m256 v = _mm256_loadu_ps(x+i); vss = _mm256_fmadd_ps(v,v,vss); }
    __m128 ss4 = _mm_add_ps(_mm256_castps256_ps128(vss), _mm256_extractf128_ps(vss,1));
    ss4 = _mm_hadd_ps(ss4, ss4); ss4 = _mm_hadd_ps(ss4, ss4);
    float ss = _mm_cvtss_f32(ss4);
    for (; i < D; i++) ss += x[i]*x[i];
    float sc = 1.0f / sqrtf(ss/(float)D + eps);
    __m256 vsc = _mm256_set1_ps(sc);
    i = 0;
    for (; i <= D-8; i+=8)
        _mm256_storeu_ps(out+i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x+i),vsc),_mm256_loadu_ps(w+i)));
    for (; i < D; i++) out[i] = x[i]*sc*w[i];
}

// ===========================================================================
// NUEVO: decode_layer_fused  v2
//
// 1 llamada C por capa completa. Estrategia OMP:
//   - UN solo omp parallel region por grupo de GEMVs (no sections/tasks)
//   - Thread i toma filas [chunk*i, chunk*(i+1)) de los GEMVs concatenados
//   - Para mixer: range [0, 3*d_model) → forget[j], cand[j], out[j]
//   - Para ffn:   range [0, 2*d_ff)   → gate[j], up[j]
//   - Para down:  range [0, d_model)  → single parallel for
//
// Esto elimina completamente el overhead de fork/join OMP entre operaciones.
//
// No maneja fp_residual ni pfnet (los hace Python aparte, son rapidos FP).
// ===========================================================================
#ifdef __GNUC__
__attribute__((target("avx2,fma")))
#endif
EXPORT void decode_layer_fused(
    const float* x_in,          // [d_model]
    float*       x_out,         // [d_model] output
    float*       h,             // [d_model] hidden MLGRU (in/out)
    const float* norm1_w,       // [d_model]
    const float* norm2_w,       // [d_model]
    float        norm_eps,
    const int8_t* W_forget, float s_forget,
    const int8_t* W_cand,   float s_cand,
    const int8_t* W_out,    float s_out,
    const int8_t* W_gate,   float s_gate,
    const int8_t* W_up,     float s_up,
    const int8_t* W_down,   float s_down,
    int d_model,
    int d_ff
) {
    // Buffers thread_local (0 malloc/call)
    static thread_local float  _xn1[MAX_D_MODEL];   // x after norm1
    static thread_local int8_t _xi1[MAX_D_MODEL];   // x_i8 for mixer GEMVs
    static thread_local float  _fraw[MAX_D_MODEL];
    static thread_local float  _craw[MAX_D_MODEL];
    static thread_local float  _graw[MAX_D_MODEL];
    static thread_local float  _mout[MAX_D_MODEL];  // mixer output
    static thread_local float  _xmid[MAX_D_MODEL];  // x + mout (residual 1)
    static thread_local float  _xn2[MAX_D_MODEL];   // x after norm2
    static thread_local int8_t _xi2[MAX_D_MODEL];   // x_i8 for ffn GEMVs
    static thread_local float  _gater[MAX_D_FF];
    static thread_local float  _upr[MAX_D_FF];
    static thread_local float  _hid[MAX_D_FF];      // ffn hidden after silu*up
    static thread_local int8_t _hi8[MAX_D_FF];
    static thread_local float  _ffno[MAX_D_MODEL];  // ffn output

    // PASO 1: RMSNorm1
    rms_norm_inline(x_in, norm1_w, _xn1, d_model, norm_eps);

    // PASO 2: Cuantizar una vez para forget/cand/out (todos toman _xn1)
    float xs1 = quantize_x_avx2(_xn1, _xi1, d_model);
    float cs_f = xs1 * s_forget;
    float cs_c = xs1 * s_cand;
    float cs_o = xs1 * s_out;

    // PASO 3-5: GEMVs mixer en 1 omp parallel, dividir 3*d_model filas
    // Esquema: thread t procesa filas [t*chunk, (t+1)*chunk) de [0, 3*d_model)
    // j < d_model → forget, d_model <= j < 2*d_model → cand, >= 2*d_model → out
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid      = omp_get_thread_num();
        int total3   = 3 * d_model;
        int chunk    = (total3 + nthreads - 1) / nthreads;
        int jS       = tid * chunk;
        int jE       = jS + chunk < total3 ? jS + chunk : total3;

        for (int j = jS; j < jE; j++) {
            if (j < d_model) {
                _fraw[j] = (float)gemv_i8_row(_xi1, W_forget+(ptrdiff_t)j*d_model, d_model) * cs_f;
            } else if (j < 2*d_model) {
                int jj = j - d_model;
                _craw[jj] = (float)gemv_i8_row(_xi1, W_cand+(ptrdiff_t)jj*d_model, d_model) * cs_c;
            } else {
                int jj = j - 2*d_model;
                _graw[jj] = (float)gemv_i8_row(_xi1, W_out+(ptrdiff_t)jj*d_model, d_model) * cs_o;
            }
        }
    } // barrier aqui

    // PASO 6: fused_mlgru_gates + residual
    {
        int i = 0;
        for (; i <= d_model-8; i+=8) {
            __m256 vf_r = _mm256_loadu_ps(_fraw+i);
            __m256 vc_r = _mm256_loadu_ps(_craw+i);
            __m256 vg_r = _mm256_loadu_ps(_graw+i);
            __m256 vh   = _mm256_loadu_ps(h+i);
            __m256 vx   = _mm256_loadu_ps(x_in+i);

            __m256 vf = _avx2_sigmoid_ps(vf_r);
            __m256 vc = _avx2_silu_ps(vc_r);
            __m256 vg = _avx2_sigmoid_ps(vg_r);
            __m256 vhn = _mm256_fmadd_ps(vf, _mm256_sub_ps(vh, vc), vc);
            __m256 vm  = _mm256_mul_ps(vg, vhn);

            _mm256_storeu_ps(h+i, vhn);
            _mm256_storeu_ps(_mout+i, vm);
            _mm256_storeu_ps(_xmid+i, _mm256_add_ps(vx, vm));
        }
        for (; i < d_model; i++) {
            float f  = 1.f/(1.f+expf(-_fraw[i]));
            float c  = _craw[i]/(1.f+expf(-_craw[i]));
            float g  = 1.f/(1.f+expf(-_graw[i]));
            float hn = c + f*(h[i]-c);
            h[i] = hn;
            _mout[i] = g*hn;
            _xmid[i] = x_in[i] + _mout[i];
        }
    }

    // PASO 7: RMSNorm2
    rms_norm_inline(_xmid, norm2_w, _xn2, d_model, norm_eps);

    // PASO 8: Cuantizar una vez para gate/up
    float xs2 = quantize_x_avx2(_xn2, _xi2, d_model);
    float cs_gate = xs2 * s_gate;
    float cs_up   = xs2 * s_up;

    // PASO 9-10: GEMVs ffn gate+up en 1 omp parallel, dividir 2*d_ff filas
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid      = omp_get_thread_num();
        int total2   = 2 * d_ff;
        int chunk    = (total2 + nthreads - 1) / nthreads;
        int jS       = tid * chunk;
        int jE       = jS + chunk < total2 ? jS + chunk : total2;

        for (int j = jS; j < jE; j++) {
            if (j < d_ff) {
                _gater[j] = (float)gemv_i8_row(_xi2, W_gate+(ptrdiff_t)j*d_model, d_model) * cs_gate;
            } else {
                int jj = j - d_ff;
                _upr[jj] = (float)gemv_i8_row(_xi2, W_up+(ptrdiff_t)jj*d_model, d_model) * cs_up;
            }
        }
    } // barrier

    // PASO 11: silu(gate)*up → hidden, luego cuantizar
    {
        int i = 0;
        for (; i <= d_ff-8; i+=8) {
            __m256 vg = _mm256_loadu_ps(_gater+i);
            __m256 vu = _mm256_loadu_ps(_upr+i);
            _mm256_storeu_ps(_hid+i, _mm256_mul_ps(_avx2_silu_ps(vg), vu));
        }
        for (; i < d_ff; i++) _hid[i] = _gater[i]/(1.f+expf(-_gater[i])) * _upr[i];
    }
    float hs = quantize_x_avx2(_hid, _hi8, d_ff);

    // PASO 12: down GEMV paralelo
    float cs_down = hs * s_down;
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < d_model; j++)
        _ffno[j] = (float)gemv_i8_row(_hi8, W_down+(ptrdiff_t)j*d_ff, d_ff) * cs_down;

    // PASO 13: x_out = x_mid + ffn_out
    {
        int i = 0;
        for (; i <= d_model-8; i+=8)
            _mm256_storeu_ps(x_out+i,
                _mm256_add_ps(_mm256_loadu_ps(_xmid+i), _mm256_loadu_ps(_ffno+i)));
        for (; i < d_model; i++) x_out[i] = _xmid[i] + _ffno[i];
    }
}


// ===========================================================================
// NUEVO: ternary_i8_fused_gemv_avx2_omp  (con x_i8 precomputado opcional)
// ===========================================================================
#ifdef __GNUC__
__attribute__((target("avx2,fma")))
#endif
EXPORT void ternary_i8_fused_gemv_avx2_omp(
    const float*  x,
    const int8_t* w,
    float*        output,
    int           d_in,
    int           d_out,
    float         w_scale,
    const int8_t* x_i8_pre,   // NULL = cuantizar internamente
    float         x_scale_pre
) {
    static thread_local int8_t _xi[MAX_D_FF];

    const int8_t* x_i8;
    float xs;
    if (x_i8_pre) { x_i8 = x_i8_pre; xs = x_scale_pre; }
    else           { xs = quantize_x_avx2(x, _xi, d_in); x_i8 = _xi; }

    float cs = xs * w_scale;

    // Paralelo solo si d_out es grande (>= 1024 para amortizar overhead OMP)
    if (d_out >= 1024) {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < d_out; j++)
            output[j] = (float)gemv_i8_row(x_i8, w+(ptrdiff_t)j*d_in, d_in) * cs;
    } else {
        gemv_i8_block(x_i8, w, output, d_in, 0, d_out, cs);
    }
}


// ===========================================================================
// ternary_lut_matmul_omp  (sin cambios del original)
// ===========================================================================
#ifdef __GNUC__
__attribute__((target("avx2,fma")))
#endif
EXPORT void ternary_lut_matmul_omp(
    const float*   x,
    const uint8_t* w_packed,
    float          w_scale,
    const float*   decode_table,
    float*         output,
    int M, int d_in, int d_out, int group_size
) {
    int n_groups = d_in / group_size;

    if (M == 1) {
        const float* xm  = x;
        float*       outm = output;
        memset(outm, 0, d_out * sizeof(float));

        float decode_T_shared[8 * 128];
        if (group_size == 4) {
            for (int i = 0; i < 4; i++)
                for (int c = 0; c < 128; c++)
                    decode_T_shared[i * 128 + c] = decode_table[c * 4 + i];
        }

        static float tile_luts_shared[TILE_G * 256] __attribute__((aligned(64)));

        #pragma omp parallel
        {
            for (int g_base = 0; g_base < n_groups; g_base += TILE_G) {
                int g_end     = g_base + TILE_G < n_groups ? g_base + TILE_G : n_groups;
                int tile_size = g_end - g_base;

                #pragma omp single
                {
                    for (int g_local = 0; g_local < tile_size; g_local++) {
                        int          g     = g_base + g_local;
                        const float* x_grp = xm + g * group_size;
                        float*       lut_g = tile_luts_shared + g_local * 256;
                        if (group_size == 4) {
                            memset(lut_g, 0, 128 * sizeof(float));
                            for (int i = 0; i < 4; i++) {
                                __m256 vxi = _mm256_broadcast_ss(x_grp + i);
                                const float* col = decode_T_shared + i * 128;
                                for (int c = 0; c < 128; c += 8) {
                                    _mm256_storeu_ps(lut_g + c,
                                        _mm256_fmadd_ps(vxi, _mm256_loadu_ps(col+c),
                                                        _mm256_loadu_ps(lut_g+c)));
                                }
                            }
                            const __m256 vneg = _mm256_set1_ps(-1.0f);
                            for (int c = 0; c < 128; c += 8)
                                _mm256_storeu_ps(lut_g+128+c,
                                    _mm256_mul_ps(vneg, _mm256_loadu_ps(lut_g+c)));
                        }
                    }
                }

                #pragma omp for schedule(static) nowait
                for (int j = 0; j < d_out; j++)
                    outm[j] += ternary_lut_phase2_scalar(tile_luts_shared, w_packed+j*n_groups, g_base, tile_size);
            }
            #pragma omp for schedule(static) nowait
            for (int j = 0; j < d_out; j++) outm[j] *= w_scale;
        }
        return;
    }

    for (int m = 0; m < M; m++) {
        const float* xm  = x + m * d_in;
        float*       outm = output + m * d_out;
        memset(outm, 0, d_out * sizeof(float));
        #pragma omp parallel if(d_out >= 512)
        {
            static thread_local float tile_luts[TILE_G * 256];
            static thread_local float decode_T[8 * 128];
            if (group_size == 4) {
                for (int i = 0; i < 4; i++)
                    for (int c = 0; c < 128; c++)
                        decode_T[i*128+c] = decode_table[c*4+i];
            } else {
                int gs = group_size < 8 ? group_size : 8;
                for (int i = 0; i < gs; i++)
                    for (int c = 0; c < 128; c++)
                        decode_T[i*128+c] = decode_table[c*group_size+i];
            }
            for (int g_base = 0; g_base < n_groups; g_base += TILE_G) {
                int g_end = g_base+TILE_G < n_groups ? g_base+TILE_G : n_groups;
                int tile_size = g_end - g_base;
                for (int g_local = 0; g_local < tile_size; g_local++) {
                    int g = g_base+g_local;
                    const float* x_grp = xm + g*group_size;
                    float* lut_g = tile_luts + g_local*256;
                    if (group_size == 4) {
                        memset(lut_g, 0, 128*sizeof(float));
                        for (int i = 0; i < 4; i++) {
                            __m256 vxi = _mm256_broadcast_ss(x_grp+i);
                            const float* col = decode_T + i*128;
                            for (int c = 0; c < 128; c += 8)
                                _mm256_storeu_ps(lut_g+c,
                                    _mm256_fmadd_ps(vxi,_mm256_loadu_ps(col+c),_mm256_loadu_ps(lut_g+c)));
                        }
                        const __m256 vneg = _mm256_set1_ps(-1.f);
                        for (int c = 0; c < 128; c += 8)
                            _mm256_storeu_ps(lut_g+128+c, _mm256_mul_ps(vneg,_mm256_loadu_ps(lut_g+c)));
                    }
                }
                #pragma omp for schedule(static)
                for (int j = 0; j < d_out-1; j += 2) {
                    outm[j  ] += ternary_lut_phase2_scalar(tile_luts, w_packed+j*n_groups, g_base, tile_size);
                    outm[j+1] += ternary_lut_phase2_scalar(tile_luts, w_packed+(j+1)*n_groups, g_base, tile_size);
                }
                #pragma omp single nowait
                if (d_out&1) outm[d_out-1] += ternary_lut_phase2_scalar(tile_luts, w_packed+(d_out-1)*n_groups, g_base, tile_size);
            }
            #pragma omp for schedule(static) nowait
            for (int j = 0; j < d_out; j++) outm[j] *= w_scale;
        }
    }
}


// ---------------------------------------------------------------------------
// fp32_gemv_avx2  (sin cambios)
// ---------------------------------------------------------------------------
EXPORT void fp32_gemv_avx2(
    const float* x, const float* W, float* output,
    int M, int d_in, int d_out
) {
    for (int m = 0; m < M; m++) {
        const float* xm = x + m*d_in;
        float* outm = output + m*d_out;
        for (int j = 0; j < d_out-7; j += 8) {
            const float* w0=W+j*d_in, *w1=w0+d_in, *w2=w1+d_in, *w3=w2+d_in;
            const float* w4=w3+d_in, *w5=w4+d_in, *w6=w5+d_in, *w7=w6+d_in;
            __m256 vs0=_mm256_setzero_ps(), vs1=vs0, vs2=vs0, vs3=vs0;
            __m256 vs4=_mm256_setzero_ps(), vs5=vs4, vs6=vs4, vs7=vs4;
            for (int i=0; i<=d_in-8; i+=8) {
                __m256 vx=_mm256_loadu_ps(xm+i);
                vs0=_mm256_fmadd_ps(vx,_mm256_loadu_ps(w0+i),vs0);
                vs1=_mm256_fmadd_ps(vx,_mm256_loadu_ps(w1+i),vs1);
                vs2=_mm256_fmadd_ps(vx,_mm256_loadu_ps(w2+i),vs2);
                vs3=_mm256_fmadd_ps(vx,_mm256_loadu_ps(w3+i),vs3);
                vs4=_mm256_fmadd_ps(vx,_mm256_loadu_ps(w4+i),vs4);
                vs5=_mm256_fmadd_ps(vx,_mm256_loadu_ps(w5+i),vs5);
                vs6=_mm256_fmadd_ps(vx,_mm256_loadu_ps(w6+i),vs6);
                vs7=_mm256_fmadd_ps(vx,_mm256_loadu_ps(w7+i),vs7);
            }
            outm[j]=_hsum256_ps(vs0); outm[j+1]=_hsum256_ps(vs1);
            outm[j+2]=_hsum256_ps(vs2); outm[j+3]=_hsum256_ps(vs3);
            outm[j+4]=_hsum256_ps(vs4); outm[j+5]=_hsum256_ps(vs5);
            outm[j+6]=_hsum256_ps(vs6); outm[j+7]=_hsum256_ps(vs7);
        }
        int jt=(d_out/8)*8;
        for (int j=jt;j<d_out;j++) {
            const float* wj=W+j*d_in; __m256 vs=_mm256_setzero_ps();
            for (int i=0;i<=d_in-8;i+=8) vs=_mm256_fmadd_ps(_mm256_loadu_ps(xm+i),_mm256_loadu_ps(wj+i),vs);
            float s=_hsum256_ps(vs);
            for (int i=(d_in/8)*8;i<d_in;i++) s+=xm[i]*wj[i];
            outm[j]=s;
        }
    }
}


// ---------------------------------------------------------------------------
// mlgru_scan_fusion  (sin cambios)
// ---------------------------------------------------------------------------
EXPORT void mlgru_scan_fusion(
    const float* c, const float* f, const float* g_gate,
    float* h, float* o, int B, int d_model
) {
    int total = B * d_model;
    for (int i=0; i<=total-8; i+=8) {
        __m256 vc=_mm256_loadu_ps(c+i), vf=_mm256_loadu_ps(f+i);
        __m256 vh=_mm256_loadu_ps(h+i), vg=_mm256_loadu_ps(g_gate+i);
        __m256 hn=_mm256_fmadd_ps(vf,_mm256_sub_ps(vh,vc),vc);
        _mm256_storeu_ps(h+i,hn);
        _mm256_storeu_ps(o+i,_mm256_mul_ps(vg,hn));
    }
    int ts=(total/8)*8;
    for (int i=ts;i<total;i++) {
        float hn=c[i]+f[i]*(h[i]-c[i]); h[i]=hn; o[i]=g_gate[i]*hn;
    }
}


// ---------------------------------------------------------------------------
// rms_norm_f32  (sin cambios)
// ---------------------------------------------------------------------------
EXPORT void rms_norm_f32(
    const float* x, const float* weight, float* out, int D, float eps
) {
    rms_norm_inline(x, weight, out, D, eps);
}


// ---------------------------------------------------------------------------
// silu_f32  (sin cambios)
// ---------------------------------------------------------------------------
EXPORT void silu_f32(const float* x, float* out, int N) {
    for (int i=0;i<=N-8;i+=8) _mm256_storeu_ps(out+i,_avx2_silu_ps(_mm256_loadu_ps(x+i)));
    int t=(N/8)*8;
    for (int i=t;i<N;i++) { float s=1.f/(1.f+expf(-x[i])); out[i]=x[i]*s; }
}


// ---------------------------------------------------------------------------
// fused_mlgru_gates_avx2  (sin cambios)
// ---------------------------------------------------------------------------
EXPORT void fused_mlgru_gates_avx2(
    const float* f_raw, const float* c_raw, const float* g_raw,
    float* h, float* out, int D
) {
    int i=0;
    for (;i<=D-8;i+=8) {
        __m256 vfr=_mm256_loadu_ps(f_raw+i), vcr=_mm256_loadu_ps(c_raw+i);
        __m256 vgr=_mm256_loadu_ps(g_raw+i), vh=_mm256_loadu_ps(h+i);
        __m256 vf=_avx2_sigmoid_ps(vfr), vc=_avx2_silu_ps(vcr), vg=_avx2_sigmoid_ps(vgr);
        __m256 vhn=_mm256_fmadd_ps(vf,_mm256_sub_ps(vh,vc),vc);
        _mm256_storeu_ps(out+i,_mm256_mul_ps(vg,vhn));
        _mm256_storeu_ps(h+i,vhn);
    }
    for (;i<D;i++) {
        float f=1.f/(1.f+expf(-f_raw[i])), c=c_raw[i]/(1.f+expf(-c_raw[i]));
        float g=1.f/(1.f+expf(-g_raw[i])), hn=c+f*(h[i]-c);
        out[i]=g*hn; h[i]=hn;
    }
}


// ---------------------------------------------------------------------------
// elem_add_f32  (sin cambios)
// ---------------------------------------------------------------------------
EXPORT void elem_add_f32(const float* a, const float* b, float* out, int N) {
    int i=0;
    for (;i<=N-8;i+=8)
        _mm256_storeu_ps(out+i,_mm256_add_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i)));
    for (;i<N;i++) out[i]=a[i]+b[i];
}


// ---------------------------------------------------------------------------
// ternary_i8_fused_gemv_avx2  (backward compat — ahora llama OMP version)
// ---------------------------------------------------------------------------
#ifdef __GNUC__
__attribute__((target("avx2,fma")))
#endif
EXPORT void ternary_i8_fused_gemv_avx2(
    const float* x, const int8_t* w, float* output,
    int d_in, int d_out, float w_scale
) {
    ternary_i8_fused_gemv_avx2_omp(x, w, output, d_in, d_out, w_scale, NULL, 0.f);
}


// ---------------------------------------------------------------------------
// ternary_i8_gemv_avx2  (sin cambios)
// ---------------------------------------------------------------------------
#ifdef __GNUC__
__attribute__((target("avx2,fma")))
#endif
EXPORT void ternary_i8_gemv_avx2(
    const int8_t* x, const int8_t* w, float* output,
    int d_in, int d_out, float scale
) {
    const __m256i vones16=_mm256_set1_epi16(1);
    for (int j=0;j<d_out;j++) {
        const int8_t* wj=w+(ptrdiff_t)j*d_in;
        __m256i vacc=_mm256_setzero_si256();
        int i=0;
        for (;i<=d_in-32;i+=32) {
            __m256i vact=_mm256_loadu_si256((const __m256i*)(x+i));
            __m256i vw=_mm256_loadu_si256((const __m256i*)(wj+i));
            __m256i vp=_mm256_sign_epi8(vact,vw);
            vacc=_mm256_add_epi32(vacc,_mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(vp)),vones16));
            vacc=_mm256_add_epi32(vacc,_mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(vp,1)),vones16));
        }
        __m128i lo=_mm256_castsi256_si128(vacc), hi=_mm256_extracti128_si256(vacc,1);
        __m128i s=_mm_add_epi32(lo,hi); s=_mm_hadd_epi32(s,s); s=_mm_hadd_epi32(s,s);
        int32_t sum=_mm_cvtsi128_si32(s);
        for (;i<d_in;i++) sum+=(int32_t)x[i]*(int32_t)wj[i];
        output[j]=(float)sum*scale;
    }
}


// ---------------------------------------------------------------------------
// benchmark_lut_matmul  (sin cambios)
// ---------------------------------------------------------------------------
EXPORT double benchmark_lut_matmul(int d_in, int d_out, int n_tokens, int use_gather) {
    const int group_size = 4;
    int n_groups = d_in / group_size;
    float*   x            = (float*)malloc(d_in*sizeof(float));
    uint8_t* w_packed     = (uint8_t*)malloc((size_t)d_out*n_groups);
    float*   decode_table = (float*)malloc(128*group_size*sizeof(float));
    float*   output       = (float*)malloc(d_out*sizeof(float));
    if (!x||!w_packed||!decode_table||!output) { free(x);free(w_packed);free(decode_table);free(output); return -1.; }
    for (int i=0;i<d_in;i++) x[i]=(float)(i%17-8)*0.1f;
    for (size_t i=0;i<(size_t)d_out*n_groups;i++) w_packed[i]=(uint8_t)(i%256);
    for (int i=0;i<128*group_size;i++) decode_table[i]=(float)(i%7-3)*0.5f;
    for (int t=0;t<5;t++) ternary_lut_matmul_omp(x,w_packed,0.01f,decode_table,output,1,d_in,d_out,group_size);
    struct timespec t0,t1;
    clock_gettime(CLOCK_MONOTONIC,&t0);
    for (int t=0;t<n_tokens;t++) ternary_lut_matmul_omp(x,w_packed,0.01f,decode_table,output,1,d_in,d_out,group_size);
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double el=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
    double tps=el>0.0?(double)n_tokens/el:0.;
    free(x);free(w_packed);free(decode_table);free(output);
    return tps;
}


// ===========================================================================
// decode_layer_fused_2bit
//
// Identico a decode_layer_fused pero con pesos empaquetados a 2 bits.
//
// Reduccion de bandwidth por capa:
//   INT8:  6 matrices × ~1024² B = ~6 MB  de pesos por capa a leer
//   2-bit: 6 matrices × ~256  KB = ~1.5MB por capa → 4× menor bandwidth
//
// Para 14 capas de 170M:
//   INT8:  14 × 11.6 MB = 163 MB total a leer por patch → 38/163  = 233 patch/s teórico
//   2-bit: 14 × 2.9  MB = 40.6 MB total               → 38/40.6 = 936 patch/s teórico
//
// Firma compatible con decode_layer_fused excepto que W_* son uint8* 2-bit.
// w_*_rows = (d_in+3)/4 bytes por fila.
//
// FPResidual y PFNet se manejan en Python igualmente (son FP32, triviales).

// Helper: gemv_2bit_block — GEMV 2-bit para un rango de filas
#ifdef __GNUC__
__attribute__((target("avx2,fma")))
#endif
static void gemv_2bit_block(
    const int8_t* x_i8,    // [d_in] cuantizado
    const uint8_t* W_2b,   // [d_out, (d_in+3)/4] pesos 2-bit
    float* out,            // [d_out]
    int d_in, int j_start, int j_end, float cs
) {
    int bytes_per_row = (d_in + 3) / 4;
    for (int j = j_start; j < j_end; j++) {
        const uint8_t* wj = W_2b + (ptrdiff_t)j * bytes_per_row;
        out[j] = (float)gemv_2bit_row(x_i8, wj, d_in) * cs;
    }
}

#ifdef __GNUC__
__attribute__((target("avx2,fma")))
#endif
EXPORT void decode_layer_fused_2bit(
    const float*   x_in,          // [d_model]
    float*         x_out,         // [d_model] output
    float*         h,             // [d_model] hidden MLGRU (in/out)
    const float*   norm1_w,       // [d_model]
    const float*   norm2_w,       // [d_model]
    float          norm_eps,
    const uint8_t* W_forget, float s_forget,   // [d_model, (d_model+3)/4]
    const uint8_t* W_cand,   float s_cand,
    const uint8_t* W_out,    float s_out,
    const uint8_t* W_gate,   float s_gate,     // [d_ff, (d_model+3)/4]
    const uint8_t* W_up,     float s_up,
    const uint8_t* W_down,   float s_down,     // [d_model, (d_ff+3)/4]
    int d_model,
    int d_ff
) {
    // Buffers thread_local (0 malloc/call)
    static thread_local float  _xn1[MAX_D_MODEL];
    static thread_local int8_t _xi1[MAX_D_MODEL];
    static thread_local float  _fraw[MAX_D_MODEL];
    static thread_local float  _craw[MAX_D_MODEL];
    static thread_local float  _graw[MAX_D_MODEL];
    static thread_local float  _mout[MAX_D_MODEL];
    static thread_local float  _xmid[MAX_D_MODEL];
    static thread_local float  _xn2[MAX_D_MODEL];
    static thread_local int8_t _xi2[MAX_D_MODEL];
    static thread_local float  _gater[MAX_D_FF];
    static thread_local float  _upr[MAX_D_FF];
    static thread_local float  _hid[MAX_D_FF];
    static thread_local int8_t _hi8[MAX_D_FF];
    static thread_local float  _ffno[MAX_D_MODEL];

    // PASO 1: RMSNorm1
    rms_norm_inline(x_in, norm1_w, _xn1, d_model, norm_eps);

    // PASO 2: Cuantizar x una vez para las 3 GEMVs del mixer
    float xs1   = quantize_x_avx2(_xn1, _xi1, d_model);
    float cs_f  = xs1 * s_forget;
    float cs_c  = xs1 * s_cand;
    float cs_o  = xs1 * s_out;

    // PASO 3-5: GEMVs mixer en parallel OMP, dividir 3*d_model filas
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid      = omp_get_thread_num();
        int total3   = 3 * d_model;
        int chunk    = (total3 + nthreads - 1) / nthreads;
        int jS       = tid * chunk;
        int jE       = jS + chunk < total3 ? jS + chunk : total3;

        int bpr_m = (d_model + 3) / 4;  // bytes por fila para matrices d_model×d_model

        for (int j = jS; j < jE; j++) {
            if (j < d_model) {
                const uint8_t* wj = W_forget + (ptrdiff_t)j * bpr_m;
                _fraw[j] = (float)gemv_2bit_row(_xi1, wj, d_model) * cs_f;
            } else if (j < 2*d_model) {
                int jj = j - d_model;
                const uint8_t* wj = W_cand + (ptrdiff_t)jj * bpr_m;
                _craw[jj] = (float)gemv_2bit_row(_xi1, wj, d_model) * cs_c;
            } else {
                int jj = j - 2*d_model;
                const uint8_t* wj = W_out + (ptrdiff_t)jj * bpr_m;
                _graw[jj] = (float)gemv_2bit_row(_xi1, wj, d_model) * cs_o;
            }
        }
    } // barrier

    // PASO 6: fused MLGRU gates + residual (AVX2, identico a decode_layer_fused)
    {
        int i = 0;
        for (; i <= d_model-8; i+=8) {
            __m256 vf_r = _mm256_loadu_ps(_fraw+i);
            __m256 vc_r = _mm256_loadu_ps(_craw+i);
            __m256 vg_r = _mm256_loadu_ps(_graw+i);
            __m256 vh   = _mm256_loadu_ps(h+i);
            __m256 vx   = _mm256_loadu_ps(x_in+i);
            __m256 vf = _avx2_sigmoid_ps(vf_r);
            __m256 vc = _avx2_silu_ps(vc_r);
            __m256 vg = _avx2_sigmoid_ps(vg_r);
            __m256 vhn = _mm256_fmadd_ps(vf, _mm256_sub_ps(vh, vc), vc);
            __m256 vm  = _mm256_mul_ps(vg, vhn);
            _mm256_storeu_ps(h+i, vhn);
            _mm256_storeu_ps(_mout+i, vm);
            _mm256_storeu_ps(_xmid+i, _mm256_add_ps(vx, vm));
        }
        for (; i < d_model; i++) {
            float f  = 1.f/(1.f+expf(-_fraw[i]));
            float c  = _craw[i]/(1.f+expf(-_craw[i]));
            float g  = 1.f/(1.f+expf(-_graw[i]));
            float hn = c + f*(h[i]-c);
            h[i] = hn; _mout[i] = g*hn; _xmid[i] = x_in[i]+_mout[i];
        }
    }

    // PASO 7: RMSNorm2
    rms_norm_inline(_xmid, norm2_w, _xn2, d_model, norm_eps);

    // PASO 8: Cuantizar para gate/up
    float xs2     = quantize_x_avx2(_xn2, _xi2, d_model);
    float cs_gate = xs2 * s_gate;
    float cs_up   = xs2 * s_up;

    // PASO 9-10: GEMVs gate+up en parallel OMP
    int bpr_m = (d_model + 3) / 4;  // bytes por fila W_gate/W_up [d_ff x d_model]
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid      = omp_get_thread_num();
        int total2   = 2 * d_ff;
        int chunk    = (total2 + nthreads - 1) / nthreads;
        int jS       = tid * chunk;
        int jE       = jS + chunk < total2 ? jS + chunk : total2;

        for (int j = jS; j < jE; j++) {
            if (j < d_ff) {
                const uint8_t* wj = W_gate + (ptrdiff_t)j * bpr_m;
                _gater[j] = (float)gemv_2bit_row(_xi2, wj, d_model) * cs_gate;
            } else {
                int jj = j - d_ff;
                const uint8_t* wj = W_up + (ptrdiff_t)jj * bpr_m;
                _upr[jj] = (float)gemv_2bit_row(_xi2, wj, d_model) * cs_up;
            }
        }
    } // barrier

    // PASO 11: silu(gate)*up → hidden, cuantizar
    {
        int i = 0;
        for (; i <= d_ff-8; i+=8) {
            __m256 vg = _mm256_loadu_ps(_gater+i);
            __m256 vu = _mm256_loadu_ps(_upr+i);
            _mm256_storeu_ps(_hid+i, _mm256_mul_ps(_avx2_silu_ps(vg), vu));
        }
        for (; i < d_ff; i++) _hid[i] = _gater[i]/(1.f+expf(-_gater[i])) * _upr[i];
    }

    int bpr_ff = (d_ff + 3) / 4;   // bytes por fila W_down [d_model x d_ff]
    float hs   = quantize_x_avx2(_hid, _hi8, d_ff);
    float cs_down = hs * s_down;

    // PASO 12: down GEMV paralelo (2-bit)
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < d_model; j++) {
        const uint8_t* wj = W_down + (ptrdiff_t)j * bpr_ff;
        _ffno[j] = (float)gemv_2bit_row(_hi8, wj, d_ff) * cs_down;
    }

    // PASO 13: x_out = x_mid + ffn_out
    {
        int i = 0;
        for (; i <= d_model-8; i+=8)
            _mm256_storeu_ps(x_out+i,
                _mm256_add_ps(_mm256_loadu_ps(_xmid+i), _mm256_loadu_ps(_ffno+i)));
        for (; i < d_model; i++) x_out[i] = _xmid[i] + _ffno[i];
    }
}

} // extern "C"
