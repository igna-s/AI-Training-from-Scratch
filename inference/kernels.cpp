#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

extern "C" {

// ternary_lut_matmul_omp: Optimizacion Extrema Estilo Llama.cpp / BitNet
// Usa memoria Row-Major precompilada (w_packed de shape [d_out, n_groups])
// Estrategia:
// 1. Precomputar todas las LUT de n_groups [n_groups, 256] en la L2 Cache.
// 2. Multi-thread (OpenMP) iterando sobre las filas d_out.
// 3. Cache-friendly access.
EXPORT void ternary_lut_matmul_omp(
    const float* x,             // [M, d_in]
    const uint8_t* w_packed,    // [d_out, n_groups] Contiguo y en Row-Major
    float w_scale,
    const float* decode_table,  // [128, group_size]
    float* output,              // [M, d_out]
    int M, int d_in, int d_out, int group_size
) {
    int n_groups = d_in / group_size;

    // Buffer temporal para LUTs. Para n_groups=256, 256*256*4 bytes = 256KB
    // Usamos OpenMP para compartir memoria o thread local si M > 1
    // Para simplificar, operamos M por ciclo (M suele ser 1 en inferencia)
    
    for(int m = 0; m < M; m++) {
        const float* xm = x + m * d_in;
        float* outm = output + m * d_out;
        
        // --- FASE 1: PRE-CALCULAR LUTs ---
        // Eliminar overhead masivo de `new float[]` usando un buffer estático por hilo global.
        // Como M=1 normalmente, evitamos bloqueos.
        static thread_local float* thread_luts = nullptr;
        static thread_local int thread_luts_cap = 0;
        
        int req_size = n_groups * 256;
        if (req_size > thread_luts_cap) {
            delete[] thread_luts;
            thread_luts = new float[req_size];
            thread_luts_cap = req_size;
        }
        float* luts = thread_luts;
        
        #pragma omp parallel for schedule(static)
        for(int g = 0; g < n_groups; g++) {
            const float* x_grp = xm + g * group_size;
            float* lut_g = luts + g * 256;
            
            for(int c = 0; c < 128; c++) {
                float sum = 0.0f;
                // Unroll group_size=4 que es el por defecto
                if (group_size == 4) {
                    const float* d_row = decode_table + c * 4;
                    sum = x_grp[0]*d_row[0] + x_grp[1]*d_row[1] + x_grp[2]*d_row[2] + x_grp[3]*d_row[3];
                } else {
                    const float* d_row = decode_table + c * group_size;
                    for(int i = 0; i < group_size; i++) {
                        sum += x_grp[i] * d_row[i];
                    }
                }
                lut_g[c] = sum;
                lut_g[128 + c] = -sum;
            }
        }
        
        // --- FASE 2: GATHER REDUCTION ---
        // Aqui w_packed esta en layout [d_out, n_groups], por tanto
        // leer una fila de longitud `n_groups` para un `d_out` determinado
        // es 100% secuencial y amigable para el Prefetcher del CPU.
        #pragma omp parallel for schedule(static)
        for(int j = 0; j < d_out; j++) {
            float out_val = 0.0f;
            const uint8_t* w_row = w_packed + j * n_groups;
            
            // Loop de acumulacion central (Hot Path)
            // Manual Data Dependency Unrolling x8 (Software Pipelining)
            float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
            float sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0;
            int g = 0;
            
            for(; g <= n_groups - 8; g += 8) {
                sum0 += luts[(g+0) * 256 + w_row[g+0]];
                sum1 += luts[(g+1) * 256 + w_row[g+1]];
                sum2 += luts[(g+2) * 256 + w_row[g+2]];
                sum3 += luts[(g+3) * 256 + w_row[g+3]];
                sum4 += luts[(g+4) * 256 + w_row[g+4]];
                sum5 += luts[(g+5) * 256 + w_row[g+5]];
                sum6 += luts[(g+6) * 256 + w_row[g+6]];
                sum7 += luts[(g+7) * 256 + w_row[g+7]];
            }
            for(; g < n_groups; g++) {
                sum0 += luts[g * 256 + w_row[g]];
            }
            
            out_val += sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;
            outm[j] = out_val * w_scale;
        }
    }
}

// mlgru_scan_fusion: Fusión en RAM del update recurrente 
EXPORT void mlgru_scan_fusion(
    const float* c, // [B, d_model] candidate
    const float* f, // [B, d_model] forget
    const float* g, // [B, d_model] output gate
    float* h,       // [B, d_model] hidden state (mutable IN/OUT)
    float* o,       // [B, d_model] output final
    int B, int d_model
) {
    int total_elems = B * d_model;
    #pragma omp parallel for simd schedule(static)
    for(int i = 0; i < total_elems; i++) {
        // lerp: f*h + (1-f)*c = c + f*(h - c)
        float h_prev = h[i];
        float c_i = c[i];
        float h_new = c_i + f[i] * (h_prev - c_i);
        
        h[i] = h_new; // update estado in-place
        o[i] = g[i] * h_new;
    }
}

} // extern "C"
