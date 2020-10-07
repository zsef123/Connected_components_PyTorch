#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>

namespace {

template <typename T>
__device__ __forceinline__ unsigned char hasBit(T bitmap, unsigned char pos) {
    return (bitmap >> pos) & 1;
}


__device__ int32_t find(const int32_t *s_buf, int32_t n) {
    while (s_buf[n] != n)
        n = s_buf[n];
    return n;
}

__device__ int32_t find_n_compress(int32_t *s_buf, int32_t n) {
    const int32_t id = n;
    while (s_buf[n] != n) {
        n = s_buf[n];
        s_buf[id] = n;
    }
    return n;
}

__device__ void union_(int32_t *s_buf, int32_t a, int32_t b)
{
    bool done;
    do
    {
        a = find(s_buf, a);
        b = find(s_buf, b);

        if (a < b) {
            int32_t old = atomicMin(s_buf + b, a);
            done = (old == b);
            b = old;
        }
        else if (b < a){
            int32_t old = atomicMin(s_buf + a, b);
            done = (old == a);
            a = old;
        }
        else
            done = true;

    } while (!done);
}

}

namespace cc2d
{
    __global__ void init_labeling(int32_t *label, const uint32_t W, const uint32_t H);
    __global__ void merge(uint8_t *img, int32_t *label, const uint32_t W, const uint32_t H);
    __global__ void compression(int32_t *label, const int32_t W, const int32_t H);
    __global__ void final_labeling(const uint8_t *img, int32_t *label, const int32_t W, const int32_t H);
}

namespace cc3d
{
    __global__ void init_labeling(int32_t *label, const uint32_t W, const uint32_t H, const uint32_t D);
    __global__ void merge(uint8_t * const img, int32_t *label, uint8_t *last_cube_fg, const uint32_t W, const uint32_t H, const uint32_t D);
    __global__ void compression(int32_t *label, const uint32_t W, const uint32_t H, const uint32_t D);
    __global__ void final_labeling(int32_t *label, uint8_t *last_cube_fg, const uint32_t W, const uint32_t H, const uint32_t D);
}

torch::Tensor connected_componnets_labeling_2d(const torch::Tensor &input);
torch::Tensor connected_componnets_labeling_3d(const torch::Tensor &input);
