#include "buf.h"

// 3d
#define BLOCK_X 8
#define BLOCK_Y 4
#define BLOCK_Z 4

namespace cc3d
{
    __global__ void init_labeling(int32_t *label, const uint32_t W, const uint32_t H, const uint32_t D) {
        const uint32_t x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const uint32_t y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        const uint32_t z = (blockIdx.z * blockDim.z + threadIdx.z) * 2;

        const uint32_t idx = z * W * H + y * W + x;

        if (x < W && y < H && z < D)
            label[idx] = idx;
    }

    __global__ void merge(uint8_t * const img, int32_t *label, uint8_t *last_cube_fg, const uint32_t W, const uint32_t H, const uint32_t D)
    {
        const uint32_t x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const uint32_t y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        const uint32_t z = (blockIdx.z * blockDim.z + threadIdx.z) * 2;
        const uint32_t stepz = W * H;
        const uint32_t idx = z * stepz + y * W + x;

        if (x >= W || y >= H || z >= D)
            return;

        uint64_t P = 0;
        uint8_t fg = 0;

        uint16_t buffer;
#define P0 0x77707770777
#define CHECK_BUF(P_shift, fg_shift, P_shift2, fg_shift2)  \
    if (buffer & 1) {                                      \
        P |= P0 << (P_shift);                              \
        fg |= 1 << (fg_shift);                             \
    }                                                      \
    if (buffer & (1 << 8)) {                               \
        P |= P0 << (P_shift2);                             \
        fg |= 1 << (fg_shift2);                            \
    }

        if (x + 1 < W) {
            buffer = *reinterpret_cast<uint16_t*>(img + idx);
            CHECK_BUF(0, 0, 1, 1)

            if (y + 1 < H) {
                buffer = *reinterpret_cast<uint16_t*>(img + idx + W);
                CHECK_BUF(4, 2, 5, 3)
            }

            if (z + 1 < D) {
                buffer = *reinterpret_cast<uint16_t*>(img + idx + stepz);
                CHECK_BUF(16, 4, 17, 5)

                if (y + 1 < H) {
                    buffer = *reinterpret_cast<uint16_t*>(img + idx + stepz + W);
                    CHECK_BUF(20, 6, 21, 7)
                }
            }
        }
#undef CHECK_BUF

        // Store fg voxels bitmask into memory
        if (x + 1 < W)          label[idx + 1] = fg;
        else if (y + 1 < H)     label[idx + W] = fg;
        else if (z + 1 < D)     label[idx + stepz] = fg;
        else                    *last_cube_fg = fg;

        // checks on borders

        if (x == 0)             P &= 0xEEEEEEEEEEEEEEEE;
        if (x + 1 >= W)         P &= 0x3333333333333333;
        else if (x + 2 >= W)    P &= 0x7777777777777777;

        if (y == 0)             P &= 0xFFF0FFF0FFF0FFF0;
        if (y + 1 >= H)         P &= 0x00FF00FF00FF00FF;
        else if (y + 2 >= H)    P &= 0x0FFF0FFF0FFF0FFF;

        if (z == 0)             P &= 0xFFFFFFFFFFFF0000;
        if (z + 1 >= D)         P &= 0x00000000FFFFFFFF;
        // else if (z + 2 >= D)    P &= 0x0000FFFFFFFFFFFF;

        if (P > 0) {
            // Lower plane
            const uint32_t img_idx = idx - stepz;
            const uint32_t label_idx = idx - 2 * stepz;

            if (hasBit(P, 0) && img[img_idx - W - 1])
                union_(label, idx, label_idx - 2 * W - 2);

            if ((hasBit(P, 1) && img[img_idx - W]) || (hasBit(P, 2) && img[img_idx - W + 1]))
                union_(label, idx, label_idx - 2 * W);

            if (hasBit(P, 3) && img[img_idx - W + 2])
                union_(label, idx, label_idx - 2 * W + 2);

            if ((hasBit(P, 4) && img[img_idx - 1]) || (hasBit(P, 8) && img[img_idx + W - 1]))
                union_(label, idx, label_idx - 2);

            if ((hasBit(P, 5) && img[img_idx]) || (hasBit(P, 6) && img[img_idx + 1]) || \
                (hasBit(P, 9) && img[img_idx + W]) || (hasBit(P, 10) && img[img_idx + W + 1]))
                union_(label, idx, label_idx);

            if ((hasBit(P, 7) && img[img_idx + 2]) || (hasBit(P, 11) && img[img_idx + W + 2]))
                union_(label, idx, label_idx + 2);

            if (hasBit(P, 12) && img[img_idx + 2 * W - 1])
                union_(label, idx, label_idx + 2 * W - 2);

            if ((hasBit(P, 13) && img[img_idx + 2 * W]) || (hasBit(P, 14) && img[img_idx + 2 * W + 1]))
                union_(label, idx, label_idx + 2 * W);

            if (hasBit(P, 15) && img[img_idx + 2 * W + 2])
                union_(label, idx, label_idx + 2 * W + 2);

            // Current planes
            if ((hasBit(P, 16) && img[idx - W - 1]) || (hasBit(P, 32) && img[idx + stepz - W - 1]))
                union_(label, idx, idx - 2 * W - 2);

            if ((hasBit(P, 17) && img[idx - W]) || (hasBit(P, 18) && img[idx - W + 1]) || \
                (hasBit(P, 33) && img[idx + stepz - W]) || (hasBit(P, 34) && img[idx + stepz - W + 1]))
                union_(label, idx, idx - 2 * W);

            if ((hasBit(P, 19) && img[idx - W + 2]) || (hasBit(P, 35) && img[idx + stepz - W + 2]))
                union_(label, idx, idx - 2 * W + 2);

            if ((hasBit(P, 20) && img[idx - 1]) || (hasBit(P, 24) && img[idx + W - 1]) || \
                (hasBit(P, 36) && img[idx + stepz - 1]) || (hasBit(P, 40) && img[idx + stepz + W - 1]))
                union_(label, idx, idx - 2);

        }

    }

    __global__ void compression(int32_t *label, const uint32_t W, const uint32_t H, const uint32_t D)
    {
        const uint32_t x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const uint32_t y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        const uint32_t z = (blockIdx.z * blockDim.z + threadIdx.z) * 2;

        const uint32_t idx = z * W * H + y * W + x;

        if (x < W && y < H && z < D)
            find_n_compress(label, idx);
    }


    __global__ void final_labeling(int32_t *label, uint8_t *last_cube_fg, const uint32_t W, const uint32_t H, const uint32_t D)
    {
        const uint32_t x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const uint32_t y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
        const uint32_t z = (blockIdx.z * blockDim.z + threadIdx.z) * 2;

        const uint32_t idx = z * W * H + y * W + x;

        if (x >= W || y >= H || z >= D)
            return;

        int tmp;
        uint8_t fg;
        uint64_t buf;

        if (x + 1 < W) {
            buf = *reinterpret_cast<uint64_t*>(label + idx);
            tmp = (buf & (0xFFFFFFFF)) + 1;
            fg = (buf >> 32) & 0xFFFFFFFF;
        }
        else {
            tmp = label[idx] + 1;
            if (y + 1 < H)       fg = label[idx + W];
            else if (z + 1 < D)  fg = label[idx + W * H];
            else                 fg = *last_cube_fg;
        }

        if (x + 1 < W) {
            *reinterpret_cast<uint64_t*>(label + idx) =
                (static_cast<uint64_t>(((fg >> 1) & 1) * tmp) << 32) | (((fg >> 0) & 1) * tmp);

            if (y + 1 < H)
                *reinterpret_cast<uint64_t*>(label + idx + W) =
                    (static_cast<uint64_t>(((fg >> 3) & 1) * tmp) << 32) | (((fg >> 2) & 1) * tmp);
            if (z + 1 < D) {
                *reinterpret_cast<uint64_t*>(label + idx + W * H) =
                    (static_cast<uint64_t>(((fg >> 5) & 1) * tmp) << 32) | (((fg >> 4) & 1) * tmp);

                if (y + 1 < H)
                    *reinterpret_cast<uint64_t*>(label + idx + W * H + W) =
                        (static_cast<uint64_t>(((fg >> 7) & 1) * tmp) << 32) | (((fg >> 6) & 1) * tmp);
            }
        }
        else {
            label[idx] = ((fg >> 0) & 1) * tmp;
            if (y + 1 < H)
                label[idx + (W)] = ((fg >> 2) & 1) * tmp;

            if (z + 1 < D) {
                label[idx + W * H] = ((fg >> 4) & 1) * tmp;
                if (y + 1 < H)
                    label[idx + W * H + W] = ((fg >> 6) & 1) * tmp;
            }
        }
    } // final_labeling
} // namespace cc3d


torch::Tensor connected_componnets_labeling_3d(const torch::Tensor &input) {
    AT_ASSERTM(input.is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(input.ndimension() == 3, "input must be a  [D, H, W] shape");
    AT_ASSERTM(input.scalar_type() == torch::kUInt8, "input must be a uint8 type");

    const uint32_t D = input.size(-3);
    const uint32_t H = input.size(-2);
    const uint32_t W = input.size(-1);

    AT_ASSERTM((D % 2) == 0, "shape must be a even number");
    AT_ASSERTM((H % 2) == 0, "shape must be a even number");
    AT_ASSERTM((W % 2) == 0, "shape must be a even number");

    // label must be uint32_t
    auto label_options = torch::TensorOptions().dtype(torch::kInt32).device(input.device());
    torch::Tensor label = torch::zeros({D, H, W}, label_options);

    dim3 grid = dim3(((W + 1) / 2 + BLOCK_X - 1) / BLOCK_X, ((H + 1) / 2 + BLOCK_Y - 1) / BLOCK_Y, ((D + 1) / 2 + BLOCK_Z - 1) / BLOCK_Z);
    dim3 block = dim3(BLOCK_X, BLOCK_Y, BLOCK_Z);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    uint8_t *last_cube_fg;
    bool allocated_last_cude_fg_ = false;
    if ((W % 2 == 1) && (H % 2 == 1) && (D % 2 == 1)) {
        if (W > 1 && H > 1)
            last_cube_fg = reinterpret_cast<uint8_t*>(
                label.data_ptr<uint8_t>() + (D - 1) * W * H + (H - 2) * W
            ) + W - 2;
        else if (W > 1 && D > 1)
            last_cube_fg = reinterpret_cast<uint8_t*>(
                label.data_ptr<uint8_t>() + (D - 2) * W * H + (H - 1) * W
            ) + W - 2;
        else if (H > 1 && D > 1)
            last_cube_fg = reinterpret_cast<uint8_t*>(
                label.data_ptr<uint8_t>() + (D - 2) * W * H + (H - 2) * W
            ) + W - 1;
        else {
            cudaMalloc(&last_cube_fg, sizeof(uint8_t));
            allocated_last_cude_fg_ = true;
        }
    }

    cc3d::init_labeling<<<grid, block, 0, stream>>>(
        label.data_ptr<int32_t>(), W, H, D
    );
    cc3d::merge<<<grid, block, 0, stream>>>(
        input.data_ptr<uint8_t>(),
        label.data_ptr<int32_t>(),
        last_cube_fg,
        W, H, D
    );
    cc3d::compression<<<grid, block, 0, stream>>>(
        label.data_ptr<int32_t>(), W, H, D
    );
    cc3d::final_labeling<<<grid, block, 0, stream>>>(
        label.data_ptr<int32_t>(),
        last_cube_fg,
        W, H, D
    );

    if (allocated_last_cude_fg_)
        cudaFree(last_cube_fg);
    return label;
}
