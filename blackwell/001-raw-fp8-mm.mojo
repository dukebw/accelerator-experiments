# Original: https://github.com/StuartSul/gpu-experiments/blob/2e0cd7f5de2dcbf4551a9e834c0a3ce6f227fe02/blackwell/001-raw-fp8-mm.cu

# Generic notes on tcgen05 instructions
#
# - Tensor memory (TM) is on-chip memory (likely SRAM)
# - It is organized as 2D matrix (rows are called "lanes" and columns are called as is)
#     - On sm_100a, this is 128 x 512 per CTA, each cell 32-bit in size
# - TM address is 32-bit, where first 16 significant bits are lane index and next 16 are column index
# - TM must be allocated by a single warp in a CTA
#     - Allocation is done in columns only (all lanes in the columns are allocated)
#     - Granularity is (1) powers of 2 and (2) at least 32
# - Supported matrix multiply and accumulate shapes: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-shape
#     - For 1-CTA dense (MX)FP8 matrix multiply, K is always 32
#     - M and N are specified in the instruction descriptor
# - Data movement shapes are in format lane x bits: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-movement-shape
#     - Each movement type (16x32b, 32x32b, ...) has its unique way of how values across registers are spread throughout TM
# - A warp in a warpgroup can access only 1/4 of the lanes, and all columns of the TM
#     - Warp 0: lanes 0-31
#     - Warp 1: lanes 32-63
#     - Warp 2: lanes 64-95
#     - Warp 3: lanes 96-127

import itertools
from gpu.host import DeviceContext
from random import randn, seed

# Global dimension
comptime N = 128
comptime M = 128
comptime K = 32

# Tile dimension
comptime TILE_N = 128
comptime TILE_M = 128
comptime TILE_K = 32

# Quantization
comptime Q_BLOCK = 32
comptime NUM_BLOCKS = K // Q_BLOCK
comptime DEST_MAX: Float32 = 448.0

# Kernel
comptime SM_COUNT = 148
comptime WARP_THREADS = 32
comptime WARPGROUP_WARPS = 4
comptime WARPGROUP_THREADS = WARP_THREADS * WARPGROUP_WARPS
comptime NUM_WARPGROUPS = 2
comptime NUM_THREADS = WARPGROUP_THREADS * NUM_WARPGROUPS
comptime MAX_SHARED_MEMORY = 227000  # Hopper/Blackwell
comptime DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - 1000
comptime PIPELINE_STAGES = 4


fn quantize_fp8_blockwise(
    src: List[Float32],
    mut dst_fp8: List[Float8_e4m3fn],
    mut dst_sc: List[Float8_e8m0fnu],
    rows: Int,
    cols: Int,
    q_block: Int,
    dest_max: Float32,
):
    """Quantize a matrix to FP8 with block-wise scaling.

    Args:
        src: Source matrix in row-major layout (rows x cols).
        dst_fp8: Output FP8 quantized values (rows x cols).
        dst_sc: Output scales (rows x num_blocks where num_blocks = cols // q_block).
        rows: Number of rows in the matrix.
        cols: Number of columns (K dimension).
        q_block: Block size for quantization.
        dest_max: Max representable value in destination format (448.0 for e4m3fn).
    """
    num_blocks = cols // q_block

    for row, b_idx in itertools.product(range(rows), range(num_blocks)):
        # Get block absolute maximum.
        block_offset = (row * cols) + (b_idx * q_block)
        amax = abs(src[block_offset])

        for i in range(1, q_block):
            amax = max(amax, abs(src[block_offset + i]))

        # Compute scale: ceilf(log2f(amax / dest_max)) with round to +inf & clamp to [2^-127, 2^127].
        scale_idx = row * num_blocks + b_idx
        dst_sc[scale_idx] = Float8_e8m0fnu(amax / dest_max)

        # Quantize all elements in the block.
        scale_value = Float32(dst_sc[scale_idx])

        for i in range(q_block):
            dst_fp8[block_offset + i] = Float8_e4m3fn(
                src[block_offset + i] / scale_value
            )


def main() -> None:
    __comptime_assert (K % Q_BLOCK) == 0, "K must be divisible by Q_BLOCK"

    print("M =", M, ", N =", N, ", K =", K, ", Q_BLOCK =", Q_BLOCK)

    # Allocate host memory.
    h_A = List[Float32](unsafe_uninit_length=M * K)
    h_B = List[Float32](unsafe_uninit_length=N * K)
    h_A_fp8 = List[Float8_e4m3fn](unsafe_uninit_length=M * K)
    h_B_fp8 = List[Float8_e4m3fn](unsafe_uninit_length=N * K)
    h_A_sc = List[Float8_e8m0fnu](unsafe_uninit_length=M * NUM_BLOCKS)
    h_B_sc = List[Float8_e8m0fnu](unsafe_uninit_length=N * NUM_BLOCKS)

    h_C = List[Float32](unsafe_uninit_length=M * N)  # h_C - unused placeholder
    h_C_ref = List[Float32](unsafe_uninit_length=M * N)

    print("Allocated host memory!")

    seed(42)

    # Initialize host A and B matrices.
    randn(h_A.unsafe_ptr(), M * K, mean=0.0, standard_deviation=1.0)
    randn(h_B.unsafe_ptr(), N * K, mean=0.0, standard_deviation=1.0)

    # Quantize host A matrix (layout: M x K, scales: M x NUM_BLOCKS).
    quantize_fp8_blockwise(h_A, h_A_fp8, h_A_sc, M, K, Q_BLOCK, DEST_MAX)
    print("Quantized A matrix!")

    # Quantize host B matrix (layout: N x K, scales: N x NUM_BLOCKS).
    quantize_fp8_blockwise(h_B, h_B_fp8, h_B_sc, N, K, Q_BLOCK, DEST_MAX)
    print("Quantized B matrix!")

    # Compute reference result: C = A @ B^T
    # A is M x K, B is N x K (stored row-major, so B^T means we use B[n, k])
    # Result C is M x N
    for m, n in itertools.product(range(M), range(N)):
        accum: Float32 = 0.0

        for b_idx in range(NUM_BLOCKS):
            a_scale = h_A_sc[m * NUM_BLOCKS + b_idx]
            b_scale = h_B_sc[n * NUM_BLOCKS + b_idx]
            combined_scale = Float32(a_scale) * Float32(b_scale)

            for k in range(b_idx * Q_BLOCK, (b_idx + 1) * Q_BLOCK):
                a_val = Float32(h_A_fp8[m * K + k])
                b_val = Float32(h_B_fp8[n * K + k])
                accum += a_val * b_val * combined_scale

        h_C_ref[m * N + n] = accum

    print("Computed reference result!")

    # Print some sample values for verification
    print("\nSample A values (original vs quantized):")
    for i in range(min(4, M * K)):
        print(
            "  A[",
            i,
            "]: original=",
            h_A[i],
            ", fp8=",
            Float32(h_A_fp8[i]),
            ", scale=",
            Float32(h_A_sc[i // Q_BLOCK]),
        )

    print("\nSample B values (original vs quantized):")
    for i in range(min(4, N * K)):
        print(
            "  B[",
            i,
            "]: original=",
            h_B[i],
            ", fp8=",
            Float32(h_B_fp8[i]),
            ", scale=",
            Float32(h_B_sc[i // Q_BLOCK]),
        )

    print("\nSample reference C values (from quantized A and B):")
    for i in range(min(4, M)):
        for j in range(min(4, N)):
            print("  C_ref[", i, ",", j, "] =", h_C_ref[i * N + j])

    # Compute original (non-quantized) reference for comparison
    h_C_orig = List[Float32](unsafe_uninit_length=M * N)

    for m in range(M):
        for n in range(N):
            acc: Float32 = 0.0

            for k in range(K):
                acc += h_A[m * K + k] * h_B[n * K + k]

            h_C_orig[m * N + n] = acc

    print("\nSample original (non-quantized) C values:")
    for i in range(min(4, M)):
        for j in range(min(4, N)):
            print("  C_orig[", i, ",", j, "] =", h_C_orig[i * N + j])

    # Compute error between quantized and original reference
    max_abs_error: Float32 = 0.0
    max_rel_error: Float32 = 0.0

    for i in range(M * N):
        abs_error = abs(h_C_ref[i] - h_C_orig[i])
        if abs(h_C_orig[i]) > 1e-6:
            rel_error = abs_error / abs(h_C_orig[i])
            max_rel_error = max(max_rel_error, rel_error)
        max_abs_error = max(max_abs_error, abs_error)

    print("\nQuantization error analysis:")
    print("  Max absolute error:", max_abs_error)
    print("  Max relative error:", max_rel_error)

    with DeviceContext() as ctx:
        # Allocate device memory.
        d_A_fp8 = ctx.enqueue_create_buffer[DType.float8_e4m3fn](M * K)
        d_B_fp8 = ctx.enqueue_create_buffer[DType.float8_e4m3fn](K * N)
        d_A_sc = ctx.enqueue_create_buffer[DType.float8_e8m0fnu](M * NUM_BLOCKS)
        d_B_sc = ctx.enqueue_create_buffer[DType.float8_e8m0fnu](N * NUM_BLOCKS)
        d_C = ctx.enqueue_create_buffer[DType.float8_e8m0fnu](M * N)
        print("Allocated device memory")

        # Copy data to device.
        ctx.enqueue_copy(d_A_fp8, h_A_fp8.unsafe_ptr())
        ctx.enqueue_copy(d_B_fp8, h_B_fp8.unsafe_ptr())
        ctx.enqueue_copy(d_A_sc, h_A_sc.unsafe_ptr())
        ctx.enqueue_copy(d_B_sc, h_B_sc.unsafe_ptr())
        ctx.enqueue_memset(d_C, val=0)
        print("Copied data to device")

    print("\nDone!")
