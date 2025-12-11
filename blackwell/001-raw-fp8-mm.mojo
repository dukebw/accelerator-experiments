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


def main() -> None:
    pass
