from gpu import block_dim, block_idx, grid_dim, thread_idx
from gpu.host import DeviceContext


fn print_thread() -> None:
    print(
        "Block:",
        block_idx.x,
        "/",
        grid_dim.x,
        " thread: ",
        thread_idx.x,
        "/",
        block_dim.x,
    )


def main() -> None:
    with DeviceContext() as ctx:
        ctx.enqueue_function_checked[print_thread, print_thread](
            grid_dim=148, block_dim=1
        )
        ctx.synchronize()
