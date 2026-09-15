"""Private native producer experiment; no public staging ownership API."""
from layout import TileTensor, TensorLayout
from std.math import nan
from gpu_samples import mix


def fill_pinned[
    dtype: DType, LT: TensorLayout
](
    tensor: TileTensor[dtype, LT, MutAnyOrigin],
    size: Int,
    features: Int,
    seed: Int,
    missing: Int,
    pattern: String,
):
    comptime assert tensor.flat_rank == 1
    for i in range(size):
        var value: Scalar[dtype]
        if pattern == "legacy":
            value = Scalar[dtype](Float64(i % 13 - 6) * 0.25)
        else:
            var index = i % features if pattern == "coherent" else i
            var h = mix(UInt32(index + seed))
            value = Scalar[dtype](Float64(Int(h % 4096) - 2048) / 1024)
            if Int(mix(h ^ UInt32(0x9E3779B9)) % 100) < missing:
                value = nan[dtype]()
        tensor[i] = rebind[tensor.ElementType](value)
