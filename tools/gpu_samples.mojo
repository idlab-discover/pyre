"""Deterministic benchmark producer, mirrored and byte-checked by corpus tooling."""
from numojo import NDArray
from std.math import nan


def mix(value: UInt32) -> UInt32:
    var x = value
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    return x


def fill_samples[
    dtype: DType
](mut input: NDArray[dtype], seed: Int, missing: Int, pattern: String) raises:
    for i in range(input.size):
        if pattern == "legacy":
            input.itemset(i, Scalar[dtype](Float64(i % 13 - 6) * 0.25))
        else:
            var index = i % input.shape[1] if pattern == "coherent" else i
            var h = mix(UInt32(index + seed))
            var value = Scalar[dtype](Float64(Int(h % 4096) - 2048) / 1024)
            if Int(mix(h ^ UInt32(0x9E3779B9)) % 100) < missing:
                value = nan[dtype]()
            input.itemset(i, value)
