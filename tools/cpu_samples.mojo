"""Untimed binary exchange with Python; values retain their exact bit patterns."""
from balsa.codec import read_file
from numojo import NDArray
from numojo.core.type_aliases import Shape
from std.sys import size_of


def samples[
    dtype: DType
](path: String, rows: Int, features: Int) raises -> NDArray[dtype]:
    var bytes = read_file(path)
    if len(bytes) != rows * features * size_of[Scalar[dtype]]():
        raise Error("Input byte count does not match shape and precision")
    var x = NDArray[dtype](Shape(rows, features))
    var source = bytes.unsafe_ptr().unsafe_bitcast[Scalar[dtype]]()
    for i in range(x.size):
        x.itemset(i, source[unsafe_offset=i])
    return x^


def save[dtype: DType](x: NDArray[dtype], path: String) raises:
    with open(path, "w") as file:
        file.write_all(
            Span(
                unsafe_ptr=x.unsafe_ptr().unsafe_bitcast[UInt8](),
                length=x.size * size_of[Scalar[dtype]](),
            )
        )
    print("shape", path, x.shape)
