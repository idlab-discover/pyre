"""Native Pyre comparison: full-output gate, calibration, raw timed intervals.

Loading, preparation, oracle checks and reporting are outside the timer.
Each timed CPU call allocates and disposes its output.
"""
from std.sys import argv
from std.time import perf_counter_ns
from std.benchmark import keep
from std.math import isfinite
from balsa import load_editable
from pyre import Forest
from numojo import NDArray
from numojo.core.type_aliases import Shape
from cpu_samples import samples


def gate[
    dtype: DType
](
    actual: NDArray[dtype], expected: NDArray[DType.float64], tolerance: Float64
) raises:
    if actual.size != expected.size:
        raise Error("Comparison output shape mismatch")
    var maximum = Float64(0)
    var scaled = Float64(0)
    for i in range(actual.size):
        var a = Float64(actual.item(i))
        var e = expected.item(i)
        if not isfinite(a) or not isfinite(e):
            raise Error("Nonfinite comparison output")
        var error = abs(a - e)
        var relative = error / (1 + abs(e))
        maximum = max(maximum, error)
        scaled = max(scaled, relative)
        if relative > tolerance:
            raise Error("Comparison oracle gate failed")
    print("gate", maximum, scaled)


def measure[
    call: def() raises capturing[_]
](count: Int, minimum: Float64) raises:
    for _ in range(3):
        call()
    var loops = 1
    while True:
        var start = perf_counter_ns()
        for _ in range(loops):
            call()
        var elapsed = perf_counter_ns() - start
        if Float64(elapsed) >= minimum * 1e9:
            break
        if loops >= 1 << 28:
            raise Error("Comparison calibration exceeded loop budget")
        loops *= 2
    for sample in range(count):
        var start = perf_counter_ns()
        for _ in range(loops):
            call()
        var elapsed = perf_counter_ns() - start
        print("sample", sample, loops, elapsed)


def run[dtype: DType](args: List[String]) raises:
    var device = args[1]
    var rows = Int(args[5])
    var features = Int(args[6])
    var threads = Int(args[8])
    var count = Int(args[9])
    var minimum = Float64(args[10])
    var tolerance = Float64(args[11])
    var group = Int(args[12])
    if rows < 1 or features < 1 or threads < 1 or count < 1 or minimum <= 0:
        raise Error("Invalid comparison dimensions or timing configuration")
    if group < 0 or group > 32 or (group != 0 and group & (group - 1) != 0):
        raise Error("GPU row group must be zero or a power of two <=32")
    var model = load_editable[dtype](args[2])
    var input = samples[dtype](args[4], rows, features)
    var expected = samples[DType.float64](args[7], rows, 1)
    if device == "cpu":
        if group != 0:
            raise Error("GPU row groups do not apply to CPU inference")
        var forest = Forest(model^)
        gate(
            forest.predict_margin(input, num_threads=threads),
            expected,
            tolerance,
        )
        print("meta", forest._cpu.kernel(rows), 0)

        def predict() raises capturing:
            var result = forest.predict_margin(input, num_threads=threads)
            keep(result)

        measure[predict](count, minimum)
    else:
        raise Error("This comparison worker supports CPU only")


def main() raises:
    var args = List[String]()
    for arg in argv():
        args.append(String(arg))
    if len(args) != 13:
        raise Error(
            "Usage: comparison-worker DEVICE MODEL DTYPE INPUT ROWS FEATURES"
            " ORACLE THREADS SAMPLES MIN_SECONDS TOLERANCE GPU_ROW_GROUP"
        )
    if args[3] == "float32":
        run[DType.float32](args)
    elif args[3] == "float64":
        run[DType.float64](args)
    else:
        raise Error("Comparison dtype must be float32 or float64")
