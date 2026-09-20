"""Complete CPU inference calls; baseline gates and a native monotonic timer."""
from std.sys import argv
from std.time import perf_counter_ns
from std.benchmark import keep
from balsa import load
from pyre import Forest
from pyre._cpu import _CPUPlan, _GENERAL, _DIRECT
from cpu_samples import samples, save
from numojo import NDArray


def predict[
    dtype: DType
](forest: Forest[dtype], x: NDArray[dtype], mode: String, threads: Int) raises:
    if mode == "margin":
        var r = forest.predict_margin(x, num_threads=threads)
        keep(r)
    elif mode == "class":
        var r = forest.predict_class(x, num_threads=threads)
        keep(r)
    elif mode == "proba":
        var r = forest.predict_proba(x, num_threads=threads)
        keep(r)
    else:
        raise Error("Call must be margin, class, or proba")


def run[dtype: DType](args: List[String]) raises:
    var threads = Int(args[6])
    var model = load[dtype](args[1])
    var x = samples[dtype](args[3], Int(args[4]), Int(args[5]))
    var forest = Forest(model^)
    # Development controls validate the route without changing prepared storage.
    if args[10] == "generic":
        forest._cpu._use_strategy(_GENERAL)
    elif args[10] == "direct":
        forest._cpu._use_strategy(_DIRECT)
    elif args[10] != "auto":
        raise Error("Kernel must be auto, direct, or generic")
    if args[7] != "-":
        var p = forest.predict_proba(x, num_threads=threads)
        var m = forest.predict_margin(x, num_threads=threads)
        var c = forest.predict_class(x, num_threads=threads)
        save(p, args[7] + "/probabilities.bin")
        save(m, args[7] + "/margins.bin")
        save(c, args[7] + "/labels.bin")
        return

    for _ in range(3):
        predict(forest, x, args[9], threads)

    var loops = Int(args[8])
    var start = perf_counter_ns()
    for _ in range(loops):
        predict(forest, x, args[9], threads)
    print(Float64(perf_counter_ns() - start) / 1e9)


def main() raises:
    var args = List[String]()
    for arg in argv():
        args.append(String(arg))
    if len(args) != 11:
        raise Error(
            "Usage: worker MODEL DTYPE INPUT ROWS FEATURES THREADS"
            " OUTPUT_OR_DASH LOOPS CALL KERNEL"
        )
    if args[2] == "float32":
        run[DType.float32](args)
    elif args[2] == "float64":
        run[DType.float64](args)
    else:
        raise Error("Unsupported precision")
