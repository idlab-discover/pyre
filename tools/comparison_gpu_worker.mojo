"""Resident GPU comparisons using the shared oracle gate and interval protocol."""
from std.sys import argv
from std.benchmark import keep
from balsa import load_editable
from max.gpu.host import DeviceContext
from pyre.gpu import GpuForest, upload
from cpu_samples import samples
from comparison_worker import gate, measure


def run[dtype: DType](args: List[String]) raises:
    var rows = Int(args[5])
    var features = Int(args[6])
    var count = Int(args[9])
    var minimum = Float64(args[10])
    var tolerance = Float64(args[11])
    var group = Int(args[12])
    if args[1] != "gpu" or rows < 1 or features < 1 or count < 1 or minimum <= 0:
        raise Error("Invalid GPU comparison configuration")
    if group < 0 or group > 32 or (group != 0 and group & (group - 1) != 0):
        raise Error("GPU row group must be zero or a power of two <=32")
    var model = load_editable[dtype](args[2])
    var input = samples[dtype](args[4], rows, features)
    var expected = samples[DType.float64](args[7], rows, 1)
    var ctx = DeviceContext()
    var forest = GpuForest.prepare(model, ctx)
    var batch = upload(input, ctx)
    var plan = forest.plan(rows)
    if group != 0:
        plan.chunk_shift = 0
        while (1 << plan.chunk_shift) < group:
            plan.chunk_shift += 1
        plan.rows_per_thread = min(plan.rows_per_thread, group)
        forest.use_plan(rows, plan)
    gate(forest.predict_margin(batch).to_host(), expected, tolerance)
    print("meta", -1, 1 << plan.chunk_shift)
    print(
        "plan", plan.chunk_shift, plan.threads, plan.rows_per_thread,
        plan.trees_per_thread, Int(plan.staged),
    )

    def predict() raises capturing:
        # predict_margin joins its stream before returning the device output.
        var result = forest.predict_margin(batch)
        keep(result)

    measure[predict](count, minimum)


def main() raises:
    var args = List[String]()
    for arg in argv():
        args.append(String(arg))
    if len(args) != 13:
        raise Error(
            "Usage: comparison-gpu-worker gpu MODEL DTYPE INPUT ROWS FEATURES"
            " ORACLE THREADS SAMPLES MIN_SECONDS TOLERANCE GPU_ROW_GROUP"
        )
    if args[3] == "float32":
        run[DType.float32](args)
    elif args[3] == "float64":
        run[DType.float64](args)
    else:
        raise Error("Comparison dtype must be float32 or float64")
