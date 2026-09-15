"""Native GPU phases with parity gates and per-phase duration calibration."""
from std.sys import argv
from std.time import perf_counter_ns
from std.benchmark import keep
from std.math import nan, inf
from balsa import decode
from balsa.codec import read_file
from max.gpu.host import DeviceContext
from pyre import Forest
from pyre.gpu import GpuForest, GpuBatch, upload, upload_into
from pyre._gpu_storage import _Join
from numojo import NDArray
from numojo.core.type_aliases import Shape
from gpu_samples import fill_samples
from gpu_staging import fill_pinned
from layout import TileTensor, row_major


def number(value: StringSlice) raises -> Float64:
    if value == "nan":
        return nan[DType.float64]()
    if value == "inf":
        return inf[DType.float64]()
    if value == "-inf":
        return -inf[DType.float64]()
    return Float64(value)


def close(actual: Float64, expected: Float64, tolerance: Float64) raises:
    if not (abs(actual - expected) <= tolerance * (1 + abs(expected))):
        raise Error(
            "GPU benchmark parity failure: "
            + String(actual)
            + " vs "
            + String(expected)
        )


def check_oracle[
    dtype: DType
](
    path: String,
    cpu: Forest[dtype],
    forest: GpuForest[dtype],
    ctx: DeviceContext,
    tolerance: Float64,
) raises -> Int:
    if path == "-":
        return 0
    var contents: String
    with open(path, "r") as file:
        contents = file.read()
    var fields = contents.split()
    var rows = Int(fields[0])
    var features = Int(fields[1])
    if (
        features != forest.num_features()
        or rows < 1
        or len(fields) != 2 + rows * (features + 2)
    ):
        raise Error("Invalid GTIL oracle shape")
    var input = NDArray[dtype](Shape(rows, features))
    for row in range(rows):
        for col in range(features):
            input.itemset(
                row * features + col,
                Scalar[dtype](number(fields[2 + row * (features + 2) + col])),
            )
    var batch = upload(input, ctx)
    var gpu_margin = forest.predict_margin(batch).to_host()
    var cpu_margin = cpu.predict_margin(input, num_threads=1)
    for row in range(rows):
        var expected = number(fields[2 + row * (features + 2) + features])
        close(Float64(gpu_margin.item(row)), expected, tolerance)
        close(Float64(cpu_margin.item(row)), expected, tolerance)
    if forest.is_classifier():
        var gpu_p = forest.predict_proba(batch).to_host()
        var cpu_p = cpu.predict_proba(input, num_threads=1)
        for row in range(rows):
            var expected = number(fields[3 + row * (features + 2) + features])
            close(Float64(gpu_p.item(2 * row + 1)), expected, tolerance)
            close(Float64(cpu_p.item(2 * row + 1)), expected, tolerance)
    return rows


def run[
    dtype: DType
](
    path: String,
    rows: Int,
    initial_loops: Int,
    phase: String,
    seed: Int,
    missing: Int,
    pattern: String,
    oracle: String,
    min_seconds: Float64,
    tolerance: Float64,
) raises:
    var t = perf_counter_ns()
    var ctx = DeviceContext()
    var context_ns = perf_counter_ns() - t
    t = perf_counter_ns()
    var bytes = read_file(path)
    var read_ns = perf_counter_ns() - t
    t = perf_counter_ns()
    var model = decode[dtype](bytes^)
    var decode_ns = perf_counter_ns() - t
    t = perf_counter_ns()
    var forest = GpuForest.prepare(model, ctx)
    var prepare_ns = perf_counter_ns() - t
    t = perf_counter_ns()
    var input = NDArray[dtype](Shape(rows, forest.num_features()))
    fill_samples(input, seed, missing, pattern)
    var input_ns = perf_counter_ns() - t
    t = perf_counter_ns()
    var batch = GpuBatch[dtype](ctx, input.shape)
    var batch_allocate_ns = perf_counter_ns() - t
    t = perf_counter_ns()
    upload_into(input, batch)
    var upload_ns = perf_counter_ns() - t
    t = perf_counter_ns()
    var result = forest.predict_margin(batch)
    var first_ns = perf_counter_ns() - t
    t = perf_counter_ns()
    var host = result.to_host()
    var download_ns = perf_counter_ns() - t
    var cpu = Forest(model.copy())
    var expected = cpu.predict_margin(input, num_threads=1)
    for row in range(rows):
        close(Float64(host.item(row)), Float64(expected.item(row)), tolerance)
    if forest.is_classifier():
        var gp = forest.predict_proba(batch).to_host()
        var cp = cpu.predict_proba(input, num_threads=1)
        var gc = forest.predict_class(batch).to_host()
        for row in range(rows):
            close(
                Float64(gp.item(2 * row)), Float64(cp.item(2 * row)), tolerance
            )
            close(
                Float64(gp.item(2 * row + 1)),
                Float64(cp.item(2 * row + 1)),
                tolerance,
            )
            if gc.item(row) != Int64(gp.item(2 * row + 1) > 0.5):
                raise Error("GPU class/margin disagreement")
    var oracle_rows = check_oracle(oracle, cpu, forest, ctx, tolerance)
    t = perf_counter_ns()
    var staging = ctx.enqueue_create_host_buffer[dtype](input.size)
    ctx.synchronize()
    var pinned_allocate_ns = perf_counter_ns() - t
    var staging_layout = row_major(input.size)
    var staging_tensor = TileTensor(staging, staging_layout)
    comptime assert staging_tensor.flat_rank == 1
    fill_pinned(
        staging_tensor,
        input.size,
        forest.num_features(),
        seed,
        missing,
        pattern,
    )
    for i in range(input.size):
        var a = Float64(staging_tensor[i])
        var b = Float64(input.item(i))
        if not (a == b or (a != a and b != b)):
            raise Error("Pinned/NuMojo producer values differ")
    # Both reused transfer paths must pass before any timed phase.
    for path_b in [True, False]:
        if path_b:
            upload_into(input, batch)
        else:
            with _Join(ctx):
                if input.size:
                    ctx.enqueue_copy(batch._buffer, staging)
        forest.predict_margin_into(batch, result)
        var again = result.to_host()
        for row in range(rows):
            close(
                Float64(again.item(row)), Float64(expected.item(row)), tolerance
            )
    if phase == "validate":
        print("validated", rows, oracle_rows)
        return
    for _ in range(3):
        forest.predict_margin_into(batch, result)
    var encoded = read_file(path)
    for measured in [
        "device",
        "reuse",
        "allocate",
        "upload",
        "download",
        "prepare",
        "read",
        "decode",
        "samples",
        "end_to_end",
        "pipeline",
        "upload_into",
        "pinned_fill",
        "pinned_upload",
        "pinned_allocate",
        "batch_allocate",
        "ingest_a",
        "ingest_b",
        "ingest_c",
    ]:
        if phase != "all":
            var selected = False
            for requested in phase.split(","):
                if requested == measured:
                    selected = True
            if not selected:
                continue
        var loops = initial_loops
        var seconds: Float64
        # First pass warms this phase, subsequent passes calibrate independently.
        var warmed = False
        while True:
            if measured == "device":
                forest._check(batch, result)

                def launch() raises {forest, batch, mut result}:
                    forest._enqueue[dtype, 1, 0](batch, result)

                var elapsed: Int
                with _Join(ctx):
                    elapsed = ctx.execution_time(launch, loops)
                seconds = Float64(elapsed) / 1e9
            else:
                t = perf_counter_ns()
                for _ in range(loops):
                    if measured == "reuse":
                        forest.predict_margin_into(batch, result)
                    elif measured == "allocate":
                        var y = forest.predict_margin(batch)
                        keep(y)
                    elif measured == "upload":
                        var x = upload(input, ctx)
                        keep(x)
                    elif measured == "download":
                        var y = result.to_host()
                        keep(y)
                    elif measured == "prepare":
                        var f = GpuForest.prepare(model, ctx)
                        keep(f)
                    elif measured == "read":
                        var data = read_file(path)
                        keep(data)
                    elif measured == "decode":
                        var f = decode[dtype](encoded.copy())
                        keep(f)
                    elif measured == "samples":
                        var x = NDArray[dtype](
                            Shape(rows, forest.num_features())
                        )
                        fill_samples(x, seed, missing, pattern)
                        keep(x)
                    elif measured == "end_to_end":
                        var x = upload(input, ctx)
                        var y = forest.predict_margin(x)
                        var h = y.to_host()
                        keep(h)
                    elif measured == "upload_into":
                        upload_into(input, batch)
                    elif measured == "pinned_fill":
                        fill_pinned(
                            staging_tensor,
                            input.size,
                            forest.num_features(),
                            seed,
                            missing,
                            pattern,
                        )
                    elif measured == "pinned_upload":
                        with _Join(ctx):
                            if input.size:
                                ctx.enqueue_copy(batch._buffer, staging)
                    elif measured == "pinned_allocate":
                        var h = ctx.enqueue_create_host_buffer[dtype](
                            input.size
                        )
                        ctx.synchronize()
                        keep(h)
                    elif measured == "batch_allocate":
                        var b = GpuBatch[dtype](ctx, input.shape)
                        keep(b)
                    elif measured == "ingest_a":
                        var x = NDArray[dtype](
                            Shape(rows, forest.num_features())
                        )
                        fill_samples(x, seed, missing, pattern)
                        var b = upload(x, ctx)
                        forest.predict_margin_into(b, result)
                        var h = result.to_host()
                        keep(h)
                    elif measured == "ingest_b":
                        var x = NDArray[dtype](
                            Shape(rows, forest.num_features())
                        )
                        fill_samples(x, seed, missing, pattern)
                        upload_into(x, batch)
                        forest.predict_margin_into(batch, result)
                        var h = result.to_host()
                        keep(h)
                    elif measured == "ingest_c":
                        fill_pinned(
                            staging_tensor,
                            input.size,
                            forest.num_features(),
                            seed,
                            missing,
                            pattern,
                        )
                        with _Join(ctx):
                            if input.size:
                                ctx.enqueue_copy(batch._buffer, staging)
                        forest.predict_margin_into(batch, result)
                        var h = result.to_host()
                        keep(h)
                    elif measured == "pipeline":
                        var x = NDArray[dtype](
                            Shape(rows, forest.num_features())
                        )
                        fill_samples(x, seed, missing, pattern)
                        var b = upload(x, ctx)
                        var y = forest.predict_margin(b)
                        var h = y.to_host()
                        keep(h)
                seconds = Float64(perf_counter_ns() - t) / 1e9
            if warmed and seconds >= min_seconds:
                break
            if warmed:
                if loops >= 16777216:
                    raise Error("Phase calibration exceeded loop cap")
                loops *= 2
            warmed = True
        print(
            measured,
            seconds,
            loops,
            context_ns,
            read_ns,
            decode_ns,
            prepare_ns,
            input_ns,
            upload_ns,
            first_ns,
            download_ns,
            batch_allocate_ns,
            pinned_allocate_ns,
        )


def main() raises:
    var args = argv()
    if len(args) != 12:
        raise Error(
            "Usage: worker MODEL DTYPE ROWS LOOPS PHASE SEED MISSING PATTERN"
            " ORACLE MIN_SECONDS TOLERANCE"
        )
    var rows = Int(args[3])
    var loops = Int(args[4])
    var missing = Int(args[7])
    if (
        rows < 0
        or loops <= 0
        or missing < 0
        or missing > 100
        or Float64(args[10]) <= 0
    ):
        raise Error("Invalid dimensions or calibration parameters")
    if args[8] != "divergent" and args[8] != "coherent" and args[8] != "legacy":
        raise Error("Unknown sample pattern")
    if args[2] == "float32":
        run[DType.float32](
            String(args[1]),
            rows,
            loops,
            String(args[5]),
            Int(args[6]),
            missing,
            String(args[8]),
            String(args[9]),
            Float64(args[10]),
            Float64(args[11]),
        )
    elif args[2] == "float64":
        run[DType.float64](
            String(args[1]),
            rows,
            loops,
            String(args[5]),
            Int(args[6]),
            missing,
            String(args[8]),
            String(args[9]),
            Float64(args[10]),
            Float64(args[11]),
        )
    else:
        raise Error("Unsupported dtype")
