"""Native, synchronous GPU tree inference with explicit device storage."""
from balsa import Model, TaskType
from max.gpu.host import (
    DeviceContext,
    DeviceBuffer,
    DeviceAttribute,
    FuncAttribute,
)
from std.memory import unsafe_memcpy
from std.bit import next_power_of_two
from std.math import ceildiv, inf
from std.sys import size_of
from std.time import perf_counter_ns
from numojo import NDArray
from numojo.core.type_aliases import Shape
from ._validation import validate_inference
from ._gpu_storage import (
    GpuArray,
    upload,
    upload_into,
    _Join,
    _gpu,
    _same_device,
    _check_upload,
)
from ._gpu_streams import _StreamCache, _JoinAll
from ._gpu_kernel import predict_kernel
from ._layout import OP_LT, node_count, node_words, pack, path_length


# A batch is a rank-two GPU array; results use the same checked storage type.
comptime GpuBatch[dtype: DType] = GpuArray[dtype, 2]
# (rows, trees) chains per thread compiled into every kernel variant.
comptime _SHAPES = [(1, 1), (2, 1), (4, 1), (1, 2), (1, 4), (2, 2)]
# Largest rows per block, as a shift.
comptime _MAX_SHIFT = 8
comptime _MAX_THREADS = 1024


@fieldwise_init
struct GpuPlan(Equatable, ImplicitlyCopyable, Writable):
    """How one prediction maps onto the GPU.

    A block owns `1 << chunk_shift` consecutive rows and `threads` threads.
    Each thread walks `rows_per_thread` rows on `trees_per_thread` trees at a
    time as independent chains. `staged` copies the block's rows into shared
    memory, transposed, before traversal. Every valid plan computes the same
    values; tree summation order, and so the last bits, depend on the plan.
    """

    var chunk_shift: Int
    var threads: Int
    var rows_per_thread: Int
    var trees_per_thread: Int
    var staged: Bool


@fieldwise_init
struct _Device(ImplicitlyCopyable):
    """Launch limits discovered from the device at preparation."""

    var sms: Int
    var warp: Int
    var threads_per_sm: Int
    var shared_per_block: Int
    var shared_optin: Int
    var shared_per_sm: Int
    var blocks_per_sm: Int

    @staticmethod
    def query(ctx: DeviceContext) raises -> Self:
        var shared = Int(
            ctx.get_attribute(DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK)
        )
        return Self(
            Int(ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)),
            Int(ctx.get_attribute(DeviceAttribute.WARP_SIZE)),
            Int(
                ctx.get_attribute(
                    DeviceAttribute.MAX_THREADS_PER_MULTIPROCESSOR
                )
            ),
            shared,
            max(
                shared,
                Int(
                    ctx.get_attribute(
                        DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
                    )
                ),
            ),
            Int(
                ctx.get_attribute(
                    DeviceAttribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
                )
            ),
            Int(
                ctx.get_attribute(DeviceAttribute.MAX_BLOCKS_PER_MULTIPROCESSOR)
            ),
        )


@fieldwise_init
struct _Tuned(ImplicitlyCopyable):
    var bucket: Int
    var plan: GpuPlan


def _bucket(rows: Int) -> Int:
    """Tuned plans apply to every row count with the same power-of-two ceiling.
    """
    return next_power_of_two(max(rows, 1))


struct GpuForest[dtype: DType = DType.float32](Movable):
    """Prepared forest. Calls on the same context must not overlap.

    Prediction is immutable. `tune` and `use_plan` change only the launch
    plans of later calls.
    """

    var _context: DeviceContext
    var _nodes: DeviceBuffer[DType.uint32]
    var _roots: DeviceBuffer[DType.int32]
    var _count: Int
    var _words: Int
    var _general: Bool
    var _features: Int
    var _trees: Int
    var _classifier: Bool
    var _average: Int32
    var _bias: Float64
    var _alpha: Float32
    var _device: _Device
    var _depth: Float64
    var _tuned: List[_Tuned]
    var _streams: _StreamCache

    def __init__(out self, model: Model[Self.dtype], ctx: DeviceContext) raises:
        _gpu(ctx)
        _ = validate_inference(model)
        var words = node_words(model)
        var count = node_count(model)
        var features = Int(model.num_feature)
        var nh = ctx.enqueue_create_host_buffer[DType.uint32](count * words)
        var rh = ctx.enqueue_create_host_buffer[DType.int32](len(model.trees))
        var used = pack[keyed=True, strict=False](
            model, nh.unsafe_ptr(), rh.unsafe_ptr(), words
        )
        var general = (used & ~(UInt32(1) << OP_LT)) != 0
        var nd = ctx.enqueue_create_buffer[DType.uint32](count * words)
        var rd = ctx.enqueue_create_buffer[DType.int32](len(model.trees))
        with _Join(ctx):
            ctx.enqueue_copy(nd, nh)
            ctx.enqueue_copy(rd, rh)
        var trees = len(model.trees)
        self._context = ctx.copy()
        self._nodes = nd^
        self._roots = rd^
        self._count = count
        self._words = words
        self._general = general
        self._features = features
        self._trees = trees
        self._classifier = model.task_type == TaskType.BINARY_CLF
        self._average = Int32(model.average_tree_output != 0)
        self._bias = model.base_scores[0]
        self._alpha = model.sigmoid_alpha
        self._device = _Device.query(ctx)
        self._depth = path_length(model) / Float64(max(trees, 1))
        self._tuned = List[_Tuned]()
        self._streams = _StreamCache()

    def __deinit__(deinit self):
        _ = self._streams^
        _ = self._nodes^
        _ = self._roots^
        _ = self._context^

    @staticmethod
    def prepare(
        model: Model[Self.dtype], context: DeviceContext
    ) raises -> Self:
        return Self(model, context)

    def num_features(self) -> Int:
        return self._features

    def num_trees(self) -> Int:
        return self._trees

    def is_classifier(self) -> Bool:
        return self._classifier

    # ----------------------------------------------------------------------
    # Launch plans

    def plan(self, rows: Int) -> GpuPlan:
        """The plan a call with `rows` rows uses: a tuned or pinned plan for
        its power-of-two bucket, the largest tuned plan for bigger batches,
        otherwise the device-derived default."""
        var bucket = _bucket(rows)
        var top = -1
        for i in range(len(self._tuned)):
            ref entry = self._tuned[i]
            if entry.bucket == bucket:
                return entry.plan
            if top < 0 or entry.bucket > self._tuned[top].bucket:
                top = i
        if top >= 0 and bucket > self._tuned[top].bucket:
            return self._tuned[top].plan
        return self._default_plan(rows)

    def use_plan(mut self, rows: Int, plan: GpuPlan) raises:
        """Pin `plan` for row counts in the power-of-two bucket of `rows`.

        Raises if the plan is malformed or exceeds this device's limits.
        Pinned plans make repeated runs choose identical launches.
        """
        self._check_plan(plan)
        if not self._fits(plan):
            raise Error("Pyre GPU plan exceeds this device's resources")
        self._store(_bucket(rows), plan)

    def tune(mut self, x: GpuBatch[Self.dtype]) raises -> GpuPlan:
        """Measure candidate plans on `x` and keep the fastest for its bucket.

        Candidates are generated from the device's limits and the forest's
        shape, then searched one dimension at a time from the default plan.
        Batches beyond a few full waves of the device are measured on a
        prefix, since per-row cost no longer changes there. Inputs are only
        read. Returns the stored plan.
        """
        self._check_input(x)
        _same_device(self._context, x._buffer.context())
        var rows = x._shape[0]
        if rows == 0:
            raise Error("Pyre GPU tuning requires at least one row")
        var sample = min(rows, self._tune_rows())
        var scratch = self._context.enqueue_create_buffer[Self.dtype](sample)
        var input = (
            x._buffer.unsafe_ptr()
            .unsafe_offset(x._offset)
            .unsafe_origin_cast[MutAnyOrigin]()
            .as_imm()
        )
        var output = scratch.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var best = self._default_plan(rows)
        var best_time = self._time(
            best, input, sample, output, inf[DType.float64]()
        )
        for _ in range(2):
            var before = best
            # Rows per block x threads x staging, then chains per thread.
            var seed = best
            for shift in range(_MAX_SHIFT + 1):
                if (1 << shift) > 2 * next_power_of_two(sample):
                    break
                var threads = self._device.warp
                while threads <= _MAX_THREADS:
                    for staged in [False, True]:
                        var p = seed
                        p.chunk_shift = shift
                        p.threads = threads
                        p.staged = staged
                        self._consider(
                            p, input, sample, output, best, best_time
                        )
                    threads *= 2
            comptime for i in range(len(_SHAPES)):
                comptime r = _SHAPES[i][0]
                comptime q = _SHAPES[i][1]
                var p = best
                p.rows_per_thread = r
                p.trees_per_thread = q
                p = self._repair(p)
                self._consider(p, input, sample, output, best, best_time)
            if best == before:
                break
        self._context.synchronize()
        _ = scratch^
        self._store(_bucket(rows), best)
        return best

    def _store(mut self, bucket: Int, plan: GpuPlan):
        for i in range(len(self._tuned)):
            if self._tuned[i].bucket == bucket:
                self._tuned[i].plan = plan
                return
        self._tuned.append(_Tuned(bucket, plan))

    def _resident(self, threads: Int) -> Int:
        """Blocks of `threads` threads an SM holds by thread and block limits.
        """
        var t = max(self._device.warp, min(_MAX_THREADS, threads))
        return max(
            1, min(self._device.blocks_per_sm, self._device.threads_per_sm // t)
        )

    def _tune_rows(self) -> Int:
        """Rows that fill every SM with several waves of the largest blocks;
        per-row cost no longer changes beyond this."""
        return max(4096, self._device.sms * 4096)

    def _consider(
        self,
        plan: GpuPlan,
        input: ImmPointer[Scalar[Self.dtype], ImmutAnyOrigin],
        rows: Int,
        output: MutPointer[Scalar[Self.dtype], MutAnyOrigin],
        mut best: GpuPlan,
        mut best_time: Float64,
    ):
        if plan == best or not self._valid(plan) or not self._fits(plan):
            return
        var t = self._time(plan, input, rows, output, best_time)
        # Switch only on a clear gain, so timing noise cannot walk the plan.
        if t < 0.98 * best_time:
            best = plan
            best_time = t

    def _time(
        self,
        plan: GpuPlan,
        input: ImmPointer[Scalar[Self.dtype], ImmutAnyOrigin],
        rows: Int,
        output: MutPointer[Scalar[Self.dtype], MutAnyOrigin],
        bound: Float64,
    ) -> Float64:
        """Best completed-call seconds; abandons plans far behind `bound`.
        Plans the device rejects at launch measure as infinitely slow."""
        try:
            # The first launch also loads the kernel variant.
            self._run[Self.dtype, 0](
                input, rows, output, Scalar[Self.dtype](0.5), plan
            )
            self._context.synchronize()
            var best = inf[DType.float64]()
            var spent = 0
            var repeats = 0
            while repeats < 3 or (spent < 2_000_000 and repeats < 50):
                var start = perf_counter_ns()
                self._run[Self.dtype, 0](
                    input,
                    rows,
                    output,
                    Scalar[Self.dtype](0.5),
                    plan,
                )
                self._context.synchronize()
                var elapsed = perf_counter_ns() - start
                spent += Int(elapsed)
                best = min(best, Float64(elapsed) * 1e-9)
                repeats += 1
                if best > 1.5 * bound:
                    break
            return best
        except:
            return inf[DType.float64]()

    def _valid(self, plan: GpuPlan) -> Bool:
        var chunk = 1 << plan.chunk_shift
        var known = False
        comptime for i in range(len(_SHAPES)):
            comptime r = _SHAPES[i][0]
            comptime q = _SHAPES[i][1]
            if plan.rows_per_thread == r and plan.trees_per_thread == q:
                known = True
        return (
            known
            and plan.chunk_shift >= 0
            and plan.chunk_shift <= _MAX_SHIFT
            and plan.threads >= self._device.warp
            and plan.threads <= _MAX_THREADS
            and plan.threads % self._device.warp == 0
            and next_power_of_two(plan.threads) == plan.threads
            and chunk >= plan.rows_per_thread
            and chunk // plan.rows_per_thread <= plan.threads
        )

    def _check_plan(self, plan: GpuPlan) raises:
        if not self._valid(plan):
            raise Error(
                String(
                    t"Pyre GPU plan is invalid: {plan}. Rows per block must"
                    t" be 1 << 0..{_MAX_SHIFT}; threads a power of two from"
                    t" the warp size to {_MAX_THREADS} covering rows per"
                    t" block / rows per thread; (rows, trees) per thread one of"
                    t" (1, 1), (2, 1), (4, 1), (1, 2), (1, 4), (2, 2)"
                )
            )

    def _shared(self, plan: GpuPlan) -> Int:
        comptime item = size_of[Scalar[Self.dtype]]()
        var bytes = plan.threads * plan.rows_per_thread * item
        if plan.staged:
            # Float64 rows stage as Int64 keys: the same width.
            bytes += self._features * ((1 << plan.chunk_shift) + 1) * item
        return bytes

    def _fits(self, plan: GpuPlan) -> Bool:
        return self._shared(plan) <= self._device.shared_optin

    def _repair(self, plan: GpuPlan) -> GpuPlan:
        """Raise threads until they cover the block's row lanes."""
        var p = plan
        var lanes = (1 << p.chunk_shift) // max(p.rows_per_thread, 1)
        while p.threads < lanes and p.threads < _MAX_THREADS:
            p.threads *= 2
        return p

    def _default_plan(self, rows: Int) -> GpuPlan:
        """Device-derived plan without measurement; `tune` refines it.

        Rows are staged when the forest reads them often: float64 rows
        (staging converts each value to a key once) or rows visiting at least
        16 nodes per feature. Staged blocks own 32 to 64 rows (keeping two
        blocks per SM) and add threads while the trees in flight per block
        stay within one SM's shared memory, a proxy for its L1; grids too
        small to occupy every SM widen blocks to all their trees. Unstaged
        blocks cover every tree in one pass: four trees per thread, about four
        warps per block, and as many rows as keep the rows resident on an SM
        within a quarter of its shared memory; an underfilled GPU walks one
        tree per thread instead. Blocks are never so small that
        the per-SM block limit, rather than threads, caps occupancy.
        """
        var d = self._device
        # Smaller blocks would hit the per-SM block limit before its threads.
        var floor = max(
            d.warp,
            next_power_of_two(ceildiv(d.threads_per_sm, d.blocks_per_sm)),
        )
        var trees = max(self._trees, 1)
        var tree_bytes = self._count * self._words * 4 // trees
        var visits = Float64(trees) * self._depth
        var plan = GpuPlan(0, d.warp, 1, 1, False)
        plan.trees_per_thread = 4 if trees >= 16 else (2 if trees >= 2 else 1)
        if Self.dtype == DType.float64 or visits >= Float64(
            16 * self._features
        ):
            var staged = plan
            staged.staged = True
            staged.chunk_shift = 6
            while (
                staged.chunk_shift > 5
                and ceildiv(rows, 1 << staged.chunk_shift) < 2 * d.sms
            ):
                staged.chunk_shift -= 1
            var few = ceildiv(rows, 1 << staged.chunk_shift) < 2 * d.sms
            if trees >= 256 and staged.chunk_shift >= 1:
                staged.rows_per_thread = 2
                staged.trees_per_thread = 2
            var lanes = (1 << staged.chunk_shift) // staged.rows_per_thread
            staged.threads = max(floor, lanes)
            # Few blocks cannot occupy every SM: widen each to all its trees.
            while (
                staged.threads < (_MAX_THREADS if few else 512)
                and (
                    few
                    or (2 * staged.threads // lanes)
                    * staged.trees_per_thread
                    * tree_bytes
                    <= d.shared_per_sm
                )
                and (staged.threads // lanes) * staged.trees_per_thread < trees
            ):
                staged.threads *= 2
            staged = self._repair(staged)
            # Too few blocks to reach half the SMs: per-row latency matters
            # more than shared row reads.
            if 2 * ceildiv(
                rows, 1 << staged.chunk_shift
            ) >= d.sms and self._shared(staged) <= min(
                d.shared_per_block, d.shared_per_sm // 4
            ):
                return staged
        var groves = next_power_of_two(ceildiv(trees, plan.trees_per_thread))
        var chunk = max(1, 4 * d.warp // groves)
        # Rows resident per SM stay within a quarter of its shared memory.
        var row_bytes = self._features * size_of[Scalar[Self.dtype]]()
        while chunk > 1 and (
            ceildiv(rows, chunk) < 2 * d.sms
            or chunk * row_bytes * self._resident(chunk * groves)
            > d.shared_per_sm // 4
        ):
            chunk //= 2
        while (1 << plan.chunk_shift) < chunk:
            plan.chunk_shift += 1
        if ceildiv(rows, chunk) < 2 * d.sms:
            # An underfilled GPU is latency-bound: one chain per thread
            # gives the shortest walk per block.
            plan.trees_per_thread = 1
            groves = next_power_of_two(trees)
        plan.threads = max(floor, min(_MAX_THREADS, chunk * groves))
        # Keep the whole grid within half the device's thread slots.
        var blocks = ceildiv(rows, chunk)
        while (
            plan.threads > floor
            and blocks * plan.threads > d.sms * d.threads_per_sm // 2
        ):
            plan.threads //= 2
        return self._repair(plan)

    # ----------------------------------------------------------------------
    # Launch

    def _check_input(self, x: GpuBatch[Self.dtype]) raises:
        # Kernel row counts are signed 32-bit. Check before output allocation
        # as well as before launch, including zero-feature batches.
        if x._shape[0] > Int(Int32.MAX):
            raise Error("Pyre GPU inference supports at most 2147483647 rows")
        if x._shape[1] != self._features:
            raise Error("Pyre input feature count does not match the forest")

    def _check[
        output_dtype: DType, rank: Int
    ](
        self, x: GpuBatch[Self.dtype], output: GpuArray[output_dtype, rank]
    ) raises:
        self._check_input(x)
        if output._shape[0] != x._shape[0]:
            raise Error("Pyre GPU output row count mismatch")
        comptime if rank == 2:
            if output._shape[1] != 2:
                raise Error("Pyre GPU probabilities require [rows, 2] output")
        _same_device(self._context, x._buffer.context())
        _same_device(self._context, output._buffer.context())
        if x._size and output._size:
            var xb = (
                Int(x._buffer.unsafe_ptr())
                + x._offset * size_of[Scalar[Self.dtype]]()
            )
            var yb = (
                Int(output._buffer.unsafe_ptr())
                + output._offset * size_of[Scalar[output_dtype]]()
            )
            var xe = xb + x._size * size_of[Scalar[Self.dtype]]()
            var ye = yb + output._size * size_of[Scalar[output_dtype]]()
            if xb < ye and yb < xe:
                raise Error("Pyre GPU input and output must not overlap")

    def _enqueue[
        output_dtype: DType, rank: Int, mode: Int
    ](
        self,
        x: GpuBatch[Self.dtype],
        mut output: GpuArray[output_dtype, rank],
        threshold: Scalar[Self.dtype],
        plan: GpuPlan,
    ) raises:
        self._run[output_dtype, mode](
            x._buffer.unsafe_ptr()
            .unsafe_offset(x._offset)
            .unsafe_origin_cast[MutAnyOrigin]()
            .as_imm(),
            x._shape[0],
            output._buffer.unsafe_ptr()
            .unsafe_offset(output._offset)
            .unsafe_origin_cast[MutAnyOrigin](),
            threshold,
            plan,
        )

    def _run[
        output_dtype: DType, mode: Int
    ](
        self,
        x: ImmPointer[Scalar[Self.dtype], ImmutAnyOrigin],
        rows: Int,
        y: MutPointer[Scalar[output_dtype], MutAnyOrigin],
        threshold: Scalar[Self.dtype],
        plan: GpuPlan,
    ) raises:
        """Enqueue one validated plan; binds per-forest facts to a kernel."""
        if rows == 0:
            return
        self._check_plan(plan)
        if not self._fits(plan):
            raise Error("Pyre GPU plan exceeds this device's shared memory")
        if self._words == 2:
            self._dispatch[mode=mode, words=2](x, rows, y, threshold, plan)
        else:
            self._dispatch[mode=mode, words=4](x, rows, y, threshold, plan)

    def _dispatch[
        output_dtype: DType, //, mode: Int, words: Int
    ](
        self,
        x: ImmPointer[Scalar[Self.dtype], ImmutAnyOrigin],
        rows: Int,
        y: MutPointer[Scalar[output_dtype], MutAnyOrigin],
        threshold: Scalar[Self.dtype],
        plan: GpuPlan,
    ) raises:
        comptime if Self.dtype == DType.float64 and words == 2:
            raise Error("Pyre GPU float64 nodes require the wide layout")
        else:
            if self._general:
                if plan.staged:
                    self._chains[mode, words, True, True](
                        x, rows, y, threshold, plan
                    )
                else:
                    self._chains[mode, words, True, False](
                        x, rows, y, threshold, plan
                    )
            elif plan.staged:
                self._chains[mode, words, False, True](
                    x, rows, y, threshold, plan
                )
            else:
                self._chains[mode, words, False, False](
                    x, rows, y, threshold, plan
                )

    def _chains[
        output_dtype: DType,
        //,
        mode: Int,
        words: Int,
        general: Bool,
        staged: Bool,
    ](
        self,
        x: ImmPointer[Scalar[Self.dtype], ImmutAnyOrigin],
        rows: Int,
        y: MutPointer[Scalar[output_dtype], MutAnyOrigin],
        threshold: Scalar[Self.dtype],
        plan: GpuPlan,
    ) raises:
        comptime for i in range(len(_SHAPES)):
            comptime r = _SHAPES[i][0]
            comptime q = _SHAPES[i][1]
            if plan.rows_per_thread == r and plan.trees_per_thread == q:
                self._launch[mode, words, general, staged, r, q](
                    x, rows, y, threshold, plan
                )
                return

    def _launch[
        output_dtype: DType,
        //,
        mode: Int,
        words: Int,
        general: Bool,
        staged: Bool,
        rows_per_thread: Int,
        trees_per_thread: Int,
    ](
        self,
        x: ImmPointer[Scalar[Self.dtype], ImmutAnyOrigin],
        rows: Int,
        y: MutPointer[Scalar[output_dtype], MutAnyOrigin],
        threshold: Scalar[Self.dtype],
        plan: GpuPlan,
    ) raises:
        comptime kernel = predict_kernel[
            Self.dtype,
            output_dtype,
            mode,
            words,
            general,
            staged,
            rows_per_thread,
            trees_per_thread,
        ]
        var shared = self._shared(plan)
        var grid = ceildiv(rows, 1 << plan.chunk_shift)
        if shared <= self._device.shared_per_block:
            self._context.enqueue_function[kernel](
                self._nodes.unsafe_ptr().as_imm(),
                self._roots.unsafe_ptr().as_imm(),
                x,
                y,
                Int32(rows),
                Int32(self._trees),
                Int32(self._features),
                Int32(plan.chunk_shift),
                self._average,
                self._bias,
                self._alpha,
                threshold,
                grid_dim=grid,
                block_dim=plan.threads,
                shared_mem_bytes=shared,
            )
        else:
            # Beyond the default per-block limit: opt in, up to the device's.
            self._context.enqueue_function[kernel](
                self._nodes.unsafe_ptr().as_imm(),
                self._roots.unsafe_ptr().as_imm(),
                x,
                y,
                Int32(rows),
                Int32(self._trees),
                Int32(self._features),
                Int32(plan.chunk_shift),
                self._average,
                self._bias,
                self._alpha,
                threshold,
                grid_dim=grid,
                block_dim=plan.threads,
                shared_mem_bytes=shared,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    UInt32(shared)
                ),
            )

    def _into[
        output_dtype: DType, rank: Int, mode: Int
    ](
        self,
        x: GpuBatch[Self.dtype],
        mut output: GpuArray[output_dtype, rank],
        threshold: Scalar[Self.dtype] = 0.5,
    ) raises:
        self._check(x, output)
        var plan = self.plan(x._shape[0])
        with _Join(self._context):
            self._enqueue[output_dtype, rank, mode](x, output, threshold, plan)

    # ----------------------------------------------------------------------
    # Prediction

    def predict_margin_into(
        self, x: GpuBatch[Self.dtype], mut output: GpuArray[Self.dtype, 1]
    ) raises:
        self._into[Self.dtype, 1, 0](x, output)

    def predict_into(
        self, x: GpuBatch[Self.dtype], mut output: GpuArray[Self.dtype, 1]
    ) raises:
        if self._classifier:
            raise Error(
                "Use predict_class or predict_proba for binary classification"
            )
        self.predict_margin_into(x, output)

    def predict_proba_into(
        self, x: GpuBatch[Self.dtype], mut output: GpuArray[Self.dtype, 2]
    ) raises:
        if not self._classifier:
            raise Error("predict_proba requires a binary classifier")
        self._into[Self.dtype, 2, 1](x, output)

    def predict_class_into(
        self,
        x: GpuBatch[Self.dtype],
        mut output: GpuArray[DType.int64, 1],
        threshold: Scalar[Self.dtype] = 0.5,
    ) raises:
        if not (threshold >= 0 and threshold <= 1):
            raise Error("Classification threshold must lie in [0, 1]")
        if not self._classifier:
            raise Error("predict_class requires a binary classifier")
        self._into[DType.int64, 1, 2](x, output, threshold)

    def predict_margin(
        self, x: GpuBatch[Self.dtype]
    ) raises -> GpuArray[Self.dtype, 1]:
        self._check_input(x)
        var result = GpuArray[Self.dtype, 1](
            self._context, Shape(x._shape[0]), _ordered=True
        )
        self.predict_margin_into(x, result)
        return result^

    def predict(
        self, x: GpuBatch[Self.dtype]
    ) raises -> GpuArray[Self.dtype, 1]:
        self._check_input(x)
        var result = GpuArray[Self.dtype, 1](
            self._context, Shape(x._shape[0]), _ordered=True
        )
        self.predict_into(x, result)
        return result^

    def predict_proba(
        self, x: GpuBatch[Self.dtype]
    ) raises -> GpuArray[Self.dtype, 2]:
        self._check_input(x)
        var result = GpuArray[Self.dtype, 2](
            self._context, Shape(x._shape[0], 2), _ordered=True
        )
        self.predict_proba_into(x, result)
        return result^

    def predict_class(
        self, x: GpuBatch[Self.dtype], threshold: Scalar[Self.dtype] = 0.5
    ) raises -> GpuArray[DType.int64, 1]:
        self._check_input(x)
        var result = GpuArray[DType.int64, 1](
            self._context, Shape(x._shape[0]), _ordered=True
        )
        self.predict_class_into(x, result, threshold)
        return result^

    # ----------------------------------------------------------------------
    # Pipelined host prediction

    def _slab_rows(self, rows: Int) -> Int:
        """Rows per pipelined slab: up to eight slabs, each still several
        waves of blocks so every launch fills the device."""
        var least = max(1024, self._device.sms * 256)
        return min(rows, max(least, (ceildiv(rows, 8) + 255) // 256 * 256))

    def _stream[
        output_dtype: DType, mode: Int, cols: Int
    ](
        self, x: NDArray[Self.dtype], threshold: Scalar[Self.dtype]
    ) raises -> NDArray[output_dtype]:
        """Host rows in, host results out, in overlapped slabs.

        Each slab's upload runs on a load stream, its kernel on the forest's
        context and its download on a store stream, ordered by events, so
        the next upload proceeds while earlier slabs compute and download.
        Two slots of device and pinned host storage alternate; a slot is
        refilled only after its previous kernel and download completed.
        Returns after every slab finished.
        """
        _check_upload(x)
        var rows = x.shape[0]
        var width = x.shape[1]
        if width != self._features:
            raise Error("Pyre input feature count does not match the forest")
        if rows > Int(Int32.MAX):
            raise Error("Pyre GPU inference supports at most 2147483647 rows")
        var result: NDArray[output_dtype]
        comptime if cols == 2:
            result = NDArray[output_dtype](Shape(rows, 2))
        else:
            result = NDArray[output_dtype](Shape(rows))
        if rows == 0:
            return result^
        var slab = self._slab_rows(rows)
        var slabs = ceildiv(rows, slab)
        var slots = min(2, slabs)
        var plan = self.plan(slab)
        var ctx = self._context.copy()
        var din = ctx.enqueue_create_buffer[Self.dtype](
            max(1, slots * slab * width)
        )
        var dout = ctx.enqueue_create_buffer[output_dtype](slots * slab * cols)
        var hout = ctx.enqueue_create_host_buffer[output_dtype](
            slots * slab * cols
        )
        # Copy streams use the storage only after its allocation completed.
        ctx.synchronize()
        var leased = self._streams.acquire(ctx)
        var streams = leased[0].copy()
        var loaded = [ctx.create_event(), ctx.create_event()]
        var computed = [ctx.create_event(), ctx.create_event()]
        var drained = [ctx.create_event(), ctx.create_event()]
        var source = x.unsafe_ptr()
        var target = result.unsafe_ptr()
        var din_ptr = din.unsafe_ptr()
        var dout_ptr = dout.unsafe_ptr()
        var hout_ptr = hout.unsafe_ptr()

        def collect(
            i: Int,
        ) raises {
            mut drained, imm hout_ptr, imm target, imm slab, imm rows, imm slots
        }:
            # Slab i's download completed into its slot: copy it out.
            var k = i % slots
            drained[k].synchronize()
            var lo = i * slab
            unsafe_memcpy(
                dest=target.unsafe_offset(lo * cols),
                src=hout_ptr.unsafe_offset(k * slab * cols),
                count=min(slab, rows - lo) * cols,
            )

        try:
            with _JoinAll(ctx, streams):
                for i in range(slabs):
                    var k = i % slots
                    var lo = i * slab
                    var n = min(slab, rows - lo)
                    var xin = din_ptr.unsafe_offset(k * slab * width)
                    var yout = dout_ptr.unsafe_offset(k * slab * cols)
                    if i >= slots:
                        # The slot's previous kernel must finish reading its
                        # input, and its result must leave pinned storage.
                        streams.load.stream().enqueue_wait_for(computed[k])
                        collect(i - slots)
                    if n * width > 0:
                        DeviceBuffer[Self.dtype](
                            streams.load, xin, n * width, owning=False
                        ).enqueue_copy_from(source.unsafe_offset(lo * width))
                    streams.load.stream().record_event(loaded[k])
                    ctx.stream().enqueue_wait_for(loaded[k])
                    self._run[output_dtype, mode](
                        xin.unsafe_origin_cast[MutAnyOrigin]().as_imm(),
                        n,
                        yout.unsafe_origin_cast[MutAnyOrigin](),
                        threshold,
                        plan,
                    )
                    ctx.stream().record_event(computed[k])
                    streams.store.stream().enqueue_wait_for(computed[k])
                    DeviceBuffer[output_dtype](
                        streams.store, yout, n * cols, owning=False
                    ).enqueue_copy_to(hout_ptr.unsafe_offset(k * slab * cols))
                    streams.store.stream().record_event(drained[k])
                for i in range(max(0, slabs - slots), slabs):
                    collect(i)
        finally:
            self._streams.release(leased[1])
        _ = din^
        _ = dout^
        _ = hout^
        return result^

    def predict_margin(
        self, x: NDArray[Self.dtype]
    ) raises -> NDArray[Self.dtype]:
        """Host-array counterpart of `predict_margin`: uploads, predicts and
        downloads in overlapped slabs, returning a host array."""
        return self._stream[Self.dtype, 0, 1](x, 0.5)

    def predict(self, x: NDArray[Self.dtype]) raises -> NDArray[Self.dtype]:
        if self._classifier:
            raise Error(
                "Use predict_class or predict_proba for binary classification"
            )
        return self._stream[Self.dtype, 0, 1](x, 0.5)

    def predict_proba(
        self, x: NDArray[Self.dtype]
    ) raises -> NDArray[Self.dtype]:
        if not self._classifier:
            raise Error("predict_proba requires a binary classifier")
        return self._stream[Self.dtype, 1, 2](x, 0.5)

    def predict_class(
        self, x: NDArray[Self.dtype], threshold: Scalar[Self.dtype] = 0.5
    ) raises -> NDArray[DType.int64]:
        if not (threshold >= 0 and threshold <= 1):
            raise Error("Classification threshold must lie in [0, 1]")
        if not self._classifier:
            raise Error("predict_class requires a binary classifier")
        return self._stream[DType.int64, 2, 1](x, threshold)
