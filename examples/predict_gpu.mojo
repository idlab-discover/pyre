"""Explicit preparation/upload, resident prediction, and explicit download."""
from balsa import load
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from pyre.gpu import GpuForest, GpuArray, GpuBatch, upload, upload_into
from numojo import NDArray
from numojo.core.type_aliases import Shape


def main() raises:
    var context = DeviceContext()
    var model = load(
        "build/balsa-reference/tests/fixtures/float32_op2_missing0.tl"
    )
    var forest = GpuForest.prepare(model, context)
    var samples = NDArray[DType.float32](Shape(3, 1))
    samples.itemset(0, 0)
    samples.itemset(1, 0.5)
    samples.itemset(2, 1)
    var batch = upload(samples, context)
    var output = GpuArray[DType.float32, 1](context, Shape(3))
    forest.predict_into(batch, output)
    print(output.to_host())  # [-1, 3, 3]

    # Fixed shape: reuse the device batch and prediction output.
    samples.itemset(0, 1)
    upload_into(samples, batch)
    forest.predict_into(batch, output)
    print(output.to_host())  # [3, 3, 3]

    # Changing active rows: retain capacity, create checked logical wrappers.
    var input_storage = context.enqueue_create_buffer[DType.float32](257)
    var output_storage = context.enqueue_create_buffer[DType.float32](257)
    for rows in [1, 33, 257]:
        var next_samples = NDArray[DType.float32](Shape(rows, 1))
        for row in range(rows):
            next_samples.itemset(row, 0)
        var active = GpuBatch[DType.float32].from_buffer(
            input_storage, rows, 1, context=context
        )
        var active_output = GpuArray[DType.float32, 1](
            output_storage, Shape(rows), context=context
        )
        upload_into(next_samples, active)
        forest.predict_into(active, active_output)
        print("Completed rows:", active_output.dim(0))

    # A native producer can fill pinned storage without an NDArray input adapter.
    var staging = context.enqueue_create_host_buffer[DType.float32](3)
    var values = TileTensor(staging, row_major(3))
    comptime assert values.flat_rank == 1
    for row in range(3):
        values[row] = 0
    var native_samples = context.enqueue_create_buffer[DType.float32](3)
    context.enqueue_copy(native_samples, staging)
    context.synchronize()  # Staging can be filled again after completion.
    var native_batch = GpuBatch[DType.float32].from_buffer(
        native_samples^, 3, 1, context=context
    )
    forest.predict_into(native_batch, output)
    print(output.to_host())  # [-1, -1, -1]
