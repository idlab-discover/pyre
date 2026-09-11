from balsa import load
from std.testing import assert_equal, assert_raises
from numojo import NDArray
from numojo.core.accelerator_ndarray import AcceleratorNDArray
from numojo.core.accelerator.device import Device
from numojo.core.type_aliases import Shape
from pyre import Forest
from pyre.accelerator import (
    predict_cpu,
    predict_margin_cpu,
    predict_proba_cpu,
    predict_class_cpu,
    to_host,
    to_accelerator,
)


def check_device[device: Device]() raises:
    var forest = Forest(load("../balsa/tests/fixtures/float32_op2_missing0.tl"))
    for rows in [0, 1, 9, 33]:
        var host = NDArray[DType.float32](Shape(rows, 1))
        for i in range(rows):
            host.itemset(i, Float32(i % 2))
        var x = to_accelerator[DType.float32, device](host)
        var y = predict_cpu(forest, x)
        var result = to_host(y)
        assert_equal(result.size, rows)
        for i in range(rows):
            assert_equal(result.item(i), Float32(-1 if i % 2 == 0 else 3))
        x.offset = 1
        with assert_raises():
            _ = predict_cpu(forest, x)
    var classifier = Forest(load("../balsa/tests/fixtures/float32_sigmoid.tl"))
    var h = NDArray[DType.float32](Shape(9, 1))
    for i in range(9):
        h.itemset(i, Float32(i % 2))
    var input = to_accelerator[DType.float32, device](h)
    var margins = predict_margin_cpu(classifier, input)
    var p = predict_proba_cpu(classifier, input)
    var labels = predict_class_cpu(classifier, input)
    var ph = to_host(p)
    var lh = to_host(labels)
    assert_equal(margins.size, 9)
    assert_equal(ph.shape[1], 2)
    for i in range(9):
        assert_equal(lh.item(i), Int64(ph.item(2 * i + 1) > 0.5))


def check_gpu_unimplemented() raises:
    # No allocation/context: rejection must happen before touching GPU storage.
    var x = AcceleratorNDArray[DType.float32, Device.CUDA]()
    var forest = Forest(load("../balsa/tests/fixtures/float32_op2_missing0.tl"))
    for operation in range(4):
        var raised = False
        try:
            if operation == 0:
                _ = predict_cpu(forest, x)
            elif operation == 1:
                _ = predict_margin_cpu(forest, x)
            elif operation == 2:
                _ = predict_proba_cpu(forest, x)
            else:
                _ = predict_class_cpu(forest, x)
        except error:
            raised = True
            assert_equal(
                String(error),
                (
                    "NotImplementedError: GPU inference requires GPU kernels;"
                    " CPU fallback is disabled"
                ),
            )
        assert_equal(raised, True)


def check_cuda_transfers() raises:
    var host = NDArray[DType.float32](Shape(9, 1))
    for i in range(9):
        host.itemset(i, Float32(i))
    var gpu = to_accelerator[DType.float32, Device.CUDA](host)
    var restored = to_host(gpu)
    for i in range(9):
        assert_equal(restored.item(i), host.item(i))
    var forest = Forest(load("../balsa/tests/fixtures/float32_op2_missing0.tl"))
    with assert_raises():
        _ = predict_cpu(forest, gpu)


def main() raises:
    check_device[Device.CPU]()
    check_gpu_unimplemented()
    check_cuda_transfers()
    print("CPU inference, GPU rejection, and explicit CUDA transfers passed")
