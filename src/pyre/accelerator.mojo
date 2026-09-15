"""Explicit array transfers and CPU-array inference adapters.

These legacy CPU adapters reject GPU arrays before reading or transferring
data. Native GPU inference is provided separately by pyre.gpu. Explicit
transfer helpers remain available.
"""
from numojo import NDArray
from numojo.core.accelerator_ndarray import AcceleratorNDArray
from numojo.core.accelerator.device import Device
from .forest import Forest


def _check_layout[
    dtype: DType, device: Device
](x: AcceleratorNDArray[dtype, device]) raises:
    if (
        not x.flags.C_CONTIGUOUS
        or x.offset != 0
        or not x.flags.OWNDATA
        or x.size != x._buf.size
    ):
        raise Error(
            "CPU accelerator adapter requires owned, zero-offset, contiguous"
            " storage"
        )
    var expected = 1
    for dim in range(x.ndim - 1, -1, -1):
        if x.shape[dim] > 1 and x.strides[dim] != expected:
            raise Error("CPU accelerator adapter requires C-contiguous strides")
        expected *= x.shape[dim]


def to_host[
    dtype: DType, device: Device
](x: AcceleratorNDArray[dtype, device]) raises -> NDArray[dtype]:
    """Synchronous accelerator-to-NDArray copy; rejects views.

    Pinned NuMojo 0.10.0 adapter: value() borrows optional storage. Its
    unsafe_value() accessor copies storage, making escaped pointers unsafe.
    """
    _check_layout(x)
    var result = NDArray[dtype](x.shape)
    if x.size != 0:
        comptime if device.type == "cpu":
            for i in range(x.size):
                result.unsafe_ptr()[
                    unsafe_offset=i
                ] = x._buf.host_storage.value().ptr[unsafe_offset=i]
        else:
            x._buf.device_storage.value().get_buffer().enqueue_copy_to(
                result.unsafe_ptr()
            )
            x._buf.device_storage.value().get_buffer().context().synchronize()
    return result^


def to_accelerator[
    dtype: DType, device: Device
](x: NDArray[dtype]) raises -> AcceleratorNDArray[dtype, device]:
    """Synchronous copy from a contiguous NDArray to owned accelerator storage.
    """
    var expected = 1
    for dim in range(x.ndim - 1, -1, -1):
        if x.shape[dim] > 1 and x.strides[dim] != expected:
            raise Error("to_accelerator requires C-contiguous strides")
        expected *= x.shape[dim]
    var result = AcceleratorNDArray[dtype, device](shape=x.shape)
    if x.size != 0:
        comptime if device.type == "cpu":
            for i in range(x.size):
                result._buf.host_storage.value().ptr[
                    unsafe_offset=i
                ] = x.unsafe_ptr()[unsafe_offset=i]
        else:
            result._buf.device_storage.value().get_buffer().enqueue_copy_from(
                x.unsafe_ptr()
            )
            result._buf.device_storage.value().get_buffer().context().synchronize()
    return result^


def _require_cpu[device: Device]() raises:
    comptime if device.type != "cpu":
        raise Error(
            "NotImplementedError: GPU inference requires GPU kernels; CPU"
            " fallback is disabled"
        )


def predict_cpu[
    dtype: DType, device: Device
](
    forest: Forest[dtype],
    x: AcceleratorNDArray[dtype, device],
    *,
    num_threads: Int = 0,
) raises -> AcceleratorNDArray[dtype, device]:
    """Regression inference for CPU accelerator arrays; GPU input raises."""
    _require_cpu[device]()
    var host = to_host(x)
    var result = forest.predict(host, num_threads=num_threads)
    return to_accelerator[dtype, device](result)


def predict_margin_cpu[
    dtype: DType, device: Device
](
    forest: Forest[dtype],
    x: AcceleratorNDArray[dtype, device],
    *,
    num_threads: Int = 0,
) raises -> AcceleratorNDArray[dtype, device]:
    """Raw margins for CPU accelerator arrays; GPU input raises."""
    _require_cpu[device]()
    var host = to_host(x)
    var result = forest.predict_margin(host, num_threads=num_threads)
    return to_accelerator[dtype, device](result)


def predict_proba_cpu[
    dtype: DType, device: Device
](
    forest: Forest[dtype],
    x: AcceleratorNDArray[dtype, device],
    *,
    num_threads: Int = 0,
) raises -> AcceleratorNDArray[dtype, device]:
    """Binary probabilities for CPU accelerator arrays; GPU input raises."""
    _require_cpu[device]()
    var host = to_host(x)
    var result = forest.predict_proba(host, num_threads=num_threads)
    return to_accelerator[dtype, device](result)


def predict_class_cpu[
    dtype: DType, device: Device
](
    forest: Forest[dtype],
    x: AcceleratorNDArray[dtype, device],
    threshold: Scalar[dtype] = 0.5,
    *,
    num_threads: Int = 0,
) raises -> AcceleratorNDArray[DType.int64, device]:
    """Binary labels for CPU accelerator arrays; GPU input raises."""
    _require_cpu[device]()
    var host = to_host(x)
    var result = forest.predict_class(host, threshold, num_threads=num_threads)
    return to_accelerator[DType.int64, device](result)
