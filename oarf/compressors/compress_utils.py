"""Compression utility"""
import torch
import functools
import logging
import numpy as np
from typing import List

log = logging.getLogger(__name__)

type_map = {
    np.int8: b's',
    np.int16: b'S',
    np.int32: b'i',
    np.int64: b'I',
    np.uint8: b'u',
    np.uint16: b'U',
    np.uint32: b'j',
    np.uint64: b'J',
    np.float32: b'f',
    np.float64: b'F',
}

type_reverse_map = {v: k for k, v in type_map.items()}


class Offset:
    """A convenient object to wrap `int` to pass it by reference"""

    def __init__(self, init_value: int = 0):
        self.__p = init_value

    def __add__(self, size: int):
        return self.__p + size

    def __iadd__(self, size: int):
        self.__p += size
        return self

    def __int__(self):
        return self.__p

    def __index__(self):
        return self.__p


def pack_tensor(tensor: torch.Tensor) -> bytearray:
    """pack a tensor to bytearray"""
    assert not tensor.is_cuda
    nparray = tensor.numpy()
    data = bytearray()

    # pack metadata
    data.extend(pack_tensor_shape(tensor))
    data.extend(type_map[nparray.dtype.type])

    # pack size (in bytes) and data
    tensor_data = nparray.tobytes()
    data.extend(pack_raw_data(tensor_data))

    return data


def unpack_tensor(data: bytearray, offset: Offset) -> torch.Tensor:
    """return unpacked tensor, also increase offset"""
    # unpack dimension
    shape = unpack_tensor_shape(data, offset)

    # unpack data type
    dtype = type_reverse_map[bytes(unpack_fixed_width_data(data, offset, 1))]

    # unpack size data
    nparray = np.frombuffer(unpack_raw_data(data, offset),
                            dtype).reshape(shape)

    return torch.from_numpy(nparray)


def pack_tensor_shape(tensor: torch.Tensor) -> bytes:
    data = bytearray()
    data.extend(pack_int(tensor.ndim))
    for dim_size in tensor.shape:
        data.extend(pack_int(dim_size))
    return data


def unpack_tensor_shape(data: bytearray, offset: Offset) -> List[int]:
    shape = []
    ndim = unpack_int(data, offset)
    for _ in range(ndim):
        dim_size = unpack_int(data, offset)
        shape.append(dim_size)
    return shape


def pack_raw_data(raw_data: bytearray) -> bytearray:
    data = bytearray()
    data.extend(pack_int(len(raw_data)))
    data.extend(raw_data)
    return data


def unpack_raw_data(data: bytearray, offset: Offset) -> bytearray:
    raw_data = bytearray()
    size = unpack_int(data, offset)
    raw_data.extend(data[offset:offset + size])
    offset += size
    return raw_data


def unpack_fixed_width_data(data: bytearray, offset: Offset,
                            width: int) -> bytearray:
    """
    For fixed-width data packing, simply extend the bytearray
    This function only provides a convenient way to increase the offset
    """
    result = data[offset:offset + width].copy()
    offset += width
    return result


def pack_int(val: int) -> bytearray:
    buf = bytearray()
    if val < 0:
        # we use int to represent negative num, but still encode in uint
        if val >= -255:
            buf.extend(type_map[np.int8])
            buf.extend(np.uint8([-val]).tobytes())
        elif val >= -65535:
            buf.extend(type_map[np.int16])
            buf.extend(np.uint16([-val]).tobytes())
        elif val >= -4294967295:
            buf.extend(type_map[np.int32])
            buf.extend(np.uint32([-val]).tobytes())
        elif val > -18446744073709551615:
            buf.extend(type_map[np.int64])
            buf.extend(np.uint64([-val]).tobytes())
        else:
            raise Exception("value too large to encode")
    else:
        if val <= 255:
            buf.extend(type_map[np.uint8])
            buf.extend(np.uint8([val]).tobytes())
        elif val <= 65535:
            buf.extend(type_map[np.uint16])
            buf.extend(np.uint16([val]).tobytes())
        elif val <= 4294967295:
            buf.extend(type_map[np.uint32])
            buf.extend(np.uint32([val]).tobytes())
        elif val <= 18446744073709551615:
            buf.extend(type_map[np.uint64])
            buf.extend(np.uint64([val]).tobytes())
        else:
            raise Exception("value too large to encode")
    return buf


def unpack_int(buf: bytearray, offset: Offset) -> int:
    """return: size of data (in bytes), value"""
    dtype = type_reverse_map[bytes(buf[offset:offset + 1])]
    offset += 1
    if dtype == np.int8:
        val = -int(np.frombuffer(buf[offset:offset + 1], np.uint8))
        offset += 1
    elif dtype == np.int16:
        val = -int(np.frombuffer(buf[offset:offset + 2], np.uint16))
        offset += 2
    elif dtype == np.int32:
        val = -int(np.frombuffer(buf[offset:offset + 4], np.uint32))
        offset += 4
    elif dtype == np.int64:
        val = -int(np.frombuffer(buf[offset:offset + 8], np.uint64))
        offset += 8
    elif dtype == np.uint8:
        val = int(np.frombuffer(buf[offset:offset + 1], np.uint8))
        offset += 1
    elif dtype == np.uint16:
        val = int(np.frombuffer(buf[offset:offset + 2], np.uint16))
        offset += 2
    elif dtype == np.uint32:
        val = int(np.frombuffer(buf[offset:offset + 4], np.uint32))
        offset += 4
    elif dtype == np.uint64:
        val = int(np.frombuffer(buf[offset:offset + 8], np.uint64))
        offset += 8
    else:
        raise ValueError("Unknown value type: {}".format(dtype))
    return val


def pack_float32(val: float) -> bytearray:
    buf = bytearray()
    buf.extend(np.float32([val]).tobytes())
    return buf


def unpack_float32(buf: bytearray, offset: Offset) -> float:
    val = float(np.frombuffer(buf[offset:offset + 4], np.float32))
    offset += 4
    return val


def check_single_or_plural(var, type_requirement, val_requirement):
    assert (type_requirement(var) and val_requirement(var)) or (isinstance(
        var, list) and functools.reduce(lambda a, b: a and b,
                                        [val_requirement(v) for v in var]))


# This function is from PowerSGD [1]
def orthogonalize(matrix, eps=torch.tensor(1e-8)):
    n, m = matrix.shape
    for i in range(m):
        # Normalize the i'th column
        col = matrix[:, i: i + 1]
        col /= torch.sqrt(torch.sum(col ** 2)) + eps
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1:]
            # rest -= torch.matmul(col.t(), rest) * col
            rest -= torch.sum(col * rest, dim=0) * col

# [1] T. Vogels, S. P. Karimireddy, and M. Jaggi, “PowerSGD: Practical Low-Rank
#     Gradient Compression for Distributed Optimization,” arXiv:1905.13727
#     [cs, math, stat], Feb. 2020, Accessed: Jul. 07, 2020. [Online].
