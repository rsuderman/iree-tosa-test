import hjson
import json
import numpy

type_to_nptype = {
    "i1": numpy.bool,
    "i8": numpy.int8,
    "i16": numpy.int16,
    "i32": numpy.int32,
    "f32": numpy.single,

    # These should be the values in the hjson numpy files.
    "bool": numpy.bool,
    "int8": numpy.int8,
    "int16": numpy.int16,
    "int32": numpy.int32,
    "float": numpy.single,

    # Include the versions from the hjson test file.
    "BOOL": numpy.bool,
    "INT8": numpy.int8,
    "INT16": numpy.int16,
    "INT32": numpy.int32,
    "FLOAT": numpy.single,

    # The correctness here may be questionable
    "UINT8": numpy.uint8,
    "UINT16": numpy.uint16,
    "UINT32": numpy.uint32,
}


def get_numpy_type(dtype):
    return type_to_nptype[dtype]


class Tensor:

    def __init__(self):
        self._npy = None
        pass

    def load_bytes(self, data, dtype, shape):
        dtype = get_numpy_type(dtype)
        data = numpy.asarray(data).astype(numpy.uint8)
        data = data.tobytes()
        self._npy = numpy.frombuffer(data, dtype=dtype)
        self._npy = numpy.reshape(self._npy, shape)

    def load_json_str(self, data, dtype, shape=None):
        dtype = get_numpy_type(dtype)
        npy = numpy.asarray(data).astype(dtype)
        if shape is not None:
            npy = npy.reshape(shape)
        self._npy = npy

    def load_npy(self, array):
        self._npy = array

    def load_json(self, path):
        try:
            # This uses json now????
            path = path + ".json"
            with open(path, "r") as f:
                dictionary = hjson.load(f)
                dtype = dictionary["type"]
                data = dictionary["data"]
        except Exception as e:
            path = str(path)
            raise Exception(f"Failed to load tensor {path}")

        dtype = get_numpy_type(dtype)
        self._npy = numpy.asarray(data).astype(dtype)

    def serialize_json(self):
        return json.dumps(self._npy.tolist()).lower()

    def shape(self):
        return self._npy.shape

    def npy(self):
        return self._npy


def parse_dtypes(path, tensor_type):
    dtypes = []
    try:
        with open(path, "r") as f:
            blocks = hjson.load(f)["blocks"]
            for block in blocks:
                if block["name"] != "main":
                    continue
                tensors = block["tensors"]
                names = block[tensor_type]
    except Exception as e:
        raise Exception(f"Failed to load dtypes from test: can't find {e}")

    try:
        dtype_map = {}
        shape_map = {}
        for tensor in tensors:
            dtype_map[tensor["name"]] = get_numpy_type(tensor["type"])
            shape_map[tensor["name"]] = tensor["shape"]
    except Exception as e:
        raise Exception(f"Unable to expand dtype: {e}")

    return [dtype_map[name]
            for name in names], [shape_map[name] for name in names]
