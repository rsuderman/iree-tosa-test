import json
import numpy
import tensor_loader
import tosa_attr_builder

from mlir_builder import *

_mlir_type_map = {
    "BOOL": "i1",
    "INT4": "i4",
    "INT8": "i8",
    "INT16": "i16",
    "INT32": "i32",
    "INT48": "i48",
    "INT64": "i64",
    "FLOAT": "f32",
    "UINT8": "ui8",
    "UINT16": "ui16",
    "UINT32": "ui32",
}


def _get_mlir_type(tosa_type):
    try:
        return _mlir_type_map[tosa_type]
    except:
        raise Exception(f"unknown tosa type - {tosa_type}")


def _build_const_op(func, tensor):
    attribute = tosa_attr_builder.getConstAttribute([tensor])
    opname = tosa_attr_builder.get_tosa_mlir_opname("CONST")
    func.block().add_operator(opname, [], [tensor], attribute)


def _build_tensor_const(func, tensor, etype="i32", shape=None):
    if shape is None:
        shape = [len(tensor)] if isinstance(tensor, list) else []

    name = "<attribute-embedded>"

    value = tensor_loader.Tensor()
    value.load_json_str(tensor, etype, shape)
    tensor = func.add_tensor(name, shape, etype, value)
    return tensor


def _build_extra_inputs(func, operator):
    extra_inputs = []
    if "attribute" not in operator:
        return extra_inputs

    try:
        attribute_type = operator["attribute_type"]
        attribute = operator["attribute"]
        inputs = operator["inputs"]

        if attribute_type == "TransposeAttribute":
            perms = attribute["perms"]
            perms = _build_tensor_const(func, perms, "i32")
            extra_inputs.append(perms)

        if attribute_type == "TableAttribute":
            etype = func.get_tensor(inputs[0]).mlir_type().element_type()
            table = attribute["table"]
            table = _build_tensor_const(func, table, etype)
            extra_inputs.append(table)

        if attribute_type == "PadAttribute":
            padding = attribute["padding"]
            shape = [len(padding) // 2, 2]
            padding = _build_tensor_const(func, padding, "i32", shape)
            extra_inputs.append(padding)

        # Add the const attr.
        if attribute_type == "PadAttribute":
            tensor = func.get_tensor(inputs[0])
            mlir_type = tensor.mlir_type().element_type()
            numpy_type = tensor_loader.get_numpy_type(mlir_type)

            if mlir_type == "f32":
                pad_const_fp = attribute["pad_const_fp"]
                pad_const = numpy.asarray(pad_const_fp).astype(numpy_type)
            else:
                pad_const_int = attribute["pad_const_int"]
                pad_const = numpy.asarray(pad_const_int).astype(numpy_type)
            value = tensor_loader.Tensor()
            value.load_npy(pad_const)

            tensor = func.add_tensor("<pad-attr>", value.shape(), mlir_type,
                                     value)
            extra_inputs.append(tensor)

    except Exception as e:
        raise Exception(f"Extra attr-dict missing required key {e}")

    tensors = []
    for tensor in extra_inputs:
        _build_const_op(func, tensor)
        tensors.append(tensor)

    return tensors


def _build_func_from_tosa(module, dictionary):
    try:
        name = dictionary["name"]
        inputs = dictionary["inputs"]
        outputs = dictionary["outputs"]
        tensors = dictionary["tensors"]
        operators = dictionary["operators"]
    except Exception as e:
        raise Exception("Block dictionary missing required keys")

    func = module.func(name, inputs, outputs)

    for tensor in tensors:
        try:
            name = tensor["name"]
            shape = tensor["shape"]
            tosa_type = tensor["type"]
        except Exception as e:
            raise Exception("Tensor dictionary missing required keys")

        mlir_type = _get_mlir_type(tosa_type)
        value = None
        if "data" in tensor:
            data = numpy.asarray(tensor["data"])
            value = tensor_loader.Tensor()
            value.load_bytes(data, tosa_type, shape)

        func.add_tensor(name, shape, mlir_type, value)

    for operator in operators:
        try:
            op = operator["op"]
            inputs = operator["inputs"]
            outputs = operator["outputs"]
            attribute_type = operator["attribute_type"]
        except Exception as e:
            raise Exception("Operator dictionary missing required keys")

        inputs += _build_extra_inputs(func, operator)

        tensors = func.block().get_tensors(inputs + outputs)
        attribute = tosa_attr_builder.get_tosa_mlir_attribute(
            op, attribute_type, operator, tensors)
        opname = tosa_attr_builder.get_tosa_mlir_opname(op)
        func.block().add_operator(opname, inputs, outputs, attribute)


# Constructs a module from a provided tosa dictionary
def build_from_tosa(dictionary):
    module = MlirModule()
    try:
        blocks = dictionary["blocks"]
    except Exception as e:
        raise Exception("Block-dict missing 'blocks'")

    try:
        for block in blocks:
            _build_func_from_tosa(module, block)
    except Exception as e:
        raise e

    return module
