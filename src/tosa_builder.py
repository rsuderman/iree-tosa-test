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


def _build_const_op(block, tensor):
    attribute = tosa_attr_builder.getConstAttribute([tensor])
    opname = tosa_attr_builder.get_tosa_mlir_opname("CONST")
    block.add_operator(opname, [], [tensor], attribute, {})


def _build_tensor_const(block, tensor, etype="i32", shape=None):
    if shape is None:
        shape = [len(tensor)] if isinstance(tensor, list) else []

    value = tensor_loader.Tensor()
    value.load_json_str(tensor, etype, shape)
    tensor = block.add_tensor(shape, etype, value)
    return tensor


def _build_extra_inputs(block, operator, tensor_map):
    extra_inputs = []
    if "attribute" not in operator:
        return extra_inputs

    try:
        attribute_type = operator["attribute_type"]
        attribute = operator["attribute"]
        inputs = operator["inputs"]

        if attribute_type == "TransposeAttribute":
            perms = attribute["perms"]
            perms = _build_tensor_const(block, perms, "i32")
            extra_inputs.append(perms)

        if attribute_type == "TableAttribute":
            etype = tensor_map[inputs[0]].mlir_type().element_type()
            table = attribute["table"]
            table = _build_tensor_const(block, table, etype)
            extra_inputs.append(table)

        if attribute_type == "PadAttribute":
            padding = attribute["padding"]
            shape = [len(padding) // 2, 2]
            padding = _build_tensor_const(block, padding, "i32", shape)
            extra_inputs.append(padding)

        # Add the const attr.
        if attribute_type == "PadAttribute":
            tensor = tensor_map[inputs[0]]
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

            tensor = block.add_tensor(value.shape(), mlir_type, value)
            extra_inputs.append(tensor)

    except Exception as e:
        raise Exception(f"Extra attr-dict missing required key {e}")

    tensors = []
    for tensor in extra_inputs:
        _build_const_op(block, tensor)
        tensors.append(tensor)

    return tensors


def _build_block_from_tosa(block, dictionary, block_map):
    try:
        name = dictionary["name"]
        inputs = dictionary["inputs"]
        outputs = dictionary["outputs"]
        tensors = dictionary["tensors"]
        operators = dictionary["operators"]
    except Exception as e:
        raise Exception("Block dictionary missing required keys")

    tensor_map = {}
    for tensor in tensors:
        try:
            argname = tensor["name"]
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

        tensor_map[argname] = block.add_tensor(shape, mlir_type, value)

    [block.add_input(tensor_map[tensor]) for tensor in inputs]
    [block.add_output(tensor_map[tensor]) for tensor in outputs]

    for operator in operators:
        try:
            op = operator["op"]
            args = [tensor_map[arg] for arg in operator["inputs"]]
            rets = [tensor_map[arg] for arg in operator["outputs"]]
            attribute_type = operator["attribute_type"]
        except Exception as e:
            raise Exception("Operator dictionary missing required keys")

        args += _build_extra_inputs(block, operator, tensor_map)

        tensors = args + rets
        attribute = tosa_attr_builder.get_tosa_mlir_attribute(
            op, attribute_type, operator, tensors)
        blocks = tosa_attr_builder.get_tosa_mlir_blocks(
            attribute_type, operator)
        blocks = [block_map[bn] for bn in blocks]

        opname = tosa_attr_builder.get_tosa_mlir_opname(op)

        new_op = block.add_operator(opname, args, rets, attribute, blocks)

    outputs = [tensor_map[output] for output in outputs]
    if name == "main":
        block.add_terminator("func.return", outputs)
    else:
        block.add_terminator("tosa.yield", outputs)
    return block


# Constructs a module from a provided tosa dictionary
def build_from_tosa(dictionary):
    module = MlirModule()
    try:
        blocks = dictionary["blocks"]
    except Exception as e:
        raise Exception("Block-dict missing 'blocks'")

    try:
        block_map = {block["name"]: module.block() for block in blocks}
        for block in blocks:
            name = block["name"]
            _build_block_from_tosa(block_map[name], block, block_map)
    except Exception as e:
        raise e

    if "main" not in block_map:
        raise Exception("No 'main' function found.")

    module.func("main", block_map["main"])

    return module
