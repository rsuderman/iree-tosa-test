import numpy

tosa_name_converter = {
    # Node Ops
    "CONST": "tosa.const",
    "IDENTITY": "tosa.identity",

    # Control Flow
    "WHILE_LOOP": "tosa.while_loop",
    "COND_IF": "tosa.cond_if",

    # Data Layout
    "CONCAT": "tosa.concat",
    "PAD": "tosa.pad",
    "RESHAPE": "tosa.reshape",
    "REVERSE": "tosa.reverse",
    "SLICE": "tosa.slice",
    "TILE": "tosa.tile",
    "TRANSPOSE": "tosa.transpose",

    # Numeric Operators
    "ABS": "tosa.abs",
    "ADD": "tosa.add",
    "SUB": "tosa.sub",
    "CAST": "tosa.cast",
    "CLAMP": "tosa.clamp",
    "CLZ": "tosa.clz",
    "INTDIV": "tosa.div",
    "MUL": "tosa.mul",
    "NEGATE": "tosa.negate",
    "MAXIMUM": "tosa.maximum",
    "MINIMUM": "tosa.minimum",
    "RESCALE": "tosa.rescale",
    "TABLE": "tosa.table",

    # Bitwise Operations
    "ARITHMETIC_RIGHT_SHIFT": "tosa.arithmetic_right_shift",
    "BITWISE_AND": "tosa.bitwise_and",
    "BITWISE_NOT": "tosa.bitwise_not",
    "BITWISE_OR": "tosa.bitwise_or",
    "BITWISE_XOR": "tosa.bitwise_xor",

    # Comparison
    "EQUAL": "tosa.equal",
    "GREATER_EQUAL": "tosa.greater_equal",
    "GREATER": "tosa.greater",

    # Logical Operators.
    "LOGICAL_AND": "tosa.logical_and",
    "LOGICAL_LEFT_SHIFT": "tosa.logical_left_shift",
    "LOGICAL_NOT": "tosa.logical_not",
    "LOGICAL_OR": "tosa.logical_or",
    "LOGICAL_RIGHT_SHIFT": "tosa.logical_right_shift",
    "LOGICAL_XOR": "tosa.logical_xor",

    # Ternary Operators
    "SELECT": "tosa.select",

    # Image Operators
    "RESIZE": "tosa.resize",

    # Reduce Operators
    "REDUCE_ALL": "tosa.reduce_all",
    "REDUCE_ANY": "tosa.reduce_any",
    "REDUCE_MAX": "tosa.reduce_max",
    "REDUCE_MIN": "tosa.reduce_min",
    "REDUCE_SUM": "tosa.reduce_sum",

    # Scatter Gather Operators
    "GATHER": "tosa.gather",
    "SCATTER": "tosa.scatter",

    # Tensor Operators
    "ARGMAX": "tosa.argmax",
    "AVG_POOL2D": "tosa.avg_pool2d",
    "MAX_POOL2D": "tosa.max_pool2d",
    "DEPTHWISE_CONV2D": "tosa.depthwise_conv2d",
    "FULLY_CONNECTED": "tosa.fully_connected",
    "MATMUL": "tosa.matmul",
    "CONV2D": "tosa.conv2d",
    "CONV3D": "tosa.conv3d",
    "TRANSPOSE_CONV2D": "tosa.transpose_conv2d",
}


def get_tosa_mlir_opname(name):
    if name in tosa_name_converter:
        return f"\"{tosa_name_converter[name]}\""
    raise Exception(f"Unknown tosa op name {name}")

def get_i32_dense_array_attr(val):
    return "array<i32: %s>" % ", ".join(str(x) for x in val)

def get_i64_dense_array_attr(val):
    return "array<i64: %s>" % ", ".join(str(x) for x in val)

def getArithmeticRightShiftAttribute(dictionary):
    try:
        attribute = dictionary["attribute"]
        round = attribute["round"]
        return "{ round = %s : i1 }" % (1 if round else 0)
    except Exception as e:
        raise Exception(f"Failed to parse ArithmetricRightShiftAttribute")


def getAxisAttribute(dictionary):
    try:
        attribute = dictionary["attribute"]
        axis = attribute["axis"]
        return "{ axis = %s : i64 }" % axis
    except Exception as e:
        raise Exception(f"Failed to parse AxisAttribute")


def getClampAttribute(dictionary):
    try:
        attribute = dictionary["attribute"]
        min_int = str(attribute["min_int"])
        max_int = str(attribute["max_int"])
        min_fp = "%f" % attribute["min_fp"]
        max_fp = "%f" % attribute["max_fp"]
        return "{min_int = %s : i64, max_int = %s : i64, min_fp = %s : f32, max_fp = %s : f32}" % (
            min_int, max_int, min_fp, max_fp)
    except Exception as e:
        raise Exception(f"Failed to parse ClampAttribute")


def getConstAttribute(tensors):
    if (len(tensors) != 1):
        raise Exception(
            f"ConstAttribute has incorrect number of tensors ({len(tensors)})")
    value = tensors[0].value().serialize_json()
    tensor_type = tensors[0].mlir_type()
    return "{value = dense<%s> : %s}" % (value, tensor_type)


def getMulAttribute(dictionary):
    try:
        attribute = dictionary["attribute"]
        shift = attribute["shift"]
        return "{ shift = %s : i32 }" % (shift)
    except Exception as e:
        raise Exception(f"Failed to parse MulAttribute")


def getNegateAttribute(dictionary):
    try:
        attribute = dictionary["attribute"]
        input_zp = attribute["input1_zp"]
        output_zp = attribute["output_zp"]
        return "{quantization_info = #tosa.unary_quant<input_zp = %s, output_zp = %s>}" % (
            input_zp, output_zp)
    except Exception as e:
        raise Exception(f"Failed to parse NegateAttribute")


def getPadAttribute(dictionary):
    return ""


def getPoolAttribute(dictionary):
    try:
        attribute = dictionary['attribute']
        attr_dict = {
            "kernel":  get_i64_dense_array_attr(attribute['kernel']),
            "pad": get_i64_dense_array_attr(attribute['pad']),
            "stride": get_i64_dense_array_attr(attribute['stride']),
        }

        if "input_zp" in attribute:
            input_zp = str(attribute["input_zp"])
            output_zp = str(attribute["output_zp"])
            qattr = f"#tosa.unary_quant<input_zp = {input_zp}, output_zp = {output_zp}>"
            attr_dict["quantization_info"] = qattr

    except Exception as e:
        raise Exception(f"Failed to parse PoolAttribute")

    attributes = ", ".join(f"{a} = {attr_dict[a]}" for a in attr_dict)
    return "{%s}" % attributes


def getFullyConnectedAttribute(dictionary):
    try:
        attribute = dictionary['attribute']
        input_zp = str(attribute["input_zp"])
        weight_zp = str(attribute["weight_zp"])
        qattr = f"#tosa.conv_quant<input_zp = {input_zp}, weight_zp = {weight_zp}>"
        return "{quantization_info = %s}" % qattr
    except Exception as e:
        raise Exception(f"Failed to parse FullyConnectedAttribute")


def getConvAttribute(dictionary):
    try:
        attribute = dictionary['attribute']
        attr_dict = {
            "stride": get_i64_dense_array_attr(attribute["stride"]),
            "dilation": get_i64_dense_array_attr(attribute["dilation"]),
            "pad": get_i64_dense_array_attr(attribute["pad"]),
        }

        if "input_zp" in attribute:
            input_zp = str(attribute["input_zp"])
            weight_zp = str(attribute["weight_zp"])
            qattr = f"#tosa.conv_quant<input_zp = {input_zp}, weight_zp = {weight_zp}>"
            attr_dict["quantization_info"] = qattr

        attr = ", ".join([f"{a} = {attr_dict[a]}" for a in attr_dict])
        return "{%s}" % attr
    except Exception as e:
        raise Exception(f"Failed to parse ConvAttribute")

def getTransposeConvAttribute(dictionary):
    try:
        attribute = dictionary['attribute']
        attr_dict = {
            "out_pad": get_i64_dense_array_attr(attribute["out_pad"]),
            "out_shape": get_i64_dense_array_attr(attribute["output_shape"]),
            "stride": get_i64_dense_array_attr(attribute["stride"]),
        }

        if "input_zp" in attribute:
            input_zp = str(attribute["input_zp"])
            weight_zp = str(attribute["weight_zp"])
            qattr = f"#tosa.conv_quant<input_zp = {input_zp}, weight_zp = {weight_zp}>"
            attr_dict["quantization_info"] = qattr

        attr = ", ".join([f"{a} = {attr_dict[a]}" for a in attr_dict])
        return "{%s}" % attr
    except Exception as e:
        print(e)
        raise Exception(f"Failed to parse TransposeConvAttribute")

def getMatMulConnectedAttribute(dictionary):
    try:
        attribute = dictionary['attribute']
        a_zp = str(attribute["a_zp"])
        b_zp = str(attribute["b_zp"])
        qattr = f"#tosa.matmul_quant<a_zp = {a_zp}, b_zp = {b_zp}>"
        return "{quantization_info = %s}" % qattr
    except Exception as e:
        raise Exception(f"Failed to parse MatMulConnectedAttribute")


def getRescaleAttribute(dictionary):
    try:
        attribute = dictionary["attribute"]
        attributes = []
        attributes.append("input_zp = %i : i32" % attribute["input_zp"])
        attributes.append("output_zp = %i : i32" % attribute["output_zp"])
        attributes.append("scale32 = %s" % str(attribute["scale32"]).lower())
        attributes.append("double_round = %s" %
                          str(attribute["double_round"]).lower())
        attributes.append("per_channel = %s" %
                          str(attribute["per_channel"]).lower())

        multiplier = attribute["multiplier"]
        attributes.append(f"multiplier = {get_i32_dense_array_attr(multiplier)}")

        shift = attribute["shift"]
        attributes.append(f"shift = {get_i32_dense_array_attr(shift)}")
        attributes = "{%s}" % ", ".join(attributes)
        return attributes
    except Exception as e:
        raise Exception(f"Failed to parse RescaleAttribute")


def getReshapeAttribute(dictionary):
    try:
        attribute = dictionary['attribute']
        new_shape = attribute['new_shape']
        return "{new_shape = %s}" % get_i64_dense_array_attr(new_shape)
    except Exception as e:
        raise Exception(f"Failed to parse ReshapeAttribute")


def getResizeAttribute(dictionary):
    try:
        attribute = dictionary['attribute']
        mode = attribute['mode']
        mode = "NEAREST_NEIGHBOR" if mode == "NEAREST" else mode
        attr_dict = {
            "scale": get_i64_dense_array_attr(attribute["scale"]),
            "offset": get_i64_dense_array_attr(attribute["offset"]),
            "border": get_i64_dense_array_attr(attribute["border"]),
            "mode": '"%s"' % mode
        }
        attrs = ", ".join(f"{a} = {attr_dict[a]}" for a in attr_dict)
        return "{%s}" % attrs
    except Exception as e:
        raise Exception(f"Failed to parse ReshapeAttribute")


def getSliceAttribute(dictionary):
    try:
        attribute = dictionary['attribute']
        start = get_i64_dense_array_attr(attribute['start'])
        size = get_i64_dense_array_attr(attribute['size'])
        return "{start = %s, size = %s}" % (start, size)
    except Exception as e:
        raise Exception(f"Failed to parse SliceAttribute")


def getTileAttribute(dictionary):
    try:
        attribute = dictionary['attribute']
        multiples = get_i64_dense_array_attr(attribute['multiples'])
        return "{multiples = %s}" % multiples
    except Exception as e:
        raise Exception(f"Failed to parse TileAttribute")


def getCondIfBlocks(dictionary):
    try:
        attribute = dictionary['attribute']
        then_block = attribute["then_branch"]
        else_block = attribute["else_branch"]
        return [then_block, else_block]
    except Exception as e:
        raise Exception(f"Failed to parse WhileAttribute")


def getWhileBlocks(dictionary):
    try:
        attribute = dictionary['attribute']
        cond_block = attribute["cond_branch"]
        body_block = attribute["body_branch"]
        return [cond_block, body_block]
    except Exception as e:
        raise Exception(f"Failed to parse WhileAttribute")


def get_tosa_mlir_blocks(attribute_type, dictionary):
    if (attribute_type == "CondIfAttribute"):
        return getCondIfBlocks(dictionary)

    if (attribute_type == "WhileLoopAttribute"):
        return getWhileBlocks(dictionary)

    return []


def get_tosa_mlir_attribute(opname, attribute_type, dictionary, tensors):
    if (opname == "CONST"):
        return getConstAttribute(tensors)

    if (attribute_type == "NONE"):
        return ""

    if (attribute_type == "ArithmeticRightShiftAttribute"):
        return getArithmeticRightShiftAttribute(dictionary)

    if (attribute_type == "AxisAttribute"):
        return getAxisAttribute(dictionary)

    if (attribute_type == "ClampAttribute"):
        return getClampAttribute(dictionary)

    if (attribute_type == "FullyConnectedAttribute"):
        return getFullyConnectedAttribute(dictionary)

    if (attribute_type == "ConvAttribute"):
        return getConvAttribute(dictionary)

    if (attribute_type == "MatMulAttribute"):
        return getMatMulConnectedAttribute(dictionary)

    if (attribute_type == "MulAttribute"):
        return getMulAttribute(dictionary)

    if (attribute_type == "NegateAttribute"):
        return getNegateAttribute(dictionary)

    if (attribute_type == "PadAttribute"):
        return getPadAttribute(dictionary)

    if (attribute_type == "PoolAttribute"):
        return getPoolAttribute(dictionary)

    if (attribute_type == "RescaleAttribute"):
        return getRescaleAttribute(dictionary)

    if (attribute_type == "ReshapeAttribute"):
        return getReshapeAttribute(dictionary)

    if (attribute_type == "ResizeAttribute"):
        return getResizeAttribute(dictionary)

    if (attribute_type == "SliceAttribute"):
        return getSliceAttribute(dictionary)

    if (attribute_type == "TableAttribute"):
        return ""

    if (attribute_type == "TileAttribute"):
        return getTileAttribute(dictionary)

    if (attribute_type == "TransposeConvAttribute"):
        return getTransposeConvAttribute(dictionary)

    if (attribute_type == "WhileLoopAttribute"
            or attribute_type == "CondIfAttribute"):
        return ""

    # This one is special as its serialized as an extra constant earlier
    if (attribute_type == "TransposeAttribute"):
        return ""

    raise Exception(f"Unsupported attribute type {attribute_type}")
