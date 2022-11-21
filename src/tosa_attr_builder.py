import numpy

tosa_name_converter = {
    # Node Ops
    "CONST": "tosa.const",
    "IDENTITY": "tosa.identity",

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

    # Tensor Operators
    "ARGMAX": "tosa.argmax",
    "AVG_POOL2D": "tosa.avg_pool2d",
    "MAX_POOL2D": "tosa.max_pool2d",
    "DEPTHWISE_CONV2D": "tosa.depthwise_conv2d",
    "FULLY_CONNECTED": "tosa.fully_connected",
    "MATMUL": "tosa.matmul",
    "CONV2D": "tosa.conv2d",
    "CONV3D": "tosa.conv3d",
}


def get_tosa_mlir_opname(name):
    if name in tosa_name_converter:
        return f"\"{tosa_name_converter[name]}\""
    raise Exception(f"Unknown tosa op name {name}")


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
            "kernel": str(attribute['kernel']),
            "pad": str(attribute['pad']),
            "stride": str(attribute['stride']),
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
            "stride": attribute["stride"],
            "dilation": attribute["dilation"],
            "pad": attribute["pad"],
        }

        if "input_zp" in attribute:
            input_zp = str(attribute["input_zp"])
            weight_zp = str(attribute["weight_zp"])
            qattr = f"#tosa.conv_quant<input_zp = {input_zp}, weight_zp = {weight_zp}>"
            attr_dict["quantization_info"] = qattr

        attr = ", ".join([f"{a} = {attr_dict[a]}" for a in attr_dict])
        return "{%s}" % attr
    except Exception as e:
        raise Exception(f"Failed to parse FullyConnectedAttribute")


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

        multiplier = ", ".join([f"{m} : i32" for m in attribute["multiplier"]])
        attributes.append(f"multiplier = [{multiplier}]")

        shift = ", ".join(["%i : i32" % m for m in attribute["shift"]])
        attributes.append(f"shift = [{shift}]")
        attributes = "{%s}" % ", ".join(attributes)
        return attributes
    except Exception as e:
        raise Exception(f"Failed to parse RescaleAttribute")


def getReshapeAttribute(dictionary):
    try:
        attribute = dictionary['attribute']
        new_shape = attribute['new_shape']
        return "{new_shape = %s}" % str(new_shape)
    except Exception as e:
        raise Exception(f"Failed to parse ReshapeAttribute")


def getResizeAttribute(dictionary):
    try:
        attribute = dictionary['attribute']
        mode = attribute['mode']
        mode = "NEAREST_NEIGHBOR" if mode == "NEAREST" else mode
        attr_dict = {
            "scale": str(attribute["scale"]),
            "offset": str(attribute["offset"]),
            "border": str(attribute["border"]),
            "mode": '"%s"' % mode
        }
        attrs = ", ".join(f"{a} = {attr_dict[a]}" for a in attr_dict)
        return "{%s}" % attrs
    except Exception as e:
        raise Exception(f"Failed to parse ReshapeAttribute")


def getSliceAttribute(dictionary):
    try:
        attribute = dictionary['attribute']
        start = str(attribute['start'])
        size = str(attribute['size'])
        return "{start = %s, size = %s}" % (start, size)
    except Exception as e:
        raise Exception(f"Failed to parse SliceAttribute")


def getTileAttribute(dictionary):
    try:
        attribute = dictionary['attribute']
        multiples = str(attribute['multiples'])
        return "{multiples = %s}" % multiples
    except Exception as e:
        raise Exception(f"Failed to parse TileAttribute")


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

    # This one is special as its serialized as an extra constant earlier
    if (attribute_type == "TransposeAttribute"):
        return ""

    raise Exception(f"Unsupported attribute type {attribute_type}")
