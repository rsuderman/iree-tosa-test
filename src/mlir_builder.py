class Tensor:

    def __init__(self, tensor_type):
        self._type = tensor_type
        self._ssa = None
        self._value = None

    def mlir_type(self):
        return self._type

    def type_decl(self):
        return str(self._type)

    def register_ssa(self, ssa):
        if self._ssa:
            raise Exception("Attempted to reassign SSA")
        self._ssa = ssa

    def value(self):
        return self._value

    def set_value(self, value):
        if (tuple(value.shape()) != tuple(self._type.shape())):
            raise Exception("Shapes do not match %s %s" %
                            (str(value.shape()), str(self._type.shape())))

        self._value = value

    def __str__(self):
        return f"{self._type}"

    def ssa(self):
        return self._ssa

    def decl(self):
        return f"{self._ssa} : {self._type}"


class TensorType:

    def __init__(self, shape, mlir_type):
        self._shape = shape
        self._mlir_type = mlir_type

    def element_type(self):
        return self._mlir_type

    def shape(self):
        return self._shape

    def __str__(self):
        elements = list(self._shape) + [self.element_type()]
        contents = "x".join([str(e) for e in elements])
        return f"tensor<{contents}>"


class Return:

    def __init__(self, name, inputs):
        self._name = name
        self._inputs = inputs

    def __str__(self):
        args = ", ".join([str(tensor.ssa()) for tensor in self._inputs])
        arg_types = ", ".join(
            [str(tensor.mlir_type()) for tensor in self._inputs])
        return f"  {self._name} {args} : {arg_types}"


class Operator:

    def __init__(self, name, inputs, outputs, attributes):
        self._name = name
        self._inputs = inputs
        self._outputs = outputs
        self._attributes = attributes

    def __str__(self):
        args = ", ".join([tensor.ssa() for tensor in self._inputs])
        arg_types = ", ".join(
            [str(tensor.mlir_type()) for tensor in self._inputs])
        ret_types = ", ".join(
            [str(tensor.mlir_type()) for tensor in self._outputs])

        if len(self._outputs) != 1:
            raise Exception("Error : Multiple returns not yet support")

        ret = self._outputs[0].ssa()

        attrs = str(self._attributes)
        return f"  {ret} = {self._name}({args}) {self._attributes} : ({arg_types}) -> ({ret_types})"


class Block:

    def __init__(self):
        self._inputs = []
        self._outputs = []
        self._tensors = []
        self._operators = []
        self._terminator = None

    def add_operator(self, opname, inputs, outputs, attributes):
        inputs = inputs
        outputs = outputs
        operator = Operator(opname, inputs, outputs, attributes)
        self._operators.append(operator)

    def add_terminator(self, opname, inputs):
        self._terminator = Return(opname, inputs)

    def add_input(self, input):
        self._inputs.append(input)

    def add_output(self, output):
        self._outputs.append(output)

    def add_tensor(self, shape, mlir_type, value=None):
        tensor_type = TensorType(shape, mlir_type)
        tensor = Tensor(tensor_type)
        tensor.register_ssa(f"%{str(len(self._tensors))}")
        if value is not None:
            tensor.set_value(value)
        self._tensors.append(tensor)
        return tensor

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def print_tensors(self):
        for a in self._tensors:
            print(self._tensors[a])

    def __str__(self):
        ops = [" {"]

        for operator in self._operators:
            ops.append(str(operator))

        if (self._terminator is not None):
            ops.append(str(self._terminator))

        ops.append("}")
        return "\n".join(ops)


class Func:

    def __init__(self, name, block):
        self._name = name
        self._block = block

    def add_tensor(self, shape, mlir_type, value=None):
        return self._block.add_tensor(shape, mlir_type, value)

    # def get_tensor(self, name):
    #     return self._block.get_tensor(name)

    def block(self):
        return self._block

    def _func_def(self):
        name = self._name
        argstr = ", ".join(
            [tensor.decl() for tensor in self._block.get_inputs()])
        retstr = ", ".join(
            [tensor.type_decl() for tensor in self._block.get_outputs()])
        return f"func.func @{name}({argstr}) -> ({retstr})"

    def __str__(self):
        func_def = self._func_def()
        block = str(self._block)
        return f"{func_def}{block}"


class MlirModule:

    def __init__(self):
        self._funcs = {}

    def block(self):
        return Block()

    def func(self, name, block):
        if name in self._funcs:
            raise Exception(f"Attempted to duplicate func: {name}")

        func = Func(name, block)
        self._funcs[name] = func
        return func

    def __str__(self):
        return "\n".join([str(self._funcs[func]) for func in self._funcs])
