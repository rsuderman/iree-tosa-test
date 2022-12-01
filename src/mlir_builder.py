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
        if self._ssa is None:
            raise Exception("SSA unassigned")
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
        return self.serialize_ir()

    def serialize_ir(self, indent=""):
        args = ", ".join([str(tensor.ssa()) for tensor in self._inputs])
        arg_types = ", ".join(
            [str(tensor.mlir_type()) for tensor in self._inputs])
        return f'{indent}"{self._name}" ({args}) : ({arg_types}) -> ()'


class Operator:

    def __init__(self, name, inputs, outputs, attributes, blocks):
        self._name = name
        self._inputs = inputs
        self._outputs = outputs
        self._attributes = attributes
        self._blocks = blocks

    def name(self):
        return self._name

    def blocks(self):
        return self._blocks

    def results(self):
        return self._outputs

    def __str__(self):
        return self.serialize_ir()

    def serialize_ir(self, indent=""):
        args = ", ".join([tensor.ssa() for tensor in self._inputs])
        arg_types = ", ".join(
            [str(tensor.mlir_type()) for tensor in self._inputs])
        ret_types = ", ".join(
            [str(tensor.mlir_type()) for tensor in self._outputs])

        ret = ""
        ssas = [output.ssa() for output in self._outputs]

        block_attr = ""
        if self._blocks:
            block_attr = ", ".join(
                [b.serialize_ir(indent + "  ") for b in self._blocks])
            block_attr = "(%s)" % block_attr

        if len(ssas) == 1:
            ret = f"{ssas[0]} = "
        elif len(ssas) > 1:
            base_ssa = self._outputs[0].ssa().split("#")[0]
            for ssa in ssas:
                ssa = ssa.split("#")[0]
                if ssa != base_ssa:
                    print(ssa, base_ssa)
                    raise Exception("Non matching SSA group values")

            ret = f"{base_ssa}:{len(self._outputs)} = "

        attrs = str(self._attributes)
        return f"{indent}{ret} {self._name}({args}) {block_attr} {self._attributes} : ({arg_types}) -> ({ret_types})"


class Block:

    def __init__(self):
        self._inputs = []
        self._outputs = []
        self._tensors = []
        self._operators = []
        self._terminator = None
        self._ssa_count = 0

    def add_operator(self, opname, inputs, outputs, attributes, blocks):
        inputs = inputs
        outputs = outputs
        operator = Operator(opname, inputs, outputs, attributes, blocks)
        self._operators.append(operator)
        return operator

    def add_terminator(self, opname, inputs):
        self._terminator = Return(opname, inputs)

    def add_input(self, input):
        self._inputs.append(input)

    def add_output(self, output):
        self._outputs.append(output)

    def add_tensor(self, shape, mlir_type, value=None):
        tensor_type = TensorType(shape, mlir_type)
        tensor = Tensor(tensor_type)
        if value is not None:
            tensor.set_value(value)
        self._tensors.append(tensor)
        return tensor

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def set_ssas(self, allocator):
        for tensor in self._inputs:
            tensor.register_ssa(allocator.arg_ssa())

        for operator in self._operators:
            results = operator.results()
            ssas = allocator.value_ssas(len(results))
            for tensor, ssa in zip(results, ssas):
                tensor.register_ssa(ssa)
            for block in operator.blocks():
                block.set_ssas(allocator.copy())

    def print_tensors(self):
        for a in self._tensors:
            print(self._tensors[a])

    def serialize_ir(self, indent="", include_block_header=True):
        ops = [" {"]

        if (include_block_header):
            block_args = ", ".join([
                f"{tensor.ssa()} : {tensor.mlir_type()}"
                for tensor in self._inputs
            ])
            ops.append(f"{indent}^bb0({block_args}):")

        for operator in self._operators:
            ops.append(operator.serialize_ir(f"  {indent}"))

        if (self._terminator is not None):
            ops.append(self._terminator.serialize_ir(f"  {indent}"))

        ops.append(indent + "}")
        return "\n".join(ops)

    def __str__(self):
        return self.serialize_ir()


class IdAllocator:

    def __init__(self, arg_ssa=0, value_ssa=0):
        self._arg_ssa = arg_ssa
        self._value_ssa = value_ssa

    def copy(self):
        return IdAllocator(self._arg_ssa, self._value_ssa)

    def arg_ssa(self):
        ssa = f'%arg{self._arg_ssa}'
        self._arg_ssa += 1
        return ssa

    def value_ssa(self):
        return self.value_ssas(1)[0]

    def value_ssas(self, count):
        if count == 0:
            return []
        ssa = self._value_ssa
        self._value_ssa = self._value_ssa + 1
        if count == 1:
            return [f"%{ssa}"]
        return [f"%{ssa}#{i}" for i in range(count)]


class Func:

    def __init__(self, name, block):
        self._name = name
        self._block = block

    def add_tensor(self, shape, mlir_type, value=None):
        return self._block.add_tensor(shape, mlir_type, value)

    def block(self):
        return self._block

    def set_ssas(self):
        self._block.set_ssas(IdAllocator())

    def _func_def(self):
        name = self._name
        argstr = ", ".join(
            [tensor.decl() for tensor in self._block.get_inputs()])
        retstr = ", ".join(
            [tensor.type_decl() for tensor in self._block.get_outputs()])
        return f"func.func @{name}({argstr}) -> ({retstr})"

    def __str__(self):
        func_def = self._func_def()
        block = self._block.serialize_ir("", include_block_header=False)
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
        func.set_ssas()
        self._funcs[name] = func
        return func

    def __str__(self):
        return "\n".join([str(self._funcs[func]) for func in self._funcs])
