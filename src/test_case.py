import hjson as json
import iree.compiler
import iree.runtime
import numpy
import os

import tosa_builder
import tensor_loader

DESC_FILE = "desc.json"


class Status:

    def __init__(self, message=""):
        self._details = []
        if (message == ""):
            self._err = False
            self._message = ""
            return

        self._err = True
        self._message = message

    def failure(self):
        return self._err

    def success(self):
        return not self._err

    def message(self):
        return self._message

    def details(self):
        return self._details

    def add_details(self, detail):
        self._details.append(detail)


class TestCase:

    def __init__(self, path):
        self._path = path

        # Break out each part of the test.
        path, self._test = os.path.split(path)
        path, self._operator = os.path.split(path)
        path, self._group = os.path.split(path)

        self._name = os.path.join(self._group, self._operator, self._test)

        # Requirements for the test
        self._tosa = None
        self._inputs = None
        self._outputs = None
        self._results = None

        self._status = self._parse_desc()
        self._fix_paths()

    def _parse_desc(self):
        desc_path = os.path.join(self._path, DESC_FILE)
        data = None
        try:
            with open(desc_path, "r") as f:
                data = json.load(f)
        except:
            status = Status("Failed to open descriptor")
            return status

        try:
            self._tosa_file = data["tosa_file"]
            self._inputs = data["ifm_name"]
            self._outputs = data["expected_result_file"]
        except:
            return Status("Invalid descriptor file")

        return Status()

    def _fix_paths(self):
        self._tosa_file = os.path.join(self._path, self._tosa_file)
        self._inputs = [os.path.join(self._path, f) for f in self._inputs]
        self._outputs = [os.path.join(self._path, f) for f in self._outputs]

    def _print(self):
        print(self._group, self._operator, self._test)
        print(self._error, self._message)

    def name(self):
        return self._name

    def generate_mlir(self):
        if (self._status.failure()):
            Status(self._message)

        try:
            with open(self._tosa_file, "r") as f:
                dictionary = json.load(f)
        except:
            raise Exception(f"Failed to load/parse {f}")

        try:
            self._ir = str(tosa_builder.build_from_tosa(dictionary))
        except Exception as e:
            return Status(f"failed to build test IR - {str(e)}")

        return Status()

    def compile_ir(self):
        try:
            self._compiled = iree.compiler.compile_str(
                self._ir,
                target_backends=["llvm-cpu"],
                input_type=iree.compiler.InputType.TOSA)
        except Exception as e:
            return Status(str(e))

        return Status()

    def load_tensors(self):
        # We have to use the dtypes from the TOSA file.
        input_dtypes, input_shapes = tensor_loader.parse_dtypes(
            self._tosa_file, "inputs")
        output_dtypes, output_shapes = tensor_loader.parse_dtypes(
            self._tosa_file, "outputs")

        inputs = []
        for path, dtype, shape in zip(self._inputs, input_dtypes,
                                      input_shapes):
            tensor = tensor_loader.Tensor()
            tensor.load_json(path)
            npy = tensor.npy().astype(dtype).reshape(shape)
            inputs.append(npy)

        outputs = []
        for path, dtype, shape in zip(self._outputs, output_dtypes,
                                      output_shapes):
            path, _ = os.path.splitext(path)
            tensor = tensor_loader.Tensor()
            tensor.load_json(path)
            npy = tensor.npy().astype(dtype).reshape(shape)
            outputs.append(npy)
        return inputs, outputs

    def execute(self):
        # Start by loading input data.
        try:
            inputs, expected = self.load_tensors()
        except Exception as e:
            status = Status("Failed to load input/outputs")
            status.add_details(str(e))
            return status

        # Load compiled module.
        try:
            instance = iree.runtime.VmInstance()
            rt_config = iree.runtime.system_api.Config("local-task")
            vm_module = iree.runtime.VmModule.from_flatbuffer(
                instance, self._compiled)
            self._runtime = iree.runtime.system_api.load_vm_module(
                vm_module, rt_config)
        except Exception as e:
            return Status("Failed to Initialize Runtime")

        # Execute with sample inputs.
        try:
            returned = self._runtime["main"](*inputs)
            if not isinstance(returned, list):
                returned = [returned]
        except Exception as e:
            status = Status("Failed to execute")
            status.add_details(str(e))
            return status

        self._returned = [r.to_host() for r in returned]
        self._expected = expected
        return Status()

    def compare(self):
        errors = []
        for i, (a, b) in enumerate(zip(self._returned, self._expected)):
            if a.dtype != b.dtype:
                errors.append(f"mismatch in dtype ({i})")
                continue

            if a.shape != b.shape:
                errors.append(f"mismatch in shape ({i})")
                continue

            if a.dtype == numpy.float:
                # TODO(suderman): Probably do a relative comparison.
                if (numpy.abs(a - b) > 1e-6):
                    errors.append(f"mismatch in fp value ({i})")
            else:
                if (numpy.any(a != b)):
                    errors.append(f"mismatch in int value ({i})")

        if len(errors) > 0:
            status = Status("Mismatch in results")
            for error in errors:
                status.add_details(error)
            return status

        return Status()

    def run(self):
        status = self.generate_mlir()
        if (status.failure()):
            return status

        status = self.compile_ir()
        if (status.failure()):
            return status

        status = self.execute()
        if (status.failure()):
            return status

        status = self.compare()
        return status
