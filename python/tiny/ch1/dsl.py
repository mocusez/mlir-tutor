"""DSL wrapper types for Tiny dialect."""

from ..compiler import SSAValue, IRBuilder


def _format_float(val: float) -> str:
    if val == 0.0:
        return "0.000000e+00"
    return f"{val:.6e}"


class Index:
    """Wrapper for index-typed SSA values."""
    _value: SSAValue

    @staticmethod
    def _wrap(value: SSAValue) -> "Index":
        idx = Index()
        idx._value = value
        return idx

    @staticmethod
    def constant(val: int) -> "Index":
        b = IRBuilder.get()
        result = b.op("tiny.constant", result_types=["index"],
                      attributes={"value": f"{val} : index"})
        return Index._wrap(result)

    def _binop(self, other: "Index", op_name: str) -> "Index":
        b = IRBuilder.get()
        result = b.op(f"tiny.{op_name}",
                      operands=[self._value, other._value],
                      result_types=["index"])
        return Index._wrap(result)

    def __add__(self, other): return self._binop(other, "addi")
    def __sub__(self, other): return self._binop(other, "subi")
    def __mul__(self, other): return self._binop(other, "muli")
    def __floordiv__(self, other): return self._binop(other, "divi")


class F16Vector:
    """Wrapper for vector<Nxf16> SSA values."""
    _value: SSAValue

    @staticmethod
    def _wrap(value: SSAValue) -> "F16Vector":
        vec = F16Vector()
        vec._value = value
        return vec

    @staticmethod
    def constant(vals: list[float], size: int = None) -> "F16Vector":
        n = size or len(vals)
        type_str = f"vector<{n}xf16>"
        b = IRBuilder.get()
        if len(vals) == 1:
            dense = f"dense<{_format_float(vals[0])}> : {type_str}"
        else:
            formatted = ", ".join(_format_float(v) for v in vals)
            dense = f"dense<[{formatted}]> : {type_str}"
        result = b.op("tiny.constant", result_types=[type_str],
                      attributes={"value": dense})
        return F16Vector._wrap(result)

    def _binop(self, other: "F16Vector", op_name: str) -> "F16Vector":
        b = IRBuilder.get()
        result = b.op(f"tiny.{op_name}",
                      operands=[self._value, other._value],
                      result_types=[self._value.type_str])
        return F16Vector._wrap(result)

    def __add__(self, other): return self._binop(other, "addf")
    def __sub__(self, other): return self._binop(other, "subf")
    def __mul__(self, other): return self._binop(other, "mulf")
    def __truediv__(self, other): return self._binop(other, "divf")

    def sum(self) -> "F16Vector":
        b = IRBuilder.get()
        result = b.op("tiny.sum",
                      operands=[self._value],
                      result_types=["vector<1xf16>"])
        return F16Vector._wrap(result)


class Ptr:
    """Wrapper for !tiny.ptr SSA values."""
    _value: SSAValue

    @staticmethod
    def _wrap(value: SSAValue) -> "Ptr":
        p = Ptr()
        p._value = value
        return p

    def load(self, offset: Index, num_elements: int) -> F16Vector:
        vec_type = f"vector<{num_elements}xf16>"
        b = IRBuilder.get()
        result = b.op("tiny.load",
                      operands=[self._value, offset._value],
                      result_types=[vec_type])
        return F16Vector._wrap(result)

    def store(self, offset: Index, vec: F16Vector) -> None:
        b = IRBuilder.get()
        b.op("tiny.store",
             operands=[vec._value, self._value, offset._value])


# --- Convenience functions ---

from ..compiler import MLIRModule, TutorialOpt


def _get_type_map():
    return {
        Ptr: ("!tiny.ptr", Ptr),
        Index: ("index", Index),
    }


def print_ir(fn):
    """Print generated Tiny dialect IR (verified and pretty-printed)."""
    opt = TutorialOpt()
    with MLIRModule() as m:
        tiny_ir = m.build_func_verified(fn, _get_type_map(), opt)
        print(tiny_ir)
    return fn


def compile_and_print(fn):
    """Compile and print all lowering stages."""
    opt = TutorialOpt()

    with MLIRModule() as m:
        tiny_ir = m.build_func_verified(fn, _get_type_map(), opt)

    print("=== Tiny Dialect ===")
    print(tiny_ir)

    arith_ir = opt.run(tiny_ir, ["tiny-to-arith", "canonicalize", "cse"])
    print("=== After tiny-to-arith ===")
    print(arith_ir)

    llvm_ir = opt.run(tiny_ir, ["tiny-to-arith", "canonicalize", "cse",
                                "tiny-to-llvm", "convert-to-llvm"])
    print("=== LLVM Dialect ===")
    print(llvm_ir)

    return llvm_ir
