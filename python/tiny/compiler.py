"""Common compiler infrastructure for the tutorial DSL.

Generates MLIR IR as plain text strings, avoiding mlir-python-bindings
which has a conda-forge packaging bug with builtin.unrealized_conversion_cast.
"""

import subprocess
import os
import inspect
from typing import Callable


def _get_tutorial_opt() -> str:
    path = os.environ.get("TUTORIAL_OPT")
    if not path:
        raise RuntimeError(
            "TUTORIAL_OPT environment variable not set.\n"
            "Please set it to the path of tutorial-opt binary:\n"
            "  export TUTORIAL_OPT=/path/to/build/tutorial/tutorial-opt"
        )
    return path


class TutorialOpt:
    """Wrapper around tutorial-opt for running passes."""

    def __init__(self, binary_path: str = None):
        self.binary = binary_path or _get_tutorial_opt()

    def run(self, ir: str, passes: list[str]) -> str:
        args = [self.binary] + [f"--{p}" for p in passes]
        result = subprocess.run(args, input=ir, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"tutorial-opt failed:\n{result.stderr}")
        return result.stdout


class SSAValue:
    """An SSA value reference in MLIR IR text."""
    __slots__ = ("name", "type_str")

    def __init__(self, name: str, type_str: str):
        self.name = name
        self.type_str = type_str


class IRBuilder:
    """Builds MLIR IR as text lines. Used via class-level current builder."""
    _current: "IRBuilder | None" = None

    def __init__(self):
        self._counter = 0
        self._lines: list[str] = []
        self._indent = 2

    @classmethod
    def get(cls) -> "IRBuilder":
        assert cls._current is not None, "No active IRBuilder"
        return cls._current

    def __enter__(self):
        IRBuilder._current = self
        return self

    def __exit__(self, *args):
        IRBuilder._current = None

    def new_value(self, type_str: str) -> SSAValue:
        name = f"%{self._counter}"
        self._counter += 1
        return SSAValue(name, type_str)

    def emit(self, text: str):
        self._lines.append("  " * self._indent + text)

    def op(self, op_name: str, operands: list[SSAValue] | None = None,
           result_types: list[str] | None = None,
           attributes: dict[str, str] | None = None) -> "SSAValue | list[SSAValue] | None":
        operands = operands or []
        result_types = result_types or []
        attributes = attributes or {}

        if len(result_types) == 0:
            prefix = ""
            results = []
        elif len(result_types) == 1:
            v = self.new_value(result_types[0])
            prefix = f"{v.name} = "
            results = [v]
        else:
            base = self._counter
            self._counter += 1
            results = [SSAValue(f"%{base}#{i}", t)
                       for i, t in enumerate(result_types)]
            prefix = f"%{base}:{len(results)} = "

        ops_str = ", ".join(v.name for v in operands)
        attr_str = ""
        if attributes:
            parts = [f"{k} = {v}" for k, v in attributes.items()]
            attr_str = " {" + ", ".join(parts) + "}"
        in_types = ", ".join(v.type_str for v in operands)
        if len(result_types) == 1:
            out_str = result_types[0]
        else:
            out_str = "(" + ", ".join(result_types) + ")"

        self.emit(
            f'{prefix}"{op_name}"({ops_str}){attr_str} : ({in_types}) -> {out_str}'
        )

        if len(results) == 1:
            return results[0]
        return results if results else None


class MLIRModule:
    """Builds an MLIR module as text."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def build_func(self, fn: Callable, type_map: dict) -> str:
        sig = inspect.signature(fn)

        input_types: list[str] = []
        wrapper_classes = []
        for param in sig.parameters.values():
            if param.annotation not in type_map:
                raise ValueError(f"Unsupported type: {param.annotation}")
            type_str, wrapper_cls = type_map[param.annotation]
            input_types.append(type_str)
            wrapper_classes.append(wrapper_cls)

        builder = IRBuilder()
        with builder:
            func_args = [SSAValue(f"%arg{i}", t)
                         for i, t in enumerate(input_types)]
            dsl_args = [cls._wrap(func_args[i])
                        for i, cls in enumerate(wrapper_classes)]
            fn(*dsl_args)
            builder.emit('"func.return"() : () -> ()')

        args_str = ", ".join(f"{a.name}: {a.type_str}" for a in func_args)
        body = "\n".join(builder._lines)

        return (
            f"module {{\n"
            f"  func.func @{fn.__name__}({args_str}) {{\n"
            f"{body}\n"
            f"  }}\n"
            f"}}\n"
        )

    def build_func_verified(self, fn: Callable, type_map: dict,
                            opt: "TutorialOpt") -> str:
        raw_ir = self.build_func(fn, type_map)
        return opt.run(raw_ir, [])
