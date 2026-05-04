"""DSL for TinyTile dialect (Chapter 3)."""

from ..ch1.dsl import Index, Ptr, F16Vector
from ..compiler import IRBuilder, SSAValue, MLIRModule, TutorialOpt


class Layout:
    """Thread distribution layout for tiles."""

    def __init__(self, thread: tuple[int, int], vector_size: int):
        self.thread = thread
        self.vector_size = vector_size

    def num_elements(self) -> int:
        return self.thread[0] * self.thread[1] * self.vector_size


class Tile:
    """Wrapper for !tiny_tile.tile SSA values."""
    _value: SSAValue
    _height: int
    _width: int
    _layout: Layout

    @staticmethod
    def _wrap(value: SSAValue, height: int = None, width: int = None,
              layout: Layout = None, *, template: "Tile" = None) -> "Tile":
        t = Tile()
        t._value = value
        if template is not None:
            t._height = template._height
            t._width = template._width
            t._layout = template._layout
        else:
            t._height = height
            t._width = width
            t._layout = layout
        return t

    @staticmethod
    def _type_str(height: int, width: int, layout: Layout) -> str:
        return (
            f"!tiny_tile.tile<{height}x{width}, "
            f"#tiny_tile.layout<thread = [{layout.thread[0]}, {layout.thread[1]}], "
            f"vector_size = {layout.vector_size}>>"
        )

    def _binop(self, other: "Tile", kind: str) -> "Tile":
        if (self._layout.thread != other._layout.thread or
                self._layout.vector_size != other._layout.vector_size):
            raise ValueError("Operands must have the same layout")
        result_type = Tile._type_str(self._height, self._width, self._layout)
        b = IRBuilder.get()
        result = b.op("tiny_tile.elementwise",
                      operands=[self._value, other._value],
                      result_types=[result_type],
                      attributes={"kind": f"#tiny_tile<ew_kind {kind}>"})
        return Tile._wrap(result, self._height, self._width, self._layout)

    def __add__(self, other): return self._binop(other, "add")
    def __sub__(self, other): return self._binop(other, "sub")
    def __mul__(self, other): return self._binop(other, "mul")
    def __truediv__(self, other): return self._binop(other, "div")

    @staticmethod
    def splat(value: float, height: int, width: int, layout: Layout) -> "Tile":
        tile_type = Tile._type_str(height, width, layout)
        val_str = f"{value:.6e}" if value != 0.0 else "0.000000e+00"
        b = IRBuilder.get()
        result = b.op("tiny_tile.splat",
                      result_types=[tile_type],
                      attributes={"splatValue": f"{val_str} : f64"})
        return Tile._wrap(result, height, width, layout)

    @staticmethod
    def zeros(height: int, width: int, layout: Layout) -> "Tile":
        return Tile.splat(0.0, height, width, layout)

    def sum(self) -> F16Vector:
        b = IRBuilder.get()
        result = b.op("tiny_tile.sum",
                      operands=[self._value],
                      result_types=["vector<1xf16>"])
        return F16Vector._wrap(result)


def load_tile(ptr: Ptr, row: Index, col: Index, stride: Index,
              height: int, width: int, layout: Layout) -> Tile:
    tile_type = Tile._type_str(height, width, layout)
    b = IRBuilder.get()
    result = b.op("tiny_tile.load",
                  operands=[ptr._value, row._value, col._value, stride._value],
                  result_types=[tile_type])
    return Tile._wrap(result, height, width, layout)


def store_tile(tile: Tile, ptr: Ptr, row: Index, col: Index, stride: Index):
    b = IRBuilder.get()
    b.op("tiny_tile.store",
         operands=[tile._value, ptr._value, row._value,
                   col._value, stride._value])


def block_id_x() -> Index:
    b = IRBuilder.get()
    result = b.op("gpu.block_id", result_types=["index"],
                  attributes={"dimension": "#gpu<dim x>"})
    return Index._wrap(result)


def block_id_y() -> Index:
    b = IRBuilder.get()
    result = b.op("gpu.block_id", result_types=["index"],
                  attributes={"dimension": "#gpu<dim y>"})
    return Index._wrap(result)


def block_id_z() -> Index:
    b = IRBuilder.get()
    result = b.op("gpu.block_id", result_types=["index"],
                  attributes={"dimension": "#gpu<dim z>"})
    return Index._wrap(result)


# --- Convenience functions ---

def _get_type_map():
    return {
        Ptr: ("!tiny.ptr", Ptr),
        Index: ("index", Index),
    }


def print_ir(fn):
    """Print generated TinyTile dialect IR."""
    opt = TutorialOpt()
    with MLIRModule() as m:
        tile_ir = m.build_func_verified(fn, _get_type_map(), opt)
        print(tile_ir)
    return fn


def compile_and_print(fn):
    """Compile and print all lowering stages for ch3."""
    opt = TutorialOpt()

    with MLIRModule() as m:
        tile_ir = m.build_func_verified(fn, _get_type_map(), opt)

    print("=== TinyTile Dialect ===")
    print(tile_ir)

    scf_ir = opt.run(tile_ir, ["tiny-loop-to-scf", "canonicalize", "cse"])
    print("=== After tiny-loop-to-scf ===")
    print(scf_ir)

    tiny_ir = opt.run(scf_ir, ["tiny-tile-to-tiny", "canonicalize", "cse"])
    print("=== After tiny-tile-to-tiny ===")
    print(tiny_ir)

    arith_ir = opt.run(tiny_ir, ["tiny-to-arith", "canonicalize", "cse"])
    print("=== After tiny-to-arith ===")
    print(arith_ir)

    return fn
