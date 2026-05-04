"""DSL for TinyLoop dialect (Chapter 2)."""

from ..ch1.dsl import Index, F16Vector, Ptr
from ..compiler import IRBuilder, SSAValue, MLIRModule, TutorialOpt


def _wrap_value(value: SSAValue, template):
    """Wrap SSA value using the same type as template."""
    try:
        return type(template)._wrap(value, template=template)
    except TypeError:
        return type(template)._wrap(value)


def accumulate(bound: Index, step: Index, inits: list = None):
    """Create a tiny_loop.accumulate loop.

    Usage:
        @accumulate(n, step)
        def loop_body(i: Index):
            ...

        @accumulate(n, step, inits=[acc])
        def loop_body(i: Index, acc: F16Vector):
            ...
            return new_acc
        result = loop_body  # captures the result
    """
    inits = inits or []

    def decorator(body_fn):
        b = IRBuilder.get()

        init_values = [v._value for v in inits]
        init_types = [v._value.type_str for v in inits]
        result_types = init_types[:]

        # Allocate result SSA values
        if len(result_types) == 0:
            results = []
            result_prefix = ""
        elif len(result_types) == 1:
            r = b.new_value(result_types[0])
            results = [r]
            result_prefix = f"{r.name} = "
        else:
            base = b._counter
            b._counter += 1
            results = [SSAValue(f"%{base}#{i}", t)
                       for i, t in enumerate(result_types)]
            result_prefix = f"%{base}:{len(results)} = "

        all_operands = [bound._value, step._value] + init_values
        ops_str = ", ".join(v.name for v in all_operands)
        in_types = ", ".join(v.type_str for v in all_operands)
        if len(result_types) == 1:
            out_types = result_types[0]
        else:
            out_types = "(" + ", ".join(result_types) + ")"

        # Opening line
        b.emit(f'{result_prefix}"tiny_loop.accumulate"({ops_str}) ({{')

        # Block header with arguments
        block_arg_types = ["index"] + init_types
        block_args = [b.new_value(t) for t in block_arg_types]
        args_str = ", ".join(f"{a.name}: {a.type_str}" for a in block_args)
        b.emit(f"^bb0({args_str}):")

        # Body (indented)
        b._indent += 1

        iv = Index._wrap(block_args[0])
        iter_args = [_wrap_value(block_args[i + 1], inits[i])
                     for i in range(len(inits))]

        if inits:
            body_results = body_fn(iv, *iter_args)
            if not isinstance(body_results, (list, tuple)):
                body_results = [body_results]
            yield_values = [r._value for r in body_results]
        else:
            body_fn(iv)
            yield_values = []

        # Yield terminator
        yield_ops = ", ".join(v.name for v in yield_values)
        yield_types = ", ".join(v.type_str for v in yield_values)
        b.emit(f'"tiny_loop.yield"({yield_ops}) : ({yield_types}) -> ()')

        b._indent -= 1

        # Closing line
        b.emit(f'}}) : ({in_types}) -> {out_types}')

        if results:
            return [_wrap_value(results[i], inits[i])
                    for i in range(len(results))]
        return None

    return decorator


# --- Convenience functions ---

def _get_type_map():
    return {
        Ptr: ("!tiny.ptr", Ptr),
        Index: ("index", Index),
    }


def print_ir(fn):
    """Print generated Tiny+TinyLoop dialect IR."""
    opt = TutorialOpt()
    with MLIRModule() as m:
        tiny_ir = m.build_func_verified(fn, _get_type_map(), opt)
        print(tiny_ir)
    return fn


def compile_and_print(fn):
    """Compile and print all lowering stages for ch2."""
    opt = TutorialOpt()

    with MLIRModule() as m:
        tiny_ir = m.build_func_verified(fn, _get_type_map(), opt)

    print("=== Tiny + TinyLoop Dialect ===")
    print(tiny_ir)

    scf_ir = opt.run(tiny_ir, ["tiny-loop-to-scf"])
    print("=== After tiny-loop-to-scf ===")
    print(scf_ir)

    arith_ir = opt.run(scf_ir, ["tiny-to-arith", "canonicalize", "cse"])
    print("=== After tiny-to-arith ===")
    print(arith_ir)

    llvm_ir = opt.run(arith_ir, ["tiny-to-llvm", "convert-scf-to-cf",
                                  "convert-to-llvm"])
    print("=== LLVM Dialect ===")
    print(llvm_ir)

    return fn
