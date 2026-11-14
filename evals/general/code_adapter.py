"""Declared pre-full-run adapter: body or one complete matching function.

No execution: parse AST only. Preserve raw answers separately. This is an
instruction-model input adapter to standard HumanEval tests, not a new test set.
Reject helpers, decorators, renamed parameters and module-level side effects.
"""
import ast
import textwrap


def function_body(problem, raw):
    if len(raw)>20000:return raw,'raw_over_adapter_cap'
    source=raw.strip()
    if source.startswith('```') and source.endswith('```'):
        source='\n'.join(source.splitlines()[1:-1])
    try:module=ast.parse(source)
    except SyntaxError:return raw,'original_completion'
    if len(module.body)!=1 or not isinstance(module.body[0],ast.FunctionDef):
        return raw,'original_completion'
    function=module.body[0]
    expected=ast.parse(problem['prompt']+'\n    pass\n').body[-1]
    if not isinstance(expected,ast.FunctionDef):raise ValueError('reference prompt is not a function')
    def shape(args):
        return ([a.arg for a in args.posonlyargs],[a.arg for a in args.args],
                args.vararg.arg if args.vararg else None,[a.arg for a in args.kwonlyargs],
                args.kwarg.arg if args.kwarg else None,
                [ast.dump(v) for v in args.defaults],
                [ast.dump(v) if v else None for v in args.kw_defaults])
    if (function.name!=problem['entry_point'] or function.decorator_list or
        shape(function.args)!=shape(expected.args)):
        return raw,'original_completion'
    # AST unparse preserves literal values, unlike textual dedenting of strings.
    body=textwrap.indent('\n'.join(ast.unparse(node) for node in function.body),'    ')+'\n'
    ast.parse(problem['prompt']+body)  # syntax gate, never exec/compile-and-run
    return body,'single_matching_function_ast_body_v1'
