import sys
import ast

# Python 3.12 removed ast.Num/Str/NameConstant/Ellipsis in favour of ast.Constant.
# Triton 3.5.x code_generator.py still calls ast.Num(0)/ast.Num(1) when JIT-compiling
# kernels that contain range() loops. Restore the aliases so that Triton compiles
# successfully on Python 3.12+.
if sys.version_info >= (3, 12) and not hasattr(ast, 'Num'):
    ast.Num = ast.Constant
    ast.Str = ast.Constant
    ast.NameConstant = ast.Constant
    ast.Ellipsis = ast.Constant
