# ruff: noqa: F401
# These globals are referenced by production functions compiled via AST.
"""Independent QA loader for production pivot functions.

The project requires a newer pandas build than this Windows QA host currently
provides. This AST extraction exercises the unmodified function bodies from
janitor/functions/pivot.py, with non-pivot package decorators removed. It is NOT
a substitute for upstream end-to-end pytest or CI.
"""

from __future__ import annotations

import ast
import operator
import re
import warnings
from functools import reduce
from pathlib import Path
from typing import Any, Callable, Pattern

import numpy as np
import pandas as pd
from pandas.api.types import is_extension_array_dtype
from pandas.core.dtypes.concat import concat_compat
from pandas.core.groupby.generic import DataFrameGroupBy


def check(name, value, kinds):
    if not isinstance(value, tuple(kinds)):
        raise TypeError(f"{name} should be one of {kinds}")


SOURCE = Path(__file__).resolve().parents[1] / "janitor" / "functions" / "pivot.py"
parsed = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
functions = []
for node in parsed.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno < 2096:
        node.decorator_list = []
        functions.append(node)
module = ast.Module(body=functions, type_ignores=[])
exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), globals())


def _index_converter(selector, columns):
    if isinstance(selector, slice) and selector == slice(None):
        return np.arange(len(columns))
    raise NotImplementedError("Independent QA only supports the all-columns selector")
