"""Independent dtype and shape differential QA for pyjanitor #1822.

No modifications to the active contribution branch. This suite compares
production fast code with the legacy path on the same git revision and checks
that modifying a returned frame cannot mutate its source DataFrame.
"""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import pivot_source_loader as pivot

DTYPES = ("float16", "complex128", "uint64", "object", "datetime64[ns]", "timedelta64[ns]", "bool", "Int64")


def make_frame(n_rows, dtype, n_cols):
    data = np.arange(n_rows * n_cols, dtype="int64").reshape((n_rows, n_cols))
    cols = [f"v{i}" for i in range(n_cols)]
    if dtype == "float16":
        arr = data.astype(dtype)
        if n_rows:
            arr[0, 0] = np.nan
    elif dtype == "complex128":
        arr = data.astype(dtype) + 1j * (data % 3)
        if n_rows:
            arr[0, 0] = complex(np.nan, 1)
    elif dtype == "uint64":
        arr = data.astype(dtype) + np.uint64(2**63)
    elif dtype == "bool":
        arr = data % 2 == 0
    elif dtype == "object":
        arr = data.astype("object")
        for col in range(n_cols):
            if col % 4 == 0:
                arr[:, col] = [f"item-{j}" for j in range(n_rows)]
            elif col % 4 == 1:
                arr[:, col] = [pd.Timestamp("2024-01-01") for _ in range(n_rows)]
            elif col % 4 == 2:
                arr[:, col] = [None for _ in range(n_rows)]
            else:
                arr[:, col] = [j + 0.5 for j in range(n_rows)]
    elif dtype == "datetime64[ns]":
        arr = (data.astype("timedelta64[D]") + np.datetime64("2024-01-01")).astype("datetime64[ns]")
    elif dtype == "timedelta64[ns]":
        arr = data.astype("timedelta64[ns]")
    elif dtype == "Int64":
        return pd.DataFrame(data, columns=cols).astype("Int64")
    else:
        raise AssertionError(dtype)
    return pd.DataFrame(arr, columns=cols, dtype="object" if dtype == "object" else None)


def spec_for(columns, with_other):
    num = len(columns)
    if with_other:
        # Deliberately uneven groups, but every (value, part) is unique.
        pairs = [("a", "q"), ("b", "q"), ("a", "r"), ("c", "q"),
                 ("b", "r"), ("a", "s"), ("c", "r"), ("b", "s")]
        spec = pd.DataFrame({".name": columns,
                             ".value": [p[0] for p in pairs[:num]],
                             "part": [p[1] for p in pairs[:num]]})
    else:
        groups = ["a", "b", "a", "c", "b", "a", "c", "b"]
        spec = pd.DataFrame({".name": columns, ".value": groups[:num]})
    return spec


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("with_other", [False, True])
@pytest.mark.parametrize("sort_by_appearance", [False, True])
@pytest.mark.parametrize("rows", [0, 1, 5])
def test_numpy_dtype_boundaries_match_legacy(monkeypatch, dtype, with_other, sort_by_appearance, rows):
    columns = 5 if with_other else 7
    frame = make_frame(rows, dtype, columns)
    before = frame.copy(deep=True)
    spec = spec_for(list(frame.columns), with_other)
    options = {"sort_by_appearance": sort_by_appearance,
               "dropna": rows % 2 == 0, "ignore_index": rows != 1}

    with monkeypatch.context() as patched:
        patched.setattr(pivot, "_fast_numpy_value_matrix", lambda df: None)
        legacy = pivot.pivot_longer_spec(frame.copy(deep=True), spec.copy(), **options)
    actual = pivot.pivot_longer_spec(frame, spec.copy(), **options)
    assert_frame_equal(actual, legacy)
    assert_frame_equal(frame, before)


@pytest.mark.parametrize("dtype", ["float16", "complex128", "uint64", "object", "datetime64[ns]", "timedelta64[ns]", "bool"])
@pytest.mark.parametrize("with_other", [False, True])
def test_output_mutation_never_alters_input(dtype, with_other):
    frame = make_frame(3, dtype, 5 if with_other else 7)
    source = frame.copy(deep=True)
    spec = spec_for(list(frame.columns), with_other)
    output = pivot.pivot_longer_spec(frame, spec, sort_by_appearance=True)
    assert not output.empty
    candidates = [key for key in ("a", "b", "c") if key in output.columns]
    assert candidates
    col = output.columns.get_loc(candidates[0])
    replacement = {
        "float16": -19,
        "complex128": complex(-33, 2),
        "uint64": np.uint64(2**63 + 999),
        "object": "MUTATED",
        "datetime64[ns]": pd.Timestamp("2044-10-16"),
        "timedelta64[ns]": pd.Timedelta("987 days"),
        "bool": False,
    }[dtype]
    output.iloc[0, col] = replacement
    assert_frame_equal(frame, source)


@pytest.mark.parametrize("dtype", ["float16", "complex128", "uint64", "object", "bool"])
@pytest.mark.parametrize("with_other", [False, True])
def test_balanced_numpy_frames_really_use_fast_path(monkeypatch, dtype, with_other):
    """Ensure differential tests cannot all pass due to silent legacy fallback."""
    frame = make_frame(4, dtype, 4)
    columns = list(frame.columns)
    spec = pd.DataFrame({".name": columns, ".value": ["a", "b", "a", "b"]})
    if with_other:
        spec["part"] = ["q", "q", "r", "r"]
    calls = []
    original = pivot._fast_ravel_group

    def record(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(pivot, "_fast_ravel_group", record)
        pivot.pivot_longer_spec(frame, spec)
    assert len(calls) == 2, (dtype, with_other, calls)


@pytest.mark.parametrize("with_other", [False, True])
def test_nullable_integer_extension_is_not_fast_pathed(monkeypatch, with_other):
    """Preserve pandas nullable integer and NA semantics on the legacy path."""
    frame = make_frame(4, "Int64", 4)
    spec = pd.DataFrame({".name": list(frame), ".value": ["a", "b", "a", "b"]})
    if with_other:
        spec["part"] = ["q", "q", "r", "r"]
    calls = []
    original = pivot._fast_ravel_group

    def record(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(pivot, "_fast_ravel_group", record)
        pivot.pivot_longer_spec(frame, spec)
    assert calls == []
