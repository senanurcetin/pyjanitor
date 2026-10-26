# PyJanitor #1822 — Notebook 3 independent QA handoff

Date: 2026-10-27
Upstream PR: https://github.com/pyjanitor-devs/pyjanitor/pull/1822
Tested head: `f79e47cc7c533339a7c9a9270f450f77e0252da1`
Isolated QA branch: `qa/pyjanitor-1822-dtype-boundary-20261027`
**Notebook 1/2 active PR branch was not edited.** No upstream CI was triggered.

## Reproduction (Windows)

The machine has Python 3.11, pandas 2.2.3, NumPy 2.3.5 and pytest 9.1.1.
The checked-out pyjanitor package itself cannot be imported under this pandas
version because it requires `pandas.core.col`. Rather than claiming full
integration coverage, the independent loader compiles the *unchanged*
top-level function bodies directly from `janitor/functions/pivot.py` via AST
and provides the minimal local check/index-conversion utilities; it does not
run the package's external decorators, general column selectors or optional
package dependencies.

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -m pytest -q -x -o addopts= qa_independent/test_dtype_boundary_1822.py
```

The plugin autoload override is important for this specific QA host: a stale
Hypothesis pytest plugin installation references missing `hypothesis._native`.
It has no effect on the code being tested.

Observed result: **122 passed** (2.43 seconds in the final run).
`python -m compileall -q qa_independent` also passed.
The Ruff executable was not on PATH, but the global Python installation
provided `python -m ruff`. After formatting, both `ruff check` and
`ruff format --check` passed for both QA Python files; the 122-case suite
was rerun and passed (2.69 seconds). Full pyjanitor package tests were not run in this environment.

## Precisely what the 122 cases assert

- **96 differential tests**: 8 dtype families (`float16`, `complex128`,
  `uint64` with values above `2**63`, mixed-scalar `object`,
  `datetime64[ns]`, `timedelta64[ns]`, `bool`, nullable `Int64`) crossed
  with optional `part` grouping, both output orders and 0/1/5 input rows.
  Each test compares the real #1822 implementation against the legacy code
  path obtained by temporarily disabling only `_fast_numpy_value_matrix`,
  with `pandas.testing.assert_frame_equal`, and asserts the input was unchanged.
- **14 mutation/isolation checks**: 7 value families x with/without an
  additional spec column; writing into actual returned DataFrame values must
  not mutate the original input values.
- **10 fast-path selection checks**: 5 homogeneous NumPy dtype families x
  both spec layouts. The test asserts two real `_fast_ravel_group` calls
  on each balanced 4-column input; it is not satisfied by legacy fallback.
- **2 extension fallback checks**: nullable `Int64` columns must not use
  the NumPy fast group path.

A test-fixture correction explicitly constructs heterogeneous scalar
`object` columns *as object dtype*: otherwise pandas infers mixed physical
column dtypes and bypasses the optimization, creating a false assurance.
Datetime and timedelta columns use pandas-backed array semantics in this
host and are differential-checked without insisting they hit the NumPy fast
path. This is an expected distinction, not a confirmed production defect.

## Recommendation for Notebook 1/2

1. Preserve #1822's existing head and green upstream workflows while awaiting
   maintainer response.
2. Keep this QA suite on the separate branch as provenance. If desired,
   translate the strongest cases into normal `tests/functions/` tests under
   the upstream supported pandas environments (rather than cherry-picking
   this AST-only loader unchanged).
3. Avoid no-op CI runs, force pushing, or claiming local package-level test
   results from the isolated loader.
4. Do not add another upstream PR unless a substantive maintainer-requested
   change or a verified bug arises.

No functional discrepancy between optimized and baseline code was detected
by the 122 selected isolated cases; this does not prove equivalence outside
the tested inputs.
