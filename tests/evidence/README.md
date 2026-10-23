# Tracked gate evidence (`tests/evidence/`)

These files are **the minimal, committed proof** for gates the project
documentation cites as verified. They were promoted out of the gitignored
`.forge/runs/run-003/evidence/` directory so a fresh `git clone` can reproduce
them — an unreproducible gate result is a claim, not evidence (ADR-029).

They are **not** tests: their filenames are not `test_*.py`, so pytest does not
collect them. `tests/test_architecture.py` asserts this bundle is present,
importable and `bash -n`-clean, and fails when it is deleted (ADR-029).

## Contents

| File | Proves | Run |
|---|---|---|
| `t0_bare_venv.sh` | The §10.1 **T0 tier** and the §4.9 CLI promise run on an interpreter with *nothing but* the standard library + pytest. | `bash tests/evidence/t0_bare_venv.sh` |
| `mutation_probe.py` | ADR-001's security boundary, the §2.2 root-module gates and the intra-`core/` ladder are **load-bearing** (each mutation turns the named suite subset red). | `.venv/bin/python tests/evidence/mutation_probe.py` |
| `main_py_gate_probe.py` | The §2.2 top-level-module gate cannot be satisfied by widening a constant alone, and a declared root module must still pass the third-party hygiene check (ADR-012). | `.venv/bin/python tests/evidence/main_py_gate_probe.py /tmp/stf-probe` |

Both Python probes also accept `--check-anchors` (ADR-032): a side-effect-free
self-check that every declared `(rel, old, new)` source anchor still resolves
against the live tree, exiting non-zero on any miss. The T0 gate
`test_evidence_probe_anchors_resolve_against_the_tree` runs it on every suite
run, so a production reword that moves an anchor fails the suite instead of
silently disabling a probe.

## Constraints

- **stdlib + `core.config` only.** The Python probes import nothing outside the
  standard library except `core.config`; `t0_bare_venv.sh` installs only the
  pinned `pytest`. This keeps the bundle runnable in the bare environment where
  the T0 claim matters. `--check-anchors` adds no import and performs no file
  write, so the T0 gate can invoke it on a bare interpreter.
- **No deserialisation.** The probes contain no `pickle.load` /
  `pd.read_pickle` / `numpy.load(allow_pickle=True)` / `joblib.load` pattern, so
  they stay clean under the `tests/**`-wide containment gate
  (`test_deserialization_is_confined_to_model_io`).
- **Full mutation runs write files.** `main_py_gate_probe.py` runs each scenario
  against a scratch copy of the tree and `mutation_probe.py` mutates and restores
  source files with an md5 check; both refuse to leave residue. The in-suite
  `--check-anchors` mode never does — hence its name.

See `ADR-029` in `.ai/project/decisions.md` for the promotion decision.
