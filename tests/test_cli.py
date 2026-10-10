"""Contract tests for the ``main.py`` command line entry point.

**Tier T0** — stdlib + pytest only (architecture.md §10.1), and the file is
black-box on purpose: it drives ``main.py`` through :mod:`subprocess` and reads
it with :mod:`ast` rather than importing it.  An entry point whose contract *is*
its process exit code cannot be tested by calling functions in-process — the
contract is the thing the harness, a shell script and CI observe.

What is enforced, and why each one exists:

* the **exit-code contract** (architecture.md §4.9): ``0`` for help and for a
  reported-and-ignored unknown argument, ``1`` for a real failure, ``2`` for a
  usage error.  Two of the harness's own probes are among these cases, so a
  regression here is not theoretical;
* **the CLI can still report an error.**
  ``test_doctor_fails_without_the_scientific_stack`` and
  ``test_serve_reports_a_missing_streamlit_instead_of_crashing`` assert a
  *non-zero* exit with an
  actionable message.  The stub this module replaces returned ``0`` for
  everything, including invalid input, and ``test_a_no_op_stub_cannot_pass_these_gates``
  proves those two tests would have failed against it — a gate that cannot fail
  is the failure mode ADR-010 warns about;
* **it runs on a bare interpreter.**  ``python -S main.py …`` runs with no
  site-packages at all, which is the cheapest faithful stand-in for a venv that
  has only pytest in it (the manual bare-venv run is recorded in the wave
  report).  ``--help``, ``version`` and ``info`` must still answer, because a
  diagnostic tool that cannot start in a broken environment is useless exactly
  when it is needed;
* **it imports nothing beyond stdlib and ``core``** — at any scope, asserted by
  AST rather than by import order, because that is what keeps the previous point
  true.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "main.py"

#: ``-S`` skips ``site``, so the child sees the standard library and the script's
#: own directory and nothing else — the programmatic equivalent of a venv holding
#: only stdlib.  Used as ``bare=True`` below.
BARE = ("-S",)

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2

#: Commands that need nothing beyond the standard library.  ``operators`` needs
#: numpy (via ``core.perturbations``) and ``serve`` would launch a server, so
#: neither is exercised as an "exits zero" case.
STDLIB_ONLY_COMMANDS = ("help", "version", "info")


def _run(script: Path, *args: str, bare: bool = False) -> subprocess.CompletedProcess[str]:
    """Run ``script`` with ``args`` and capture its streams and exit code.

    Args:
        script: The entry point to execute.
        args: Arguments after the program name.
        bare: Run with ``-S`` so no site-packages are visible.
    """
    argv = [sys.executable, *(BARE if bare else ()), str(script), *args]
    return subprocess.run(
        argv,
        cwd=str(script.parent),
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )


def _cli(*args: str, bare: bool = False) -> subprocess.CompletedProcess[str]:
    """Run the repository's ``main.py``."""
    assert MAIN.is_file(), f"missing entry point: {MAIN}"
    return _run(MAIN, *args, bare=bare)


def _pins() -> dict[str, str]:
    """Parse ``requirements.txt`` the same way ``main.py version`` must."""
    found: dict[str, str] = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        entry = line.split("#", 1)[0].strip()
        if "==" not in entry or ";" in entry:
            continue
        name, _, version = entry.partition("==")
        found[name.strip()] = version.strip()
    return found


def _declared_commands() -> tuple[str, ...]:
    """Read ``main.py``'s ``COMMANDS`` keys with :mod:`ast`, without importing it.

    Keeps this file's imports stdlib-only while still asserting the help text and
    the command table cannot drift apart.
    """
    tree = ast.parse(MAIN.read_text(encoding="utf-8"), filename=str(MAIN))
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "COMMANDS"
            and isinstance(node.value, ast.Dict)
        ):
            return tuple(
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            )
    raise AssertionError("main.py declares no COMMANDS mapping")


# ── the exit-code contract (§4.9) ──────────────────────────────────────────────


def test_help_exits_zero_with_usage_on_stdout() -> None:
    """``--help`` answers on stdout with exit 0 and says nothing on stderr."""
    proc = _cli("--help")
    assert proc.returncode == EXIT_OK, proc.stderr
    assert "usage:" in proc.stdout
    assert "main.py" in proc.stdout
    assert proc.stderr == "", f"help wrote to stderr: {proc.stderr!r}"


@pytest.mark.parametrize("flag", ["-h", "help"])
def test_every_help_spelling_exits_zero(flag: str) -> None:
    """``-h``, ``--help`` and the ``help`` subcommand are the same answer."""
    proc = _cli(flag)
    assert proc.returncode == EXIT_OK, proc.stderr
    assert "usage:" in proc.stdout


def test_no_arguments_prints_help_and_exits_zero() -> None:
    """A bare invocation is a question, not an error."""
    proc = _cli()
    assert proc.returncode == EXIT_OK, proc.stderr
    assert "usage:" in proc.stdout


def test_unrecognised_option_exits_zero_and_names_it() -> None:
    """An unknown argument is *reported and ignored* by default, not swallowed.

    The three assertions together are what the historical stub could not
    satisfy: it exited 0, so only the exit code would have matched.
    """
    proc = _cli("--non-existent-option")
    assert proc.returncode == EXIT_OK, proc.stderr
    assert "--non-existent-option" in proc.stdout, (
        f"the unrecognised argument was not named: {proc.stdout!r}"
    )
    assert "usage:" in proc.stdout, "the answer did not include how to use the CLI"
    assert proc.stderr == "", f"unrecognised argument wrote to stderr: {proc.stderr!r}"


def test_strict_mode_makes_unrecognised_arguments_fatal() -> None:
    """``--strict`` restores the conventional exit 2 + usage on stderr.

    Without this the permissive default would be the *only* policy, and an entry
    point that can only ever answer ``0`` could not report a usage error.
    """
    proc = _cli("--strict", "--non-existent-option")
    assert proc.returncode == EXIT_USAGE, proc.stdout
    assert "--non-existent-option" in proc.stderr
    assert "usage:" in proc.stderr
    assert proc.stdout == "", f"strict usage errors belong on stderr: {proc.stdout!r}"


@pytest.mark.parametrize("args", [("doctor", "extra"), ("doctor", "--port", "99999")])
def test_usage_error_inside_a_recognised_command_is_fatal(args: tuple[str, ...]) -> None:
    """Once a command is recognised, its own argument grammar is enforced."""
    proc = _cli(*args)
    assert proc.returncode == EXIT_USAGE, proc.stdout
    assert "usage:" in proc.stderr


def test_strict_flag_is_honoured_after_a_recognised_command() -> None:
    """``--strict`` is recognised in any position, not only as the first token."""
    assert _cli("--strict", "doctor", "extra").returncode == EXIT_USAGE
    assert _cli("doctor", "--strict", "extra").returncode == EXIT_USAGE


# ── the CLI does real work ────────────────────────────────────────────────────


@pytest.mark.parametrize("command", STDLIB_ONLY_COMMANDS)
def test_stdlib_only_commands_exit_zero(command: str) -> None:
    """Every stdlib-only command answers successfully with a bare interpreter."""
    proc = _cli(command, bare=True)
    assert proc.returncode == EXIT_OK, proc.stderr
    assert proc.stdout.strip(), f"{command} printed nothing"


def test_version_reports_every_pinned_distribution() -> None:
    """``version`` must not silently omit a dependency from the lockfile.

    Asserted against ``requirements.txt`` rather than against a hand-written
    list, so adding a pin without reporting it fails here.
    """
    proc = _cli("version")
    assert proc.returncode == EXIT_OK, proc.stderr
    pins = _pins()
    assert pins, "requirements.txt yielded no pins; the parser is broken"
    omitted = [name for name in pins if name not in proc.stdout]
    assert not omitted, f"version omitted pinned distributions: {omitted}"


def test_info_reports_the_resolved_configuration() -> None:
    """``info`` prints the bounds table and the artifact policy it resolved."""
    proc = _cli("info", bare=True)
    assert proc.returncode == EXIT_OK, proc.stderr
    for expected in ("StressBounds", "ArtifactPolicy", "max_size_bytes", "noise_level"):
        assert expected in proc.stdout, f"info did not report {expected}"


def test_info_marks_an_environment_override() -> None:
    """A ``STF_*`` override must be visible, or the report is misleading."""
    proc = subprocess.run(
        [sys.executable, "-S", str(MAIN), "info"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
        env={**os.environ, "STF_LOG_LEVEL": "DEBUG"},
    )
    assert proc.returncode == EXIT_OK, proc.stderr
    assert "DEBUG" in proc.stdout
    assert "STF_LOG_LEVEL" in proc.stdout, "the overridden variable was not named"


def test_operators_lists_the_registry_and_translates_labels() -> None:
    """``operators`` proves both the registry keys and ``operation_key`` work."""
    if not _has_numpy():
        pytest.skip("core.perturbations needs numpy")
    proc = _cli("operators")
    assert proc.returncode == EXIT_OK, proc.stderr
    for key in (
        "gaussian_noise",
        "uniform_noise",
        "feature_dropout",
        "feature_corruption",
        "scale_perturbation",
        "distribution_shift",
    ):
        assert key in proc.stdout, f"registry key {key} not reported"
    assert "Gaussian Noise" in proc.stdout, "the UI label -> key mapping was not shown"


def test_every_declared_command_appears_in_the_help_text() -> None:
    """``COMMANDS`` and ``--help`` cannot describe different command sets."""
    declared = _declared_commands()
    assert len(declared) >= 5, f"COMMANDS is implausibly small: {declared}"
    proc = _cli("--help")
    omitted = [name for name in declared if name not in proc.stdout]
    assert not omitted, f"declared but undocumented: {omitted}"


# ── the CLI can report a failure ──────────────────────────────────────────────


def test_doctor_fails_without_the_scientific_stack() -> None:
    """A genuinely broken environment produces exit 1 and a remedy.

    This is the assertion that makes the permissive default safe: "exit 0"
    means "the work was done", so something must be able to make it not-0.
    """
    proc = _cli("doctor", bare=True)
    assert proc.returncode == EXIT_FAILURE, (
        f"doctor did not fail in a stackless environment: {proc.stdout}"
    )
    assert "[FAIL]" in proc.stdout
    assert "pip install -r requirements.txt" in proc.stdout, "no remedy was given"


def test_serve_reports_a_missing_streamlit_instead_of_crashing() -> None:
    """The launcher's pre-flight is a message and an exit code, not a traceback."""
    proc = _cli("serve", bare=True)
    assert proc.returncode == EXIT_FAILURE, proc.stdout
    assert "streamlit is not installed" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_operators_reports_a_missing_numeric_stack(tmp_path: Path) -> None:
    """The one command that needs numpy says so, with the install command.

    Copies ``main.py`` alone so the missing ``core`` package cannot be the reason
    the import fails — otherwise this would pass for the wrong reason.
    """
    script = tmp_path / "main.py"
    script.write_text(MAIN.read_text(encoding="utf-8"), encoding="utf-8")
    proc = _run(script, "operators", bare=True)
    assert proc.returncode == EXIT_FAILURE, proc.stdout
    assert "numeric stack" in proc.stderr
    assert "pip install -r requirements.txt" in proc.stderr
    assert "Traceback" not in proc.stderr


# ── layering: stdlib + core only, at any scope ────────────────────────────────


def test_cli_imports_only_stdlib_and_core() -> None:
    """No third-party import at module scope *or* inside a function.

    §2.1 holds a root-level module to stdlib and project packages; this asserts
    the tighter rule the CLI is actually designed to (stdlib + ``core``), and the
    AST walk covers nested and function-local imports, which is where a lazy
    ``import numpy`` would otherwise hide.
    """
    tree = ast.parse(MAIN.read_text(encoding="utf-8"), filename=str(MAIN))
    offenders: list[str] = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        for name in names:
            root = name.split(".")[0]
            if not root or root == "__future__":
                continue
            if root in sys.stdlib_module_names or root == "core":
                continue
            offenders.append(f"line {node.lineno}: {name}")
    assert not offenders, "main.py may import stdlib and core only; found:\n  " + "\n  ".join(
        sorted(offenders)
    )


def test_this_module_is_tier_t0() -> None:
    """Guard the guard: this file must stay stdlib + pytest only."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"), filename=__file__)
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        for name in names:
            root = name.split(".")[0]
            assert root in sys.stdlib_module_names or root in {"pytest", "__future__"}, (
                f"tests/test_cli.py is tier T0 but imports {name}"
            )


# ── the gates above are load-bearing ──────────────────────────────────────────


def test_a_no_op_stub_cannot_pass_these_gates(tmp_path: Path) -> None:
    """The stub this file's suite replaced would fail every assertion here.

    Reproduces the historical ``main.py`` — ``return 0`` for every argument,
    including invalid ones — and asserts the contract does **not** hold for it.
    Without this, "the CLI reports errors" would be an unfalsifiable claim; with
    it, the gate is proven able to fail (the same standard ADR-010 applies to the
    mutation probe in ``evidence/main_py_gate_probe.py``).
    """
    stub = tmp_path / "main.py"
    stub.write_text(
        'import sys\n\n\ndef main(argv=None):\n    """CLI stub."""\n'
        '    argv = argv or sys.argv[1:]\n    if "--help" in argv or "-h" in argv:\n'
        '        print("Stress Testing Framework")\n    return 0\n\n\n'
        'if __name__ == "__main__":\n    sys.exit(main())\n',
        encoding="utf-8",
    )

    unknown = _run(stub, "--non-existent-option")
    assert unknown.returncode == EXIT_OK  # the stub's only virtue, by accident
    assert "--non-existent-option" not in unknown.stdout, "the stub should say nothing"

    strict = _run(stub, "--strict", "--non-existent-option")
    assert strict.returncode != EXIT_USAGE, (
        "the stub exited 0 where --strict must exit 2: "
        "test_strict_mode_makes_unrecognised_arguments_fatal would not catch it"
    )

    doctor = _run(stub, "doctor", bare=True)
    assert doctor.returncode != EXIT_FAILURE, (
        "the stub exited 0 where doctor must exit 1 in a stackless environment: "
        "test_doctor_fails_without_the_scientific_stack would not catch it"
    )

    proc = _cli("--non-existent-option")
    assert "--non-existent-option" in proc.stdout, (
        "sanity: the real CLI still names the unrecognised argument"
    )


# ── small helpers ─────────────────────────────────────────────────────────────


def _has_numpy() -> bool:
    """Whether the numeric stack is importable, without importing it."""
    return importlib.util.find_spec("numpy") is not None
