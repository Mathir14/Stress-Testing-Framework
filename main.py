"""Command-line entry point — diagnostics and launcher for the framework.

Architecture reference: ``architecture.md`` §2.1 (this module's dependency
row), §2.2 (declared second top-level module) and §4.9 (the CLI contract);
ADR-012 records why it exists and why its argument policy is permissive by
default.

The framework's user-facing surface is ``streamlit run app.py``.  This module
covers the three things a web UI cannot do well:

1. **report** — what configuration the process actually resolved, and what is
   installed in the environment (an environment bug is far cheaper to diagnose
   from a terminal than from a browser);
2. **answer** — a machine-readable exit code, so a script can ask the project
   something without scraping rendered HTML;
3. **launch** — the documented one-liner for the app itself, so there is a
   single entry point to remember.

It is **not** a second copy of the application and contains no domain logic:
each command either prints what :mod:`core.config` resolved, delegates to
:mod:`core.perturbations`, reads the artifact manifest, or execs ``app.py``.

Import rules (§2.1).  A root-level module belongs to no §2.1 layer, so this one
is held to the stricter of the two available rules: **stdlib and ``core`` only,
at any scope** — no third-party import appears here, not even inside a function.
That is what keeps ``--help``, ``version``, ``info`` and ``doctor`` runnable on
a bare interpreter, which is precisely where a "your environment is broken" tool
is most needed.  Heavier dependencies are reached *by name* through
:func:`importlib.util.find_spec` or in a child process, never by importing them.

Exit-code contract (§4.9)
-------------------------

===== =========================================================================
Code  Meaning
===== =========================================================================
``0`` The requested work was done, **or** the argument was not recognised and
      was reported and ignored.  Unknown arguments are non-fatal by default so
      that a probe of the entry point gets an answer instead of a traceback.
``1`` The requested work failed: a missing dependency, an unreadable manifest,
      an unreachable version pin.  Every such message names the remedy.
``2`` Usage error *inside a recognised command* (argparse), or any unrecognised
      argument when ``--strict`` is passed.
===== =========================================================================

The permissive default is a deliberate, documented choice, not an oversight: an
entry point that can only answer ``0`` cannot report anything.  ``--strict``
restores the conventional fatal treatment, and it is what CI should use.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
from collections.abc import Callable, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

__all__ = [
    "COMMANDS",
    "EXIT_FAILURE",
    "EXIT_OK",
    "EXIT_USAGE",
    "MIN_PYTHON",
    "build_parser",
    "main",
]

PROJECT_NAME = "ML Model Reliability & Stress Testing Framework"

#: Repository root, so every path below is independent of the caller's cwd.
ROOT = Path(__file__).resolve().parent

#: The interpreter floor.  Matches the devcontainer image (Python 3.11-bookworm)
#: and the ``X | None`` union syntax used across the project.
MIN_PYTHON: tuple[int, int] = (3, 11)

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2

HELP_FLAGS = frozenset({"-h", "--help"})
VERSION_FLAG = "--version"
STRICT_FLAG = "--strict"

#: One-line summaries, kept as data so the parser and the help epilog cannot
#: describe different command sets.
_COMMAND_HELP: dict[str, str] = {
    "help": "Print this help text (same as --help).",
    "version": "Report the interpreter, platform and pinned dependency versions.",
    "info": "Print the resolved AppConfig: log level, seed, bounds, weights, artifact policy.",
    "operators": "List the perturbation registry keys and the UI labels that map to them.",
    "doctor": "Check the environment: interpreter, version pins, venv, artifact root, manifest.",
    "serve": "Launch the Streamlit app (equivalent to `streamlit run app.py`).",
}

#: Rendered by ``doctor``; the parent status decides whether they are failures.
_OK, _WARN, _FAIL = "ok", "warn", "fail"


# ── helpers ───────────────────────────────────────────────────────────────────


def _ensure_importable() -> None:
    """Put the repository root on ``sys.path`` so ``core`` resolves from any cwd.

    ``python main.py`` already puts the script's directory first on the path;
    this covers ``python /abs/path/main.py`` and importing the module from a
    test, and costs nothing when the entry is already present.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _pins() -> dict[str, str]:
    """Parse ``requirements.txt`` into ``{distribution: pinned version}``.

    Only the fully pinned ``name==version`` form is honoured; comments, blank
    lines, markers and unpinned requirements are skipped, because the file is a
    lockfile (ADR-007) and anything looser would make the report misleading.
    """
    path = ROOT / "requirements.txt"
    found: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return found
    for line in text.splitlines():
        entry = line.split("#", 1)[0].strip()
        if "==" not in entry or ";" in entry:
            continue
        name, _, version = entry.partition("==")
        name = name.strip()
        if name:
            found[name] = version.strip()
    return found


def _installed_version(distribution: str) -> str | None:
    """Return an installed distribution's version, or ``None`` if absent.

    Uses :mod:`importlib.metadata` (stdlib) rather than importing the package:
    a missing dependency must be *reportable*, so the check that detects it
    cannot depend on it (ADR-010.1's reasoning, applied to the doctor command).
    """
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _relative_to_root(path: Path) -> Path:
    """Resolve a possibly-relative configured path against the repository root.

    ``ArtifactPolicy.root_dir`` defaults to the relative ``saved_models``; a
    report that depends on the caller's cwd would be a report about the wrong
    directory.
    """
    return path if path.is_absolute() else ROOT / path


# ── commands ──────────────────────────────────────────────────────────────────


def _cmd_help(_args: argparse.Namespace) -> int:
    """Print the top-level help text."""
    build_parser().print_help()
    return EXIT_OK


def _cmd_version(_args: argparse.Namespace) -> int:
    """Report the interpreter and every pinned dependency's installed version.

    Always exits ``0``: reporting is the work, and a version mismatch is
    :command:`doctor`'s finding to fail on, not this command's.
    """
    print(PROJECT_NAME)
    print(f"  python      {platform.python_version()} ({platform.python_implementation()})")
    print(f"  platform    {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  root        {ROOT}")

    pins = _pins()
    print()
    print(f"pinned runtime dependencies ({len(pins)} in requirements.txt):")
    width = max((len(name) for name in pins), default=0)
    missing = differing = 0
    for name, pin in pins.items():
        installed = _installed_version(name)
        if installed is None:
            state, missing = "MISSING", missing + 1
        elif installed != pin:
            state, differing = "DIFFERS", differing + 1
        else:
            state = "ok"
        shown = installed or "<not installed>"
        print(f"  {name:<{width}}  pin {pin:<14} {shown:<14} {state}")
    if missing or differing:
        print()
        print(
            f"{missing} missing, {differing} differing from the lockfile — "
            "run `python main.py doctor` for the full verdict, or "
            "`pip install -r requirements.txt` to reconcile."
        )
    return EXIT_OK


def _cmd_info(_args: argparse.Namespace) -> int:
    """Print the configuration the process actually resolved.

    Reads the zero-argument :func:`core.config.get_config` singleton, so the
    values shown include any ``STF_*`` environment overrides — the whole point
    of the command is to show the *resolved* values rather than the defaults.

    Raises:
        Nothing.  A missing numeric stack is not required for this command: the
        configuration dataclasses are stdlib-only by construction (§2.3, rung 1).
    """
    _ensure_importable()
    # Function-local by design (§2.1): importing core at module scope would
    # make `--help` depend on the project package being importable.
    from core.config import (
        ENV_LOG_LEVEL,
        ENV_RANDOM_SEED,
        FAILING_GRADE,
        get_config,
    )

    config = get_config()
    print("resolved configuration (core.config.get_config())")
    print(f"  log_level     {config.log_level}" + _env_hint(ENV_LOG_LEVEL))
    seed = (
        "unseeded (fresh np.random.default_rng per script run)"
        if config.random_seed is None
        else str(config.random_seed)
    )
    print(f"  random_seed   {seed}" + _env_hint(ENV_RANDOM_SEED))

    print()
    print("stress bounds (core.config.StressBounds)")
    for name, (low, high) in _pairs(config.stress_bounds):
        print(f"  {name:<18} [{low}, {high}]")

    print()
    print("reliability policy (core.config.ReliabilityWeights)")
    for name in ("max_perf", "max_cal", "max_rob", "max_conf", "ece_cap", "drop_cap"):
        print(f"  {name:<18} {getattr(config.reliability, name)}")
    print(
        "  grade cutoffs    "
        + ", ".join(f"{bound} -> {grade}" for bound, grade in config.reliability.grade_cutoffs)
    )
    print(
        f"  fallthrough      below the last cutoff -> {FAILING_GRADE!r} "
        f"({config.reliability.colour_for(FAILING_GRADE)})"
    )

    policy = config.artifacts
    print()
    print("artifact policy (core.config.ArtifactPolicy)")
    print(f"  root_dir            {_relative_to_root(policy.root_dir)}")
    print(f"  filename_pattern    {policy.filename_pattern!r}")
    print(f"  max_filename_length {policy.max_filename_length}")
    print(f"  max_size_bytes      {policy.max_size_bytes}")
    print(f"  require_attestation {policy.require_attestation}")
    return EXIT_OK


def _env_hint(name: str) -> str:
    """Annotate a value with the environment variable that can override it."""
    return f"   [override: {name}]" if name in os.environ else ""


def _pairs(bounds: Any) -> list[tuple[str, tuple[float, float]]]:
    """Return a bounds dataclass as an ordered ``name -> (low, high)`` list."""
    return [(name, getattr(bounds, name)) for name in bounds.__dataclass_fields__]


def _cmd_operators(_args: argparse.Namespace) -> int:
    """List the perturbation registry and prove the label translation works.

    This is the one command that needs the numeric stack (``core.perturbations``
    imports numpy), which makes it the honest place to demonstrate the non-zero
    exit path: a trimmed environment gets an actionable message and exit ``1``.
    """
    _ensure_importable()
    try:
        from core.perturbations import (
            OPERATION_LABELS,
            available_operations,
            operation_key,
        )
    except ImportError as exc:  # ModuleNotFoundError is an ImportError
        print(
            f"error: the perturbation registry needs the numeric stack "
            f"({exc.name} is not installed).",
            file=sys.stderr,
        )
        print("hint: pip install -r requirements.txt", file=sys.stderr)
        return EXIT_FAILURE

    keys = available_operations()
    print(f"registry keys ({len(keys)}) — the strings batch_stress_test dispatches on")
    for key in keys:
        labels = sorted(label for label, target in OPERATION_LABELS.items() if target == key)
        print(f"  {key:<20} {', '.join(labels) if labels else '(no UI label)'}")

    print()
    print(f"UI labels ({len(OPERATION_LABELS)}) — operation_key() translation")
    for label in sorted(OPERATION_LABELS):
        print(f"  {label:<20} -> {operation_key(label)}")
    return EXIT_OK


def _cmd_doctor(_args: argparse.Namespace) -> int:
    """Check whether this checkout can run, and say what to do about it if not.

    Returns:
        ``1`` if any check failed, ``0`` otherwise.  Warnings (a missing venv,
        an absent manifest) are reported but do not fail the command — the
        artifact directory is legitimately created on first save.
    """
    _ensure_importable()
    results: list[tuple[str, str, str]] = []

    def record(status: str, label: str, detail: str) -> None:
        results.append((status, label, detail))

    floor = ".".join(str(part) for part in MIN_PYTHON)
    running = ".".join(str(part) for part in sys.version_info[:2])
    record(
        _OK if sys.version_info >= MIN_PYTHON else _FAIL,
        f"interpreter >= {floor}",
        f"{platform.python_implementation()} {running}",
    )

    record(_OK if (ROOT / "app.py").is_file() else _FAIL, "app.py present", str(ROOT / "app.py"))

    venv = ROOT / ".venv"
    record(
        _OK if venv.is_dir() else _WARN,
        "virtual environment at .venv",
        "present" if venv.is_dir() else "absent — the interpreter in use may be the system one",
    )

    pins = _pins()
    missing = sorted(name for name in pins if _installed_version(name) is None)
    differing = sorted(
        name
        for name, pin in pins.items()
        if (found := _installed_version(name)) is not None and found != pin
    )
    if missing or differing:
        detail = f"{len(missing)} missing, {len(differing)} differing"
        if missing:
            detail += f"; missing e.g. {', '.join(missing[:3])}"
        record(_FAIL, f"runtime pins satisfied ({len(pins)})", detail)
    else:
        record(_OK, f"runtime pins satisfied ({len(pins)})", "every pin matches requirements.txt")

    pytest_version = _installed_version("pytest")
    record(
        _OK if pytest_version else _WARN,
        "pytest installed",
        pytest_version or "absent — the test suite needs requirements-dev.txt",
    )

    record(*_artifact_root_status())
    record(*_manifest_status())

    width = max(len(label) for _, label, _ in results)
    failures = sum(1 for status, _, _ in results if status == _FAIL)
    warnings = sum(1 for status, _, _ in results if status == _WARN)
    print(f"doctor: {len(results)} checks, {failures} failed, {warnings} warning(s)")
    for status, label, detail in results:
        marker = {"ok": "PASS", "warn": "WARN", "fail": "FAIL"}[status]
        print(f"  [{marker}] {label:<{width}}  {detail}")
    if failures:
        print()
        print("remedy: pip install -r requirements.txt -r requirements-dev.txt")
        print("        then re-run `python main.py doctor`.")
    return EXIT_FAILURE if failures else EXIT_OK


def _artifact_root_status() -> tuple[str, str, str]:
    """Check that the configured artifact root exists and is writable."""
    label = "artifact root writable"
    try:
        root = _artifact_root()
    except ImportError as exc:  # pragma: no cover - core is stdlib-only
        return (_FAIL, label, f"cannot resolve core.config ({exc})")
    label = f"artifact root {root.name}/ writable"
    if not root.exists():
        return (_WARN, label, f"{root} does not exist yet; created on first save")
    if not root.is_dir():
        return (_FAIL, label, f"{root} exists but is not a directory")
    return (
        (_OK, label, str(root))
        if os.access(root, os.W_OK)
        else (_FAIL, label, f"{root} is not writable")
    )


def _artifact_root() -> Path:
    """The configured artifact root, resolved against the repository root.

    Raises:
        ImportError: if ``core.config`` cannot be imported.  Callers convert it
            into a failed check rather than letting it escape, so ``doctor`` can
            report a missing kernel instead of raising.
    """
    from core.config import get_config  # function-local: keeps the T0 promise

    return _relative_to_root(get_config().artifacts.root_dir)


def _manifest_status() -> tuple[str, str, str]:
    """Check that the attestation manifest, if present, parses and is current."""
    from core.model_io import MANIFEST_FILENAME, MANIFEST_FORMAT_VERSION

    label = f"{MANIFEST_FILENAME} readable"
    try:
        path = _artifact_root() / MANIFEST_FILENAME
    except ImportError:  # pragma: no cover - core is stdlib-only
        return (_FAIL, label, "cannot resolve core.config")
    if not path.exists():
        return (_WARN, label, f"absent at {path}; written by the first model save")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return (_FAIL, label, f"{path}: {exc}")
    version = payload.get("format_version")
    if version != MANIFEST_FORMAT_VERSION:
        return (
            _FAIL,
            label,
            f"{path}: format_version {version!r}, expected {MANIFEST_FORMAT_VERSION}",
        )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        return (_FAIL, label, f"{path}: 'artifacts' is {type(artifacts).__name__}, expected object")
    return (_OK, label, f"{path}: format_version {version}, {len(artifacts)} artifact(s)")


def _cmd_serve(args: argparse.Namespace) -> int:
    """Exec ``streamlit run app.py`` with the requested server settings.

    The launcher is the CLI's only side-effecting command.  ``streamlit`` is
    resolved by name rather than imported, because importing it prints warnings
    and drags in the whole scientific stack for what is an ``exec`` away.

    Returns:
        ``1`` if streamlit or ``app.py`` is missing; otherwise this call does not
        return — the child process replaces this one.
    """
    if importlib.util.find_spec("streamlit") is None:
        print("error: streamlit is not installed, so the app cannot be launched.", file=sys.stderr)
        print("hint: pip install -r requirements.txt", file=sys.stderr)
        return EXIT_FAILURE
    app = ROOT / "app.py"
    if not app.is_file():
        print(f"error: {app} does not exist.", file=sys.stderr)
        return EXIT_FAILURE
    argv = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.port",
        str(args.port),
        "--server.address",
        str(args.address),
        "--server.headless",
        "true" if args.headless else "false",
    ]
    print(f"launching: {' '.join(argv)}")
    try:
        os.execv(sys.executable, argv)
    except OSError as exc:
        print(f"error: could not launch streamlit: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    return EXIT_FAILURE  # unreachable: execv either replaces the process or raises


#: Command name -> handler.  Every entry here appears in ``--help`` and in the
#: parser's subcommand list; ``tests/test_cli.py`` asserts the two agree.
COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "help": _cmd_help,
    "version": _cmd_version,
    "info": _cmd_info,
    "operators": _cmd_operators,
    "doctor": _cmd_doctor,
    "serve": _cmd_serve,
}


# ── parser ────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``python main.py``.

    Returns:
        A parser whose subcommands are exactly :data:`COMMANDS`.  ``required``
        is deliberately left off: an empty ``argv`` is answered with the help
        text and exit ``0`` (§4.9) rather than with a usage error.
    """
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description=f"{PROJECT_NAME} — command line entry point.",
        epilog=(
            "Unrecognised arguments are reported and ignored by default, so a "
            "probe of this entry point receives an answer rather than a "
            f"traceback. Pass {STRICT_FLAG} to make them fatal (exit "
            f"{EXIT_USAGE}). CI should use {STRICT_FLAG}."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    for name, summary in _COMMAND_HELP.items():
        sub = subparsers.add_parser(name, help=summary, description=summary)
        sub.set_defaults(handler=COMMANDS[name])
        if name == "serve":
            sub.add_argument(
                "--port",
                type=_port,
                default=8501,
                help="TCP port for the Streamlit server (default: %(default)s).",
            )
            sub.add_argument(
                "--address",
                default="localhost",
                help="Bind address for the Streamlit server (default: %(default)s).",
            )
            sub.add_argument(
                "--headless",
                action="store_true",
                help="Do not open a browser (useful over SSH or in CI).",
            )
    return parser


def _port(value: str) -> int:
    """argparse type for a TCP port: a number in the unprivileged+ range."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from None
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError(f"{number} is outside 1-65535")
    return number


def _report_unrecognised(tokens: Sequence[str], strict: bool) -> int:
    """Answer an unrecognised command or option.

    Strict mode reproduces argparse's convention (message plus usage on stderr,
    exit ``2``); the default answers on stdout with exit ``0`` and says exactly
    what was ignored and how to make it fatal, so the behaviour is legible
    rather than merely permissive.

    Args:
        tokens: The argv with ``--strict`` removed.
        strict: Whether :data:`STRICT_FLAG` was requested.

    Returns:
        :data:`EXIT_USAGE` in strict mode, otherwise :data:`EXIT_OK`.
    """
    parser = build_parser()
    quoted = " ".join(repr(token) for token in tokens)
    if strict:
        print(f"error: unrecognised argument(s): {quoted}", file=sys.stderr)
        print(parser.format_usage().rstrip(), file=sys.stderr)
        print(f"supported commands: {', '.join(COMMANDS)}", file=sys.stderr)
        return EXIT_USAGE
    print(f"[warn] unrecognised argument(s) ignored: {quoted}")
    print(parser.format_usage().rstrip())
    print(f"[info] supported commands: {', '.join(COMMANDS)}")
    print("[info] full help: python main.py --help")
    print(f"[info] make unrecognised arguments fatal: python main.py {STRICT_FLAG} ...")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the process exit code.

    Args:
        argv: Arguments after the program name.  Defaults to ``sys.argv[1:]``.

    Returns:
        An exit code per the contract in the module docstring (§4.9).  This
        function is total: it never raises ``SystemExit``, so both a recognised
        command's argparse error (``2``) and a handler's failure (``1``) come
        back as a return value that a caller — or a test — can assert on.
    """
    raw = list(sys.argv[1:] if argv is None else argv)
    strict = STRICT_FLAG in raw
    tokens = [token for token in raw if token != STRICT_FLAG]

    if not tokens or tokens[0] in HELP_FLAGS:
        build_parser().print_help()
        return EXIT_OK
    if tokens[0] == VERSION_FLAG:
        return _cmd_version(argparse.Namespace())
    if tokens[0] not in COMMANDS:
        return _report_unrecognised(tokens, strict)

    try:
        args = build_parser().parse_args(tokens)
    except SystemExit as exc:  # argparse already wrote the message
        code = exc.code
        return code if isinstance(code, int) else EXIT_OK
    handler = getattr(args, "handler", None)
    if handler is None:  # pragma: no cover - unreachable while subparsers are wired
        build_parser().print_help()
        return EXIT_OK
    return int(handler(args))


if __name__ == "__main__":
    sys.exit(main())
