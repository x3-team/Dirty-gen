#!/usr/bin/env python3
"""Same-runner release gate for e2e_niva, frame_bench, and the fog step.

Build the pull request and its base commit, then run both on this machine.
Discard one warmup pair. Alternate which side runs first. Fail when the
median of the paired ratios (pull request / base) is above +10%.

The absolute caps stay in the tests and must still be printed at these
digits. Do not raise them:

    E2E_BUDGET ms=11.820
    E2E_AIR_BUDGET ms=1.500
    DRIVE_SOLVER_BUDGET ms=7.000
    DRIVE_P95_BUDGET ms=7.000
    DRIVE_DUST_BUDGET ms=1.000
    SOLVER_FRAME budget=7
    FOG_STEP_BUDGET ms=7.000

A slow runner that trips those caps (EPYC 7763 printed E2E_MEAN 18.842 and
DRIVE_SOLVER 7.122) passes when it matches base. A run more than 10% slower
than base on this runner fails even if both sides are under the caps.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

LIMIT = 1.10
WARMUP_PAIRS = 1
MEASURED_PAIRS = 7
MAX_MEASURED_PAIRS = 13
METRICS = (
    "E2E_MEAN",
    "E2E_AIR",
    "DRIVE_SOLVER",
    "DRIVE_P95",
    "DRIVE_DUST",
    "SOLVER_FRAME",
    "FOG_STEP",
)
ABSOLUTE_CAP_PRINTS = (
    "E2E_BUDGET ms=11.820",
    "E2E_AIR_BUDGET ms=1.500",
    "DRIVE_SOLVER_BUDGET ms=7.000",
    "DRIVE_P95_BUDGET ms=7.000",
    "DRIVE_DUST_BUDGET ms=1.000",
)
FOG_CAP_PRINT = "FOG_STEP_BUDGET ms=7.000"
FRAME_BUDGET_RE = re.compile(r"SOLVER_FRAME ms=[0-9]+(?:\.[0-9]+)? budget=7(?:\.0+)?(?:\s|$)")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
METRIC_RE = {
    "E2E_MEAN": re.compile(r"E2E_MEAN ms=([0-9]+(?:\.[0-9]+)?)"),
    "E2E_AIR": re.compile(r"E2E_AIR ms=([0-9]+(?:\.[0-9]+)?)"),
    "DRIVE_SOLVER": re.compile(r"DRIVE_SOLVER ms=([0-9]+(?:\.[0-9]+)?)"),
    "DRIVE_P95": re.compile(r"DRIVE_P95 ms=([0-9]+(?:\.[0-9]+)?)"),
    "DRIVE_DUST": re.compile(r"DRIVE_DUST ms=([0-9]+(?:\.[0-9]+)?)"),
    "SOLVER_FRAME": re.compile(r"SOLVER_FRAME ms=([0-9]+(?:\.[0-9]+)?)"),
    "FOG_STEP": re.compile(r"step_mean_ms ([0-9]+(?:\.[0-9]+)?)"),
}
BUDGET_PANIC_RE = re.compile(
    r"^(DRIVE_SOLVER|DRIVE_P95|DRIVE_DUST|SOLVER_FRAME|e2e step|air step|FOG_STEP)\b.*\bover\b"
)
ROOT = Path(__file__).resolve().parents[1]


def median(values: list[float]) -> float:
    if not values:
        raise SystemExit("median of no samples")
    return float(statistics.median(values))


def paired_median(base: list[float], head: list[float]) -> float:
    if len(base) != len(head) or not base:
        raise SystemExit(f"pair length {len(base)} vs {len(head)}")
    ratios = []
    for base_ms, head_ms in zip(base, head):
        if not math.isfinite(base_ms) or base_ms <= 0.0 or not math.isfinite(head_ms) or head_ms <= 0.0:
            raise SystemExit(f"non-positive sample base={base_ms} head={head_ms}")
        ratios.append(head_ms / base_ms)
    return median(ratios)


def over_limit(base: list[float], head: list[float]) -> tuple[bool, float]:
    ratio = paired_median(base, head)
    return ratio > LIMIT, ratio


def needs_more(samples_base: dict[str, list[float]], samples_head: dict[str, list[float]]) -> bool:
    for key in METRICS:
        base = samples_base[key]
        head = samples_head[key]
        ratios = [h / b for b, h in zip(base, head)]
        if median(ratios) > 1.04 or statistics.pstdev(ratios) > 0.04:
            return True
    return False


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def panic_messages(text: str) -> list[str]:
    lines = strip_ansi(text).splitlines()
    messages = []
    for index, line in enumerate(lines):
        if "panicked at" not in line:
            continue
        nxt = lines[index + 1].strip() if index + 1 < len(lines) else ""
        if nxt and not nxt.startswith("note:") and not nxt.startswith("stack backtrace"):
            messages.append(nxt)
        else:
            messages.append(line.strip())
    return messages


def unexpected_panics(text: str) -> list[str]:
    return [message for message in panic_messages(text) if BUDGET_PANIC_RE.match(message) is None]


def parse_metrics(text: str) -> dict[str, float]:
    clean = strip_ansi(text)
    found: dict[str, float] = {}
    for key, pattern in METRIC_RE.items():
        matches = pattern.findall(clean)
        if not matches:
            raise SystemExit(f"missing {key} in test output")
        found[key] = float(matches[-1])
    return found


def assert_absolute_caps(text: str, *, fog_budget: bool) -> None:
    clean = strip_ansi(text)
    missing = [token for token in ABSOLUTE_CAP_PRINTS if token not in clean]
    if fog_budget and FOG_CAP_PRINT not in clean:
        missing.append(FOG_CAP_PRINT)
    if missing:
        raise SystemExit("absolute cap print missing or changed: " + ", ".join(missing))
    if FRAME_BUDGET_RE.search(clean) is None:
        raise SystemExit("SOLVER_FRAME budget print is not 7")


def fmt(values: list[float]) -> str:
    return ",".join(f"{value:.4f}" for value in values)


def decide(samples_base: dict[str, list[float]], samples_head: dict[str, list[float]]) -> list[str]:
    failed: list[str] = []
    print(
        "PERF_ABSOLUTE_CAPS "
        "E2E_MEAN=11.820 E2E_AIR=1.500 DRIVE_SOLVER=7 DRIVE_P95=7 "
        "DRIVE_DUST=1 SOLVER_FRAME=7 FOG_STEP=7"
    )
    print(
        "PERF_RULE median of paired pull-request/base ratios; "
        f"fail when median > {LIMIT:.2f}; absolute caps printed and unchanged"
    )
    for key in METRICS:
        base = samples_base[key]
        head = samples_head[key]
        fail, ratio = over_limit(base, head)
        ratios = [h / b for b, h in zip(base, head)]
        status = "fail" if fail else "pass"
        if fail:
            failed.append(key)
        print(
            f"PERF_RATIO metric={key} main_median={median(base):.3f} "
            f"head_median={median(head):.3f} paired_median={ratio:.4f} "
            f"limit={LIMIT:.2f} result={status}"
        )
        print(f"PERF_SAMPLES metric={key} main={fmt(base)} head={fmt(head)} paired={fmt(ratios)}")
    print("PERF_GATE result=" + ("fail" if failed else "pass"))
    if failed:
        print("PERF_GATE failed " + ",".join(failed))
    return failed


def run_checked(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print("PERF_CMD " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=cwd, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def cargo_bins(cwd: Path, target: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target)
    env["CARGO_TERM_COLOR"] = "never"
    cmd = [
        "cargo",
        "test",
        "--release",
        "--no-run",
        "--message-format=json",
        "--test",
        "e2e_niva",
        "--test",
        "frame_bench",
        "--test",
        "fog_air",
    ]
    print(f"PERF_BUILD cwd={cwd}", flush=True)
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"release test build failed in {cwd}")
    bins: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            message = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if message.get("reason") != "compiler-artifact":
            continue
        target_info = message.get("target") or {}
        if "test" not in (target_info.get("kind") or []):
            continue
        name = target_info.get("name")
        executable = message.get("executable")
        if name in {"e2e_niva", "frame_bench", "fog_air"} and executable:
            bins[name] = executable
    missing = sorted({"e2e_niva", "frame_bench", "fog_air"} - bins.keys())
    if missing:
        raise SystemExit(f"missing test binaries in {cwd}: {missing}")
    return bins


def run_test(executable: str, selector: str, *, exact: bool) -> tuple[int, str]:
    cmd = [executable, selector]
    if exact:
        cmd.append("--exact")
    cmd.extend(["--test-threads=1", "--nocapture"])
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=300,
    )
    text = proc.stdout or ""
    sys.stdout.write(text)
    if text and not text.endswith("\n"):
        sys.stdout.write("\n")
    return proc.returncode, text


def run_suite(bins: dict[str, str], side: str, round_index: int) -> str:
    parts: list[str] = []
    specs = (
        ("e2e_niva", "e2e_niva"),
        ("frame_bench", "baseline_solver_frame"),
        ("fog_air", "niva_fog_rides_the_dust_air_and_keeps_its_water"),
    )
    for name, selector in specs:
        print(f"PERF_RUN side={side} round={round_index} test={name}", flush=True)
        code, text = run_test(bins[name], selector, exact=True)
        print(f"PERF_RUN_EXIT side={side} round={round_index} test={name} code={code}", flush=True)
        bad = unexpected_panics(text)
        if bad:
            raise SystemExit(f"non-budget failure in {side} {name}: " + " | ".join(bad[:4]))
        if code != 0 and not panic_messages(text):
            raise SystemExit(f"{side} {name} exited {code} without a budget panic")
        parts.append(text)
    return "\n".join(parts)


def run_must_pass(executable: str, selector: str) -> None:
    print(f"PERF_CHECK {selector}", flush=True)
    code, text = run_test(executable, selector, exact=True)
    if code != 0 or panic_messages(text):
        raise SystemExit(f"{selector} failed")


def prepare_base(sha: str) -> Path:
    print(f"PERF_BASE sha={sha}", flush=True)
    run_checked(["git", "fetch", "--no-tags", "--depth=200", "origin", "main"], ROOT)
    probe = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        cwd=ROOT,
    )
    if probe.returncode != 0:
        run_checked(["git", "fetch", "--no-tags", "origin", "main"], ROOT)
        probe = subprocess.run(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
            cwd=ROOT,
        )
    if probe.returncode != 0:
        raise SystemExit(f"base commit {sha} is not in the fetched main history")
    base = Path(tempfile.gettempdir()) / "silt-perf-base"
    subprocess.run(["git", "worktree", "remove", "--force", str(base)], cwd=ROOT, check=False)
    if base.exists():
        shutil.rmtree(base)
    run_checked(["git", "worktree", "add", "--detach", str(base), sha], ROOT)
    return base


def run_gate() -> int:
    sha = os.environ.get("PERF_BASE_SHA", "").strip()
    if not sha:
        raise SystemExit("PERF_BASE_SHA is not set")
    os.chdir(ROOT)
    base_dir = prepare_base(sha)
    try:
        head_bins = cargo_bins(ROOT, ROOT / "target")
        base_bins = cargo_bins(base_dir, base_dir / "target")
        run_must_pass(head_bins["e2e_niva"], "fine_open")
        run_must_pass(head_bins["fog_air"], "fluid_sheet_fog_is_not_dust_and_closes_the_ledger")
        samples_base = {key: [] for key in METRICS}
        samples_head = {key: [] for key in METRICS}
        measured = 0
        round_index = 0
        while True:
            discard = round_index < WARMUP_PAIRS
            order = ("base", "head") if round_index % 2 == 0 else ("head", "base")
            logs: dict[str, str] = {}
            for side in order:
                bins = base_bins if side == "base" else head_bins
                logs[side] = run_suite(bins, side, round_index)
                assert_absolute_caps(logs[side], fog_budget=(side == "head"))
            if not discard:
                base_metrics = parse_metrics(logs["base"])
                head_metrics = parse_metrics(logs["head"])
                for key in METRICS:
                    samples_base[key].append(base_metrics[key])
                    samples_head[key].append(head_metrics[key])
                measured += 1
            round_index += 1
            if measured < MEASURED_PAIRS:
                continue
            if measured < MAX_MEASURED_PAIRS and needs_more(samples_base, samples_head):
                continue
            break
        print(f"PERF_PAIRS measured={measured} warmup={WARMUP_PAIRS}")
        failed = decide(samples_base, samples_head)
        return 1 if failed else 0
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(base_dir)], cwd=ROOT, check=False)


def self_test() -> None:
    equal = [10.0, 10.2, 9.9, 10.1, 10.0, 10.3, 9.8]
    fail, ratio = over_limit(equal, equal)
    assert not fail and abs(ratio - 1.0) < 1e-9

    # EPYC 7763 absolute numbers, same on both sides: cloud noise, not a regression.
    slow = [18.842, 18.9, 18.7, 19.1, 18.6, 18.8, 19.0]
    fail, ratio = over_limit(slow, [value * 1.02 for value in slow])
    assert not fail and ratio <= LIMIT

    head_bad = [value * 1.12 for value in equal]
    fail, ratio = over_limit(equal, head_bad)
    assert fail and ratio > LIMIT

    # One stolen sample must not fail a run that otherwise matches.
    noisy_head = [10.0, 10.1, 14.0, 9.9, 10.2, 10.0, 9.8]
    fail, _ratio = over_limit(equal, noisy_head)
    assert not fail

    # One quiet sample must not pass a run that is slower on the rest.
    lucky = [12.0, 12.2, 10.0, 12.1, 11.9, 12.3, 12.0]
    fail, _ratio = over_limit(equal, lucky)
    assert fail

    exactly = [10.0] * 7
    fail, ratio = over_limit(exactly, [10.0 * LIMIT] * 7)
    assert not fail and abs(ratio - LIMIT) < 1e-12
    fail, ratio = over_limit(exactly, [10.0 * LIMIT + 0.02] * 7)
    assert fail

    fixture = """
E2E_MEAN ms=18.842
E2E_P95 ms=37.564
E2E_BUDGET ms=11.820
E2E_AIR ms=0.334
E2E_AIR_BUDGET ms=1.500
DRIVE_SOLVER ms=7.122
DRIVE_P95 ms=9.380
DRIVE_DUST ms=1.400
DRIVE_SOLVER_BUDGET ms=7.000
DRIVE_P95_BUDGET ms=7.000
DRIVE_DUST_BUDGET ms=1.000
SOLVER_FRAME ms=4.026 budget=7 frames=180
step_mean_ms 1.200
FOG_STEP_BUDGET ms=7.000
thread 'e2e_niva' panicked at tests/e2e_niva.rs:890:5:
DRIVE_SOLVER 7.122 ms over 7
"""
    metrics = parse_metrics(fixture)
    assert metrics["E2E_MEAN"] == 18.842
    assert metrics["DRIVE_SOLVER"] == 7.122
    assert metrics["DRIVE_DUST"] == 1.400
    assert metrics["SOLVER_FRAME"] == 4.026
    assert metrics["FOG_STEP"] == 1.200
    assert unexpected_panics(fixture) == []
    assert_absolute_caps(fixture, fog_budget=True)
    base_fixture = fixture.replace(FOG_CAP_PRINT, "fog budget not on base")
    assert_absolute_caps(base_fixture, fog_budget=False)
    try:
        assert_absolute_caps(fixture.replace("E2E_BUDGET ms=11.820", "E2E_BUDGET ms=12.000"), fog_budget=True)
    except SystemExit as exc:
        assert "11.820" in str(exc)
    else:
        raise AssertionError("raised cap was accepted")
    correctness = "thread 'e2e_niva' panicked at tests/e2e_niva.rs:694:5:\npuddle depth 0.0100 m\n"
    assert unexpected_panics(correctness) == ["puddle depth 0.0100 m"]
    assert not needs_more(
        {key: [10.0, 10.1, 9.9, 10.0, 10.2, 9.8, 10.0] for key in METRICS},
        {key: [10.1, 10.0, 10.0, 9.9, 10.1, 10.0, 9.9] for key in METRICS},
    )


def main() -> int:
    self_test()
    if "--self-test" in sys.argv:
        print("PERF_SELF_TEST pass")
        return 0
    print("PERF_SELF_TEST pass")
    return run_gate()


if __name__ == "__main__":
    sys.exit(main())
