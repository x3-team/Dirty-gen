#!/usr/bin/env python3
"""Same-runner release gate for e2e_niva, frame_bench, and the fog step.

Build this tree and the pinned base, then run both on this machine.
Discard warmup rounds. Absolute caps are flat. Nothing multiplies them.

Fail when any of these is true:

- the base process exits non-zero (a budget panic is an error)
- the head process exits non-zero
- a base or head median is above its flat cap
- k_runner, the median of head/base, is above 1.10

k_runner is only the paired head/base comparison. It does not scale a cap.

The pinned base is 0c04f3e (main with #112 merged). Its tree is f3cd933,
which passed these caps in run 41302076195. Pairs are at most 9.
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
DEFAULT_WARMUP = 1
DEFAULT_MEASURED = 4
MAX_PAIRS = 9
# Release CI run 41302076195 passed the flat caps on this tree (#112 head f3cd933).
# e2e 10.347, air 0.191, dust 0.748, solver 3.741, water 0.565,
# solver water 0.450, frame 2.517, fog step 0.766.
PINNED_BASE_SHA = "0c04f3e73d3de5500188bfeff4ee075eefb196e6"
CAPS = {
    "E2E_MEAN": 11.820,
    "E2E_AIR": 1.5,
    "DRIVE_SOLVER": 7.0,
    "DRIVE_P95": 7.0,
    "DRIVE_DUST": 1.0,
    "DRIVE_WATER": 2.0,
    "SOLVER_FRAME": 7.0,
    "SOLVER_WATER": 2.0,
    "FOG_STEP": 7.0,
}
METRICS = (
    "E2E_MEAN",
    "E2E_AIR",
    "DRIVE_SOLVER",
    "DRIVE_P95",
    "DRIVE_DUST",
    "DRIVE_WATER",
    "SOLVER_FRAME",
    "SOLVER_WATER",
    "FOG_STEP",
)
ABSOLUTE_CAP_PRINTS = (
    "E2E_BUDGET ms=11.820",
    "E2E_AIR_BUDGET ms=1.500",
    "DRIVE_SOLVER_BUDGET ms=7.000",
    "DRIVE_P95_BUDGET ms=7.000",
    "DRIVE_DUST_BUDGET ms=1.000",
)
WATER_CAP_PRINTS = (
    "DRIVE_WATER_BUDGET ms=2.000",
    "SOLVER_WATER ms=",
    "budget=2.000",
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
    "DRIVE_WATER": re.compile(r"DRIVE_WATER ms=([0-9]+(?:\.[0-9]+)?)"),
    "SOLVER_FRAME": re.compile(r"SOLVER_FRAME ms=([0-9]+(?:\.[0-9]+)?)"),
    "SOLVER_WATER": re.compile(r"SOLVER_WATER ms=([0-9]+(?:\.[0-9]+)?)"),
    "FOG_STEP": re.compile(r"step_mean_ms ([0-9]+(?:\.[0-9]+)?)"),
}
ZERO_SHA = "0" * 40
ROOT = Path(__file__).resolve().parents[1]
SUITES = (
    ("e2e_niva", "e2e_niva"),
    ("frame_bench", "baseline_solver_frame"),
    ("fog_air", "niva_fog_rides_the_dust_air_and_keeps_its_water"),
)


def median(values: list[float]) -> float:
    if not values:
        raise SystemExit("median of no samples")
    return float(statistics.median(values))


def k_runner(base: list[float], head: list[float]) -> float:
    """Paired head/base factor. This is the only use of k_runner."""
    if len(base) != len(head) or not base:
        raise SystemExit(f"pair length {len(base)} vs {len(head)}")
    ratios = []
    for base_ms, head_ms in zip(base, head):
        if not math.isfinite(base_ms) or base_ms <= 0.0 or not math.isfinite(head_ms) or head_ms <= 0.0:
            raise SystemExit(f"non-positive sample base={base_ms} head={head_ms}")
        ratios.append(head_ms / base_ms)
    return median(ratios)


def exit_fails(code: int) -> bool:
    """Any non-zero exit is a gate error. Do not treat 101 as a pass."""
    return code != 0


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


def parse_metrics(text: str) -> dict[str, float]:
    clean = strip_ansi(text)
    found: dict[str, float] = {}
    for key, pattern in METRIC_RE.items():
        matches = pattern.findall(clean)
        if not matches:
            raise SystemExit(f"missing {key} in test output")
        found[key] = float(matches[-1])
    return found


def assert_absolute_caps(text: str, *, fog_budget: bool, water_budget: bool) -> None:
    clean = strip_ansi(text)
    missing = [token for token in ABSOLUTE_CAP_PRINTS if token not in clean]
    if fog_budget and FOG_CAP_PRINT not in clean:
        missing.append(FOG_CAP_PRINT)
    if water_budget:
        for token in WATER_CAP_PRINTS:
            if token not in clean:
                missing.append(token)
    if missing:
        raise SystemExit("absolute cap print missing or changed: " + ", ".join(missing))
    if FRAME_BUDGET_RE.search(clean) is None:
        raise SystemExit("SOLVER_FRAME budget print is not 7")


def fmt(values: list[float]) -> str:
    return ",".join(f"{value:.4f}" for value in values)


def pair_counts() -> tuple[int, int]:
    warmup = int(os.environ.get("PERF_WARMUP_PAIRS", str(DEFAULT_WARMUP)))
    measured = int(os.environ.get("PERF_MEASURED_PAIRS", str(DEFAULT_MEASURED)))
    if warmup < 0 or measured < 1:
        raise SystemExit("PERF_WARMUP_PAIRS must be >= 0 and PERF_MEASURED_PAIRS must be >= 1")
    if warmup + measured > MAX_PAIRS:
        raise SystemExit(f"pairs {warmup + measured} exceed {MAX_PAIRS}")
    return warmup, measured


def evaluate(
    samples_base: dict[str, list[float]],
    samples_head: dict[str, list[float]],
    extra: list[str] | None = None,
) -> list[str]:
    failed: list[str] = []
    print(
        "PERF_ABSOLUTE_CAPS "
        "E2E_MEAN=11.820 E2E_AIR=1.500 DRIVE_SOLVER=7 DRIVE_P95=7 "
        "DRIVE_DUST=1 DRIVE_WATER=2 SOLVER_FRAME=7 SOLVER_WATER=2 FOG_STEP=7"
    )
    print(
        "PERF_RULE flat caps; fail when base or head median > cap; "
        f"k_runner is the paired head/base median and fails above {LIMIT:.2f}; "
        "k_runner does not scale a cap"
    )
    for key in METRICS:
        base = samples_base[key]
        head = samples_head[key]
        base_med = median(base)
        head_med = median(head)
        cap = CAPS[key]
        ratio = k_runner(base, head)
        ratios = [h / b for b, h in zip(base, head)]
        cap_bits = []
        if base_med > cap:
            cap_bits.append("base")
            failed.append(f"{key}:base_cap")
        if head_med > cap:
            cap_bits.append("head")
            failed.append(f"{key}:head_cap")
        ratio_fail = ratio > LIMIT
        if ratio_fail:
            failed.append(f"{key}:base_ratio")
        cap_result = "fail" if cap_bits else "pass"
        print(
            f"PERF_CAP metric={key} base_median={base_med:.3f} head_median={head_med:.3f} "
            f"cap={cap:.3f} over={','.join(cap_bits) if cap_bits else 'none'} result={cap_result}"
        )
        print(
            f"PERF_RATIO metric={key} k_runner={ratio:.4f} limit={LIMIT:.2f} "
            f"result={'fail' if ratio_fail else 'pass'}"
        )
        print(f"PERF_SAMPLES metric={key} base={fmt(base)} head={fmt(head)} paired={fmt(ratios)}")
    if extra:
        failed.extend(extra)
    print("PERF_GATE result=" + ("fail" if failed else "pass"))
    if failed:
        print("PERF_GATE failed " + ",".join(failed))
    return failed


def blank_samples() -> dict[str, list[float]]:
    return {key: [] for key in METRICS}


def run_checked(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print("PERF_CMD " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=cwd, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PERF_STRETCH", None)
    env.pop("PERF_PROVE_SLOW", None)
    for key in list(env):
        if key.startswith("PERF_LIMIT_") or key.startswith("PERF_K_"):
            env.pop(key, None)
    env["CARGO_TERM_COLOR"] = "never"
    if extra:
        env.update(extra)
    return env


def cargo_bins(cwd: Path, target: Path, dest: Path) -> dict[str, str]:
    dest.mkdir(parents=True, exist_ok=True)
    env = child_env({"CARGO_TARGET_DIR": str(target)})
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
    proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=None, text=True)
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
    copied: dict[str, str] = {}
    for name, executable in bins.items():
        path = dest / name
        shutil.copy2(executable, path)
        path.chmod(0o755)
        copied[name] = str(path)
    return copied


def run_test(executable: str, selector: str, env: dict[str, str]) -> tuple[int, str]:
    cmd = [executable, selector, "--exact", "--test-threads=1", "--nocapture"]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
        env=env,
    )
    text = proc.stdout or ""
    sys.stdout.write(text)
    if text and not text.endswith("\n"):
        sys.stdout.write("\n")
    return proc.returncode, text


def run_suite(bins: dict[str, str], side: str, round_index: int, env: dict[str, str]) -> tuple[str, list[str]]:
    parts: list[str] = []
    fatals: list[str] = []
    for name, selector in SUITES:
        print(f"PERF_RUN side={side} round={round_index} test={name}", flush=True)
        code, text = run_test(bins[name], selector, env)
        print(f"PERF_RUN_EXIT side={side} round={round_index} test={name} code={code}", flush=True)
        if exit_fails(code):
            message = panic_messages(text)
            detail = message[0] if message else "no panic line"
            fatals.append(f"{side} {name} exit {code}: {detail}")
        parts.append(text)
    return "\n".join(parts), fatals


def run_must_pass(executable: str, selector: str, env: dict[str, str]) -> None:
    print(f"PERF_CHECK {selector}", flush=True)
    code, text = run_test(executable, selector, env)
    if code != 0 or panic_messages(text):
        raise SystemExit(f"{selector} failed")


def commit_exists(sha: str) -> bool:
    probe = subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=ROOT)
    return probe.returncode == 0


def prepare_tree(sha: str, dest: Path) -> Path:
    print(f"PERF_TREE sha={sha} dest={dest}", flush=True)
    run_checked(["git", "fetch", "--no-tags", "--depth=200", "origin", "main"], ROOT)
    if not commit_exists(sha):
        run_checked(["git", "fetch", "--no-tags", "origin", "main"], ROOT)
    if not commit_exists(sha):
        raise SystemExit(f"commit {sha} is not in the fetched main history")
    subprocess.run(["git", "worktree", "remove", "--force", str(dest)], cwd=ROOT, check=False)
    if dest.exists():
        shutil.rmtree(dest)
    run_checked(["git", "worktree", "add", "--detach", str(dest), sha], ROOT)
    return dest


def collect_side(
    bins: dict[str, str],
    side: str,
    rounds: int,
    env: dict[str, str],
    *,
    fog_budget: bool,
    water_budget: bool,
) -> tuple[dict[str, list[float]], list[str]]:
    samples = blank_samples()
    fatals: list[str] = []
    for round_index in range(rounds):
        text, found = run_suite(bins, side, round_index, env)
        fatals.extend(found)
        if found:
            continue
        assert_absolute_caps(text, fog_budget=fog_budget, water_budget=water_budget)
        metrics = parse_metrics(text)
        for key in METRICS:
            samples[key].append(metrics[key])
    return samples, fatals


def prove_stretch() -> str:
    raw = os.environ.get("PERF_PROVE_SLOW", "").strip()
    if not raw:
        return ""
    factor = float(raw)
    if not math.isfinite(factor) or factor < 1.12:
        raise SystemExit("PERF_PROVE_SLOW must be a finite slowdown of at least 1.12")
    print(f"PERF_PROVE_SLOW factor={factor:.4f} side=head", flush=True)
    return f"{factor:.4f}"


def fail_gate(reason: str, detail: str) -> int:
    print(detail)
    print("PERF_GATE result=fail")
    print(f"PERF_GATE failed {reason}")
    return 1


def run_gate(sha: str) -> int:
    warmup, measured = pair_counts()
    stretch = prove_stretch()
    print(f"PERF_MODE ratio pairs={warmup + measured} warmup={warmup} measured={measured}", flush=True)
    print(f"PERF_BASE sha={sha}", flush=True)
    print(f"PERF_DRIFT sha={sha} source=base", flush=True)
    os.chdir(ROOT)
    base_dir = Path(tempfile.gettempdir()) / "silt-perf-base"
    try:
        prepare_tree(sha, base_dir)
        shared = ROOT / "target"
        head_bins = cargo_bins(ROOT, shared, Path(tempfile.gettempdir()) / "silt-perf-head")
        base_bins = cargo_bins(base_dir, shared, Path(tempfile.gettempdir()) / "silt-perf-basebins")
        plain = child_env()
        run_must_pass(head_bins["e2e_niva"], "fine_open", plain)
        run_must_pass(head_bins["fog_air"], "fluid_sheet_fog_is_not_dust_and_closes_the_ledger", plain)
        base_all, base_fatals = collect_side(
            base_bins,
            "base",
            warmup + measured,
            plain,
            fog_budget=False,
            water_budget=False,
        )
        if base_fatals:
            return fail_gate("base_exit", "PERF_BASE_ERROR " + " | ".join(base_fatals))
        if any(len(values) != warmup + measured for values in base_all.values()):
            return fail_gate("base_exit", "PERF_BASE_ERROR incomplete base samples")
        samples_base = {key: values[warmup:] for key, values in base_all.items()}
        head_env = child_env({"PERF_STRETCH": stretch} if stretch else None)
        head_all, head_fatals = collect_side(
            head_bins,
            "head",
            warmup + measured,
            head_env,
            fog_budget=True,
            water_budget=True,
        )
        if head_fatals:
            print("PERF_HEAD_PANIC " + " | ".join(head_fatals))
            return fail_gate("head_panic", "PERF_HEAD_ERROR head exited non-zero")
        if any(len(values) != warmup + measured for values in head_all.values()):
            return fail_gate("head_panic", "PERF_HEAD_ERROR incomplete head samples")
        samples_head = {key: values[warmup:] for key, values in head_all.items()}
        print(f"PERF_PAIRS measured={measured} warmup={warmup} total={warmup + measured} max={MAX_PAIRS}")
        extra = ["head_panic"] if head_fatals else []
        failed = evaluate(samples_base, samples_head, extra)
        return 1 if failed else 0
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(base_dir)], cwd=ROOT, check=False)


def under_cap() -> dict[str, list[float]]:
    return {
        "E2E_MEAN": [10.2, 10.1, 10.3, 10.0],
        "E2E_AIR": [0.19, 0.18, 0.20, 0.19],
        "DRIVE_SOLVER": [3.7, 3.6, 3.8, 3.7],
        "DRIVE_P95": [5.0, 4.9, 5.1, 5.0],
        "DRIVE_DUST": [0.75, 0.74, 0.76, 0.75],
        "DRIVE_WATER": [0.56, 0.55, 0.57, 0.56],
        "SOLVER_FRAME": [2.5, 2.4, 2.6, 2.5],
        "SOLVER_WATER": [0.45, 0.44, 0.46, 0.45],
        "FOG_STEP": [0.77, 0.76, 0.78, 0.77],
    }


def self_test() -> None:
    base = under_cap()
    head = under_cap()
    assert evaluate(base, head) == []
    assert abs(k_runner(base["E2E_MEAN"], head["E2E_MEAN"]) - 1.0) < 0.05

    stretched = {key: list(values) for key, values in head.items()}
    stretched["E2E_MEAN"] = [value * 1.12 for value in base["E2E_MEAN"]]
    failed = evaluate(base, stretched)
    assert "E2E_MEAN:base_ratio" in failed
    assert "E2E_MEAN:head_cap" not in failed
    assert "E2E_MEAN:base_cap" not in failed

    over_head = {key: list(values) for key, values in head.items()}
    over_head["E2E_MEAN"] = [12.0, 12.0, 12.0, 12.0]
    failed = evaluate(base, over_head)
    assert "E2E_MEAN:head_cap" in failed
    assert "E2E_MEAN:base_ratio" in failed

    over_base = {key: list(values) for key, values in base.items()}
    over_base["DRIVE_DUST"] = [1.05, 1.05, 1.05, 1.05]
    under_head = {key: list(values) for key, values in head.items()}
    under_head["DRIVE_DUST"] = [0.99, 0.99, 0.99, 0.99]
    failed = evaluate(over_base, under_head)
    assert "DRIVE_DUST:base_cap" in failed
    assert "DRIVE_DUST:head_cap" not in failed

    slow_base = {key: list(values) for key, values in base.items()}
    slow_head = {key: list(values) for key, values in head.items()}
    slow_base["E2E_MEAN"] = [19.173, 19.080, 19.267, 19.325]
    slow_head["E2E_MEAN"] = [19.327, 19.187, 19.434, 19.283]
    slow_base["DRIVE_DUST"] = [1.367, 1.363, 1.370, 1.365]
    slow_head["DRIVE_DUST"] = [1.404, 1.398, 1.407, 1.399]
    slow_base["DRIVE_SOLVER"] = [7.337, 7.300, 7.360, 7.340]
    slow_head["DRIVE_SOLVER"] = [7.391, 7.360, 7.400, 7.380]
    slow_base["DRIVE_P95"] = [9.106, 9.050, 9.180, 9.120]
    slow_head["DRIVE_P95"] = [9.174, 9.100, 9.200, 9.150]
    failed = evaluate(slow_base, slow_head)
    assert "E2E_MEAN:base_cap" in failed and "E2E_MEAN:head_cap" in failed
    assert "DRIVE_DUST:base_cap" in failed and "DRIVE_SOLVER:base_cap" in failed
    assert not any(item.endswith(":base_ratio") for item in failed)

    water = {key: list(values) for key, values in base.items()}
    water_head = {key: list(values) for key, values in head.items()}
    water_head["DRIVE_WATER"] = [2.1, 2.1, 2.1, 2.1]
    water_head["SOLVER_WATER"] = [2.1, 2.1, 2.1, 2.1]
    failed = evaluate(water, water_head)
    assert "DRIVE_WATER:head_cap" in failed
    assert "SOLVER_WATER:head_cap" in failed

    exactly = [10.0, 10.0, 10.0, 10.0]
    ratio = k_runner(exactly, [value * LIMIT for value in exactly])
    assert abs(ratio - LIMIT) < 1e-12
    assert not (ratio > LIMIT)
    assert k_runner(exactly, [value * LIMIT + 0.02 for value in exactly]) > LIMIT

    assert exit_fails(101)
    assert exit_fails(1)
    assert not exit_fails(0)

    fixture = """
E2E_MEAN ms=10.347
E2E_P95 ms=12.000
E2E_BUDGET ms=11.820
E2E_AIR ms=0.191
E2E_AIR_BUDGET ms=1.500
DRIVE_SOLVER ms=3.741
DRIVE_P95 ms=5.010
DRIVE_DUST ms=0.748
DRIVE_WATER ms=0.565
DRIVE_WATER_BUDGET ms=2.000
DRIVE_SOLVER_BUDGET ms=7.000
DRIVE_P95_BUDGET ms=7.000
DRIVE_DUST_BUDGET ms=1.000
SOLVER_FRAME ms=2.517 budget=7 frames=180
SOLVER_WATER ms=0.450 budget=2.000
step_mean_ms 0.766
FOG_STEP_BUDGET ms=7.000
"""
    metrics = parse_metrics(fixture)
    assert metrics["E2E_MEAN"] == 10.347
    assert metrics["DRIVE_WATER"] == 0.565
    assert metrics["SOLVER_WATER"] == 0.450
    assert metrics["FOG_STEP"] == 0.766
    assert_absolute_caps(fixture, fog_budget=True, water_budget=True)
    try:
        assert_absolute_caps(fixture.replace("E2E_BUDGET ms=11.820", "E2E_BUDGET ms=12.000"), fog_budget=True, water_budget=True)
    except SystemExit as exc:
        assert "11.820" in str(exc)
    else:
        raise AssertionError("raised cap was accepted")
    try:
        assert_absolute_caps(fixture.replace("budget=2.000", "budget=3.000"), fog_budget=True, water_budget=True)
    except SystemExit as exc:
        assert "2.000" in str(exc)
    else:
        raise AssertionError("raised water cap was accepted")


def resolve_base_sha() -> str:
    sha = os.environ.get("PERF_BASE_SHA", "").strip()
    if not sha or sha == ZERO_SHA:
        return PINNED_BASE_SHA
    return sha


def main() -> int:
    self_test()
    print("PERF_SELF_TEST pass")
    if "--self-test" in sys.argv:
        return 0
    return run_gate(resolve_base_sha())


if __name__ == "__main__":
    sys.exit(main())
