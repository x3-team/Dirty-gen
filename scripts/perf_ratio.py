#!/usr/bin/env python3
"""Same-runner release gate for e2e_niva, frame_bench, and the fog step.

Build this tree and the base commit, then run both on this machine.
Base samples come first, so the runner scale is fixed before the head runs.
Discard warmup rounds. Fail when any of these is true:

- the head median is above cap * k_runner
- the head median is above the printed cap while the base median is inside it
- the median of head/base is above 1.10
- the median of head/drift is above 1.10

k_runner is the base median divided by main's release measurement on the
EPYC 9V45 runner (run 41218969837, commit 0cf4957). Fog uses the e2e factor.
A head process that exits non-zero fails the job. The base binary still
asserts the unscaled cap, so a slow runner can exit 101; that exit is not
a pass.

The printed caps stay at these digits. Do not raise them:

    E2E_BUDGET ms=11.820
    E2E_AIR_BUDGET ms=1.500
    DRIVE_SOLVER_BUDGET ms=7.000
    DRIVE_P95_BUDGET ms=7.000
    DRIVE_DUST_BUDGET ms=1.000
    SOLVER_FRAME budget=7
    FOG_STEP_BUDGET ms=7.000

An empty PERF_BASE_SHA (push with no parent, or workflow_dispatch) runs the
absolute caps once and does not fail just because the sha is missing.
Pairs, warmup plus measured, are at most 9.
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
# main at the moment this gate landed. Same 1.10 drift bar as the PR base.
DRIFT_SHA = "dfc05d445de19cbaa431d27bbc27fdb09036f638"
# Release log of run 41218969837, commit 0cf4957, AMD EPYC 9V45, 4 cores.
REF_COMMIT = "0cf4957fd29255f9c984b227fb2f02d3b84549d9"
REF_9V45 = {
    "E2E_MEAN": 10.278,
    "E2E_AIR": 0.177,
    "DRIVE_SOLVER": 3.547,
    "DRIVE_P95": 4.748,
    "DRIVE_DUST": 0.917,
    "SOLVER_FRAME": 2.496,
}
CAPS = {
    "E2E_MEAN": 11.820,
    "E2E_AIR": 1.500,
    "DRIVE_SOLVER": 7.0,
    "DRIVE_P95": 7.0,
    "DRIVE_DUST": 1.0,
    "SOLVER_FRAME": 7.0,
    "FOG_STEP": 7.0,
}
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


def paired_median(base: list[float], head: list[float]) -> float:
    if len(base) != len(head) or not base:
        raise SystemExit(f"pair length {len(base)} vs {len(head)}")
    ratios = []
    for base_ms, head_ms in zip(base, head):
        if not math.isfinite(base_ms) or base_ms <= 0.0 or not math.isfinite(head_ms) or head_ms <= 0.0:
            raise SystemExit(f"non-positive sample base={base_ms} head={head_ms}")
        ratios.append(head_ms / base_ms)
    return median(ratios)


def runner_k(base_med: float, key: str, e2e_k: float) -> float:
    ref = REF_9V45.get(key)
    if ref is None:
        return e2e_k
    if not math.isfinite(ref) or ref <= 0.0:
        raise SystemExit(f"bad 9V45 reference for {key}")
    return base_med / ref


def cap_limit(base_med: float, key: str, k: float) -> float:
    cap = CAPS[key]
    scaled = cap * k
    if base_med <= cap:
        return min(scaled, cap)
    return scaled


def cap_failure(base_med: float, head_med: float, key: str, e2e_k: float) -> tuple[bool, float, float, str]:
    k = runner_k(base_med, key, e2e_k)
    limit = cap_limit(base_med, key, k)
    cap = CAPS[key]
    if base_med <= cap and head_med > cap:
        return True, k, limit, "head_over_cap_base_inside"
    if head_med > limit:
        return True, k, limit, "scaled_cap"
    return False, k, limit, "pass"


def head_exit_fails(code: int) -> bool:
    """A head budget panic is a failure. Do not treat exit 101 as a pass."""
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
    samples_drift: dict[str, list[float]],
    extra: list[str] | None = None,
) -> list[str]:
    failed: list[str] = []
    e2e_base = median(samples_base["E2E_MEAN"])
    e2e_k = e2e_base / REF_9V45["E2E_MEAN"]
    print(
        "PERF_ABSOLUTE_CAPS "
        "E2E_MEAN=11.820 E2E_AIR=1.500 DRIVE_SOLVER=7 DRIVE_P95=7 "
        "DRIVE_DUST=1 SOLVER_FRAME=7 FOG_STEP=7"
    )
    print(
        "PERF_RULE fail when head > cap * k_runner; "
        "k_runner = base median / 9V45 main reference; "
        "also fail when head > cap and base <= cap; "
        f"fail when paired median > {LIMIT:.2f} against base and against {DRIFT_SHA}"
    )
    print(f"PERF_REFERENCE commit={REF_COMMIT} run=41218969837 k_e2e={e2e_k:.4f}")
    for key in METRICS:
        base = samples_base[key]
        head = samples_head[key]
        drift = samples_drift[key]
        base_med = median(base)
        head_med = median(head)
        cap_fail, k, limit, cap_reason = cap_failure(base_med, head_med, key, e2e_k)
        ratio_fail, ratio = ratio_over(base, head)
        drift_fail, drift_ratio = ratio_over(drift, head)
        ratios = [h / b for b, h in zip(base, head)]
        if cap_fail:
            failed.append(f"{key}:cap")
        if ratio_fail:
            failed.append(f"{key}:base_ratio")
        if drift_fail:
            failed.append(f"{key}:drift")
        print(
            f"PERF_CAP metric={key} base_median={base_med:.3f} head_median={head_med:.3f} "
            f"reference={'e2e' if key not in REF_9V45 else f'{REF_9V45[key]:.3f}'} k={k:.4f} "
            f"cap={CAPS[key]:.3f} limit={limit:.3f} reason={cap_reason} "
            f"result={'fail' if cap_fail else 'pass'}"
        )
        print(
            f"PERF_RATIO metric={key} main_median={base_med:.3f} "
            f"head_median={head_med:.3f} paired_median={ratio:.4f} "
            f"limit={LIMIT:.2f} result={'fail' if ratio_fail else 'pass'}"
        )
        print(
            f"PERF_DRIFT metric={key} sha={DRIFT_SHA} paired_median={drift_ratio:.4f} "
            f"limit={LIMIT:.2f} result={'fail' if drift_fail else 'pass'}"
        )
        print(f"PERF_SAMPLES metric={key} main={fmt(base)} head={fmt(head)} paired={fmt(ratios)}")
    if extra:
        failed.extend(extra)
    print("PERF_GATE result=" + ("fail" if failed else "pass"))
    if failed:
        print("PERF_GATE failed " + ",".join(failed))
    return failed


def ratio_over(base: list[float], head: list[float]) -> tuple[bool, float]:
    ratio = paired_median(base, head)
    return ratio > LIMIT, ratio


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


def run_suite(bins: dict[str, str], side: str, round_index: int, env: dict[str, str], *, strict: bool) -> tuple[str, list[str]]:
    parts: list[str] = []
    fatals: list[str] = []
    for name, selector in SUITES:
        print(f"PERF_RUN side={side} round={round_index} test={name}", flush=True)
        code, text = run_test(bins[name], selector, env)
        print(f"PERF_RUN_EXIT side={side} round={round_index} test={name} code={code}", flush=True)
        if strict and head_exit_fails(code):
            message = panic_messages(text)
            detail = message[0] if message else "no panic line"
            fatals.append(f"{side} {name} exit {code}: {detail}")
        elif not strict:
            bad = unexpected_panics(text)
            if bad:
                raise SystemExit(f"non-budget failure in {side} {name}: " + " | ".join(bad[:4]))
            if code != 0 and not panic_messages(text):
                raise SystemExit(f"{side} {name} exited {code} without a budget panic")
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


def limit_env(samples_base: dict[str, list[float]], stretch: str) -> dict[str, str]:
    e2e_k = median(samples_base["E2E_MEAN"]) / REF_9V45["E2E_MEAN"]
    extra: dict[str, str] = {}
    for key in METRICS:
        k = runner_k(median(samples_base[key]), key, e2e_k)
        extra[f"PERF_LIMIT_{key}"] = f"{cap_limit(median(samples_base[key]), key, k):.6f}"
    if stretch:
        extra["PERF_STRETCH"] = stretch
    return child_env(extra)


def collect_side(
    bins: dict[str, str],
    side: str,
    rounds: int,
    env: dict[str, str],
    *,
    strict: bool,
    fog_budget: bool,
) -> tuple[dict[str, list[float]], list[str]]:
    samples = blank_samples()
    fatals: list[str] = []
    for round_index in range(rounds):
        text, found = run_suite(bins, side, round_index, env, strict=strict)
        fatals.extend(found)
        assert_absolute_caps(text, fog_budget=fog_budget)
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


def run_caps_only() -> int:
    print("PERF_MODE caps-only", flush=True)
    print("PERF_BASE sha= absent; absolute caps only", flush=True)
    env = child_env()
    bins = cargo_bins(ROOT, ROOT / "target", Path(tempfile.gettempdir()) / "silt-perf-head")
    run_must_pass(bins["e2e_niva"], "fine_open", env)
    run_must_pass(bins["fog_air"], "fluid_sheet_fog_is_not_dust_and_closes_the_ledger", env)
    text, fatals = run_suite(bins, "head", 0, env, strict=True)
    assert_absolute_caps(text, fog_budget=True)
    if fatals:
        print("PERF_HEAD_PANIC " + " | ".join(fatals))
        print("PERF_GATE result=fail")
        return 1
    metrics = parse_metrics(text)
    failed = [key for key in METRICS if metrics[key] > CAPS[key]]
    for key in METRICS:
        status = "fail" if key in failed else "pass"
        print(
            f"PERF_CAP metric={key} head={metrics[key]:.3f} cap={CAPS[key]:.3f} "
            f"k=1 result={status}"
        )
    print("PERF_GATE result=" + ("fail" if failed else "pass"))
    return 1 if failed else 0


def run_gate(sha: str) -> int:
    warmup, measured = pair_counts()
    stretch = prove_stretch()
    print(f"PERF_MODE ratio pairs={warmup + measured} warmup={warmup} measured={measured}", flush=True)
    print(f"PERF_BASE sha={sha}", flush=True)
    print(f"PERF_DRIFT sha={DRIFT_SHA}", flush=True)
    os.chdir(ROOT)
    base_dir = Path(tempfile.gettempdir()) / "silt-perf-base"
    drift_dir = Path(tempfile.gettempdir()) / "silt-perf-drift"
    prepared = [base_dir]
    try:
        prepare_tree(sha, base_dir)
        shared = ROOT / "target"
        head_bins = cargo_bins(ROOT, shared, Path(tempfile.gettempdir()) / "silt-perf-head")
        base_bins = cargo_bins(base_dir, shared, Path(tempfile.gettempdir()) / "silt-perf-basebins")
        plain = child_env()
        run_must_pass(head_bins["e2e_niva"], "fine_open", plain)
        run_must_pass(head_bins["fog_air"], "fluid_sheet_fog_is_not_dust_and_closes_the_ledger", plain)
        base_all, _base_fatals = collect_side(
            base_bins, "base", warmup + measured, plain, strict=False, fog_budget=False
        )
        samples_base = {key: values[warmup:] for key, values in base_all.items()}
        head_env = limit_env(samples_base, stretch)
        head_all, head_fatals = collect_side(
            head_bins, "head", warmup + measured, head_env, strict=True, fog_budget=True
        )
        samples_head = {key: values[warmup:] for key, values in head_all.items()}
        if sha == DRIFT_SHA:
            print("PERF_DRIFT source=base", flush=True)
            samples_drift = samples_base
        else:
            prepare_tree(DRIFT_SHA, drift_dir)
            prepared.append(drift_dir)
            drift_bins = cargo_bins(drift_dir, shared, Path(tempfile.gettempdir()) / "silt-perf-driftbins")
            drift_all, _drift_fatals = collect_side(
                drift_bins, "drift", warmup + measured, plain, strict=False, fog_budget=False
            )
            samples_drift = {key: values[warmup:] for key, values in drift_all.items()}
        print(f"PERF_PAIRS measured={measured} warmup={warmup} total={warmup + measured} max={MAX_PAIRS}")
        extra: list[str] = []
        if head_fatals:
            print("PERF_HEAD_PANIC " + " | ".join(head_fatals))
            extra.append("head_panic")
        failed = evaluate(samples_base, samples_head, samples_drift, extra)
        return 1 if failed else 0
    finally:
        for path in prepared:
            subprocess.run(["git", "worktree", "remove", "--force", str(path)], cwd=ROOT, check=False)


def self_test() -> None:
    base = {key: [10.0, 10.2, 9.9, 10.1] for key in METRICS}
    head = {key: [10.0, 10.1, 10.2, 9.8] for key in METRICS}
    # Inside the reference scale this is a tie. Use the 9V45-shaped case below for caps.
    assert ratio_over(base["E2E_MEAN"], head["E2E_MEAN"])[0] is False

    slow_base = {
        "E2E_MEAN": [19.173, 19.080, 19.267, 19.325],
        "E2E_AIR": [0.349, 0.348, 0.350, 0.347],
        "DRIVE_SOLVER": [7.337, 7.300, 7.360, 7.340],
        "DRIVE_P95": [9.106, 9.050, 9.180, 9.120],
        "DRIVE_DUST": [1.367, 1.363, 1.370, 1.365],
        "SOLVER_FRAME": [4.129, 4.113, 4.160, 4.108],
        "FOG_STEP": [1.401, 1.390, 1.410, 1.388],
    }
    slow_head = {
        "E2E_MEAN": [19.327, 19.187, 19.434, 19.283],
        "E2E_AIR": [0.348, 0.347, 0.349, 0.346],
        "DRIVE_SOLVER": [7.391, 7.360, 7.400, 7.380],
        "DRIVE_P95": [9.174, 9.100, 9.200, 9.150],
        "DRIVE_DUST": [1.404, 1.398, 1.407, 1.399],
        "SOLVER_FRAME": [4.000, 3.983, 4.018, 3.992],
        "FOG_STEP": [1.388, 1.380, 1.400, 1.390],
    }
    assert evaluate(slow_base, slow_head, slow_base) == []

    stretched = {key: [value * 1.12 for value in values] for key, values in slow_head.items()}
    failed = evaluate(slow_base, stretched, slow_base)
    assert any(item.endswith(":base_ratio") for item in failed)
    assert any(item.endswith(":drift") for item in failed)

    # Base inside the cap, head over the cap, ratio still under 1.10.
    inside = {key: list(values) for key, values in slow_base.items()}
    over = {key: list(values) for key, values in slow_head.items()}
    inside["SOLVER_FRAME"] = [4.0, 4.0, 4.0, 4.0]
    over["SOLVER_FRAME"] = [4.2, 4.2, 4.2, 7.2]
    # median of head frame is 4.2; force every sample over the cap.
    over["SOLVER_FRAME"] = [7.2, 7.2, 7.2, 7.2]
    failed = evaluate(inside, over, inside)
    assert "SOLVER_FRAME:cap" in failed

    # Head above cap * k while both sides are over the printed cap.
    huge = {key: [value * 1.20 for value in values] for key, values in slow_base.items()}
    # 20% over base is above 1.10. Also check the e2e scaled cap:
    # limit = 11.820 * (19.173/10.278) ≈ 22.05, and 19.173*1.20 ≈ 23.0.
    failed = evaluate(slow_base, huge, slow_base)
    assert "E2E_MEAN:cap" in failed
    assert "E2E_MEAN:base_ratio" in failed

    exactly = [10.0, 10.0, 10.0, 10.0]
    fail, ratio = ratio_over(exactly, [value * LIMIT for value in exactly])
    assert not fail and abs(ratio - LIMIT) < 1e-12
    fail, ratio = ratio_over(exactly, [value * LIMIT + 0.02 for value in exactly])
    assert fail and ratio > LIMIT

    assert head_exit_fails(101)
    assert head_exit_fails(1)
    assert not head_exit_fails(0)

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
    assert metrics["FOG_STEP"] == 1.200
    assert unexpected_panics(fixture) == []
    assert_absolute_caps(fixture, fog_budget=True)
    assert head_exit_fails(101)
    try:
        assert_absolute_caps(fixture.replace("E2E_BUDGET ms=11.820", "E2E_BUDGET ms=12.000"), fog_budget=True)
    except SystemExit as exc:
        assert "11.820" in str(exc)
    else:
        raise AssertionError("raised cap was accepted")


def normalized_base_sha() -> str:
    sha = os.environ.get("PERF_BASE_SHA", "").strip()
    if sha == ZERO_SHA:
        return ""
    return sha


def main() -> int:
    self_test()
    if "--self-test" in sys.argv:
        print("PERF_SELF_TEST pass")
        return 0
    print("PERF_SELF_TEST pass")
    sha = normalized_base_sha()
    if not sha:
        return run_caps_only()
    return run_gate(sha)


if __name__ == "__main__":
    sys.exit(main())
