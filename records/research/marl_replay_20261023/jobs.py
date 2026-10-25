"""Job runner (design sections 8 and 11).

  jobs.py j0                                  development phase and timing (job J0)
  jobs.py teams --cond W1 --seeds 1-20 --job J1   train, gate set and evaluation for each team
  jobs.py score --seeds 1-20 --job J3a            monitor scores of the W1 teams
  jobs.py plan                                print the job list of the main run (J1, J2, J3)

Stop rules: S2 (projected job time after 3 teams > 110 min: stop after the current
team), S3 (NaN: stop that team, count it as failed, no replacement), a self-stop at
115 min (the watchdog backstop is 125 min). Each job prints one END line.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import evaluate as EV  # noqa: E402
import monitors as MON  # noqa: E402
from team import BUDGETS, N_ENVS, ROLLOUT, NaNError, Trainer  # noqa: E402

RUNS = ROOT / "runs"
DEV_SEEDS = (9001, 9002)
TIMING_SEED = 9999  # untrained network, J0 timing only; no output except wall times
J0_CAP_MIN = 60
S2_MIN = 110
SELF_CAP_MIN = 115
PLAN_MIN = 100  # each planned job must project at most this many minutes
PLAN_MARGIN = 1.25
PLAN_SIZES = (20, 10, 5, 4, 2, 1)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_seeds(text):
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def team_dir(cond, s):
    return RUNS / f"{cond}_s{int(s):02d}"


def save_team(trainer, path, budget_name):
    torch.save(dict(seed=trainer.seed, condition=trainer.condition, budget=budget_name,
                    updates=trainer.updates, state=trainer.team.state_dict()), path)


def plan_jobs(minutes_per_team, n_teams=20):
    for k in PLAN_SIZES:
        if k * minutes_per_team * PLAN_MARGIN <= PLAN_MIN:
            return dict(teams_per_job=k, n_jobs=-(-n_teams // k), projected_min_per_job=k * minutes_per_team)
    return dict(teams_per_job=None, n_jobs=None, projected_min_per_job=None)


def job_j0():
    out = RUNS / "j0"
    out.mkdir(parents=True, exist_ok=True)
    T0 = time.perf_counter()
    trainers = {s: Trainer(s, "W1") for s in DEV_SEEDS}
    nan = {s: None for s in DEV_SEEDS}
    records, selected, stop = [], None, None
    names = list(BUDGETS)
    for i, name in enumerate(names):
        passes = []
        for s in DEV_SEEDS:
            tr = trainers[s]
            if nan[s] is None:
                try:
                    tr.train_until(BUDGETS[name])
                except NaNError as e:
                    nan[s] = str(e)
            if nan[s] is not None:
                rec = dict(budget=name, seed=s, nan=nan[s], passed=False)
            else:
                save_team(tr, out / f"dev_s{s}_{name}.pt", name)
                tg = time.perf_counter()
                g = EV.gate_eval(tr.team, s)
                rec = dict(budget=name, seed=s, updates=tr.updates, steps=tr.updates * N_ENVS * ROLLOUT,
                           train_seconds_cumulative=tr.seconds, gate_seconds=time.perf_counter() - tg,
                           gate=g, **EV.gates_pass(g, "W1"),
                           train_log_last=tr.log[-1])
            records.append(rec); passes.append(rec["passed"])
            log(f"J0 {name} seed {s}: " + json.dumps({k: rec.get(k) for k in ["gate", "passed", "train_seconds_cumulative", "nan"]}))
        elapsed = time.perf_counter() - T0
        if all(passes):
            selected = name
            break
        if i + 1 < len(names):
            per_update = max(trainers[s].seconds / max(trainers[s].updates, 1) for s in DEV_SEEDS)
            extra = sum((BUDGETS[names[i + 1]] - trainers[s].updates) * per_update for s in DEV_SEEDS if nan[s] is None)
            if elapsed + extra > J0_CAP_MIN * 60:
                stop = f"J0 cap: {elapsed/60:.1f} min used, {extra/60:.1f} min more projected for {names[i+1]} > {J0_CAP_MIN} min"
                break
    if selected is None and stop is None:
        stop = "no budget passed G-a and G-b on both development teams (design 8.1): stop, report to [researcher]"
    decision = dict(selected=selected, stop=stop, records=records,
                    dev_full_logs={s: trainers[s].log for s in DEV_SEEDS})
    if selected is not None:
        t1 = time.perf_counter()
        et = EV.run_team(None, TIMING_SEED, "W1", timing_only=True)["seconds"]
        decision["eval_timing_W1_seconds"] = et
        decision["eval_timing_wall"] = time.perf_counter() - t1
        decision["monitor_timing"] = MON.synthetic_timing()
        train_min = max(r["train_seconds_cumulative"] for r in records if r["budget"] == selected) / 60
        gate_min = max(r["gate_seconds"] for r in records if r["budget"] == selected) / 60
        per_team_w1 = train_min + gate_min + et["total"] / 60
        per_team_w0 = train_min + gate_min + (et["total"] - et.get("monitor_sets", 0)) / 60
        mt = decision["monitor_timing"]
        per_team_j3 = (mt["seconds_test"] + mt["seconds_silent"]) / 60
        decision["plan"] = dict(minutes_per_team_W1=per_team_w1, minutes_per_team_W0=per_team_w0,
                                minutes_per_team_J3=per_team_j3,
                                J1=plan_jobs(per_team_w1), J2=plan_jobs(per_team_w0), J3=plan_jobs(per_team_j3))
        if selected == "D3":
            decision["stop"] = "D3 selected: the cost doubles; ask [researcher] before the main run (design section 11)"
    decision["seconds"] = time.perf_counter() - T0
    (out / "decision.json").write_text(json.dumps(decision, indent=2))
    log("J0 decision: " + json.dumps({k: decision.get(k) for k in ["selected", "stop", "plan"]}))
    status = "complete" if decision.get("stop") is None else "stopped"
    print("END " + json.dumps(dict(status=status, selected=selected, stop=decision.get("stop"),
                                   wall_seconds=decision["seconds"])), flush=True)
    return 0 if status == "complete" else 4


def selected_budget():
    d = json.loads((RUNS / "j0" / "decision.json").read_text())
    if d.get("stop") is not None or d.get("selected") is None:
        raise SystemExit(f"REFUSED: J0 did not select a budget without a stop: {d.get('stop')}")
    return d["selected"]


def job_teams(cond, seeds, job):
    name = selected_budget()
    T0 = time.perf_counter()
    done = []
    for i, s in enumerate(seeds):
        d = team_dir(cond, s)
        if (d / "eval.json").exists() or (d / "FAILED_NAN").exists():
            log(f"{job} {cond} s{s}: complete record exists, skip")
            continue
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        tr = Trainer(s, cond)
        try:
            tr.train_until(BUDGETS[name])
        except NaNError as e:
            (d / "FAILED_NAN").write_text(str(e) + "\n")
            (d / "train.json").write_text(json.dumps(dict(seed=s, condition=cond, budget=name, status="nan",
                                                          error=str(e), log=tr.log), indent=2))
            log(f"{job} {cond} s{s}: S3 NaN, team counted as failed: {e}")
            tr = None
        if tr is not None:
            save_team(tr, d / "team.pt", name)
            (d / "train.json").write_text(json.dumps(dict(seed=s, condition=cond, budget=name, status="ok",
                                                          updates=tr.updates, seconds=tr.seconds, log=tr.log), indent=2))
            g = EV.gate_eval(tr.team, s)
            g.update(EV.gates_pass(g, cond))
            (d / "gate.json").write_text(json.dumps(g, indent=2))
            res = EV.run_team(d, s, cond)
            done.append(time.perf_counter() - t0)
            log(f"{job} {cond} s{s}: train {tr.seconds:.1f}s gate C {g['success_C']:.4f} Z {g['success_Z']:.4f} "
                f"pass {g['passed']} eval {res['seconds']['total']:.1f}s V2 {res['V2']['passed']} V3 {res['V3']['passed']} "
                f"team {done[-1]:.1f}s")
        else:
            done.append(time.perf_counter() - t0)
        elapsed = time.perf_counter() - T0
        if len(done) == 3:
            projected = elapsed / 3 * len(seeds)
            log(f"{job} S2 check: projected {projected/60:.1f} min for {len(seeds)} teams")
            if projected > S2_MIN * 60 and i + 1 < len(seeds):
                return stop(job, f"S2: projected job time {projected/60:.1f} min > {S2_MIN} min after 3 teams", T0)
        if i + 1 < len(seeds) and elapsed + sum(done) / len(done) > SELF_CAP_MIN * 60:
            return stop(job, f"self cap: {elapsed/60:.1f} min used, next team would pass {SELF_CAP_MIN} min", T0)
    print("END " + json.dumps(dict(status="complete", job=job, wall_seconds=time.perf_counter() - T0)), flush=True)
    return 0


def stop(job, reason, T0):
    (RUNS / f"{job}_STOP.json").write_text(json.dumps(dict(job=job, reason=reason), indent=2))
    print("END " + json.dumps(dict(status="stopped", job=job, reason=reason,
                                   wall_seconds=time.perf_counter() - T0)), flush=True)
    return 4


def job_score(seeds, job):
    T0 = time.perf_counter()
    times = []
    for i, s in enumerate(seeds):
        d = team_dir("W1", s)
        if (d / "scores.npz").exists():
            log(f"{job} s{s}: scores exist, skip")
            continue
        if not (d / "monitor.npz").exists():
            log(f"{job} s{s}: no monitor.npz (failed team), skip")
            continue
        t0 = time.perf_counter()
        r = MON.score_team(d)
        times.append(time.perf_counter() - t0)
        log(f"{job} s{s}: " + json.dumps(r))
        elapsed = time.perf_counter() - T0
        if i + 1 < len(seeds) and elapsed + sum(times) / len(times) > SELF_CAP_MIN * 60:
            return stop(job, f"self cap: {elapsed/60:.1f} min used", T0)
    print("END " + json.dumps(dict(status="complete", job=job, wall_seconds=time.perf_counter() - T0)), flush=True)
    return 0


def chunks(k, n=20):
    return [f"{a}-{min(a + k - 1, n)}" for a in range(1, n + 1, k)]


def job_plan():
    """The main-run job list from the J0 plan (PREREGISTRATION.md section 11)."""
    selected_budget()
    plan = json.loads((RUNS / "j0" / "decision.json").read_text())["plan"]
    lines = []
    for job, cond, mode in [("J1", "W1", "teams"), ("J2", "W0", "teams"), ("J3", None, "score")]:
        k = plan[job]["teams_per_job"]
        if k is None:
            raise SystemExit(f"REFUSED: no plan size fits {PLAN_MIN} min for {job}")
        parts = chunks(k)
        for i, seeds in enumerate(parts):
            name = job + (chr(ord("a") + i) if len(parts) > 1 else "")
            extra = f" --cond {cond}" if cond else ""
            lines.append(f"{name} jobs.py {mode}{extra} --seeds {seeds}")
    lines.append("J3_analysis analysis.py")
    lines.append("J3_check stats_check.py")
    print("\n".join(lines))
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["j0", "teams", "score", "plan"])
    p.add_argument("--cond", choices=["W1", "W0"])
    p.add_argument("--seeds", default="1-20")
    p.add_argument("--job", default="")
    a = p.parse_args()
    if a.mode == "plan":
        return job_plan()
    log(f"start mode={a.mode} cond={a.cond} seeds={a.seeds} job={a.job} torch={torch.__version__}")
    if a.mode == "j0":
        return job_j0()
    if a.mode == "teams":
        return job_teams(a.cond, parse_seeds(a.seeds), a.job)
    return job_score(parse_seeds(a.seeds), a.job)


if __name__ == "__main__":
    sys.exit(main())
