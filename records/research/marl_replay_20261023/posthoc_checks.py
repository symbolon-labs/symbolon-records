"""Post hoc checks (NOT preregistered; written after the run). Writes posthoc.json.

Question: why does the context monitor X-kNN give replays LOWER scores than benign
episodes (AUROC below 0.5 on RE)? Check, for each W1 team on the monitor test set:
1. the fraction of steps with an active local view in C, RT and RE;
2. the mean X-kNN step score on view-active and view-empty steps of C;
3. the fraction of episodes whose maximum X-kNN step is a view-active step;
4. for RE5: the step of the maximum X-PCA score (the splice is at step index 5).
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
out = {"label": "post hoc, not preregistered", "teams": {}}
for s in range(1, 21):
    d = ROOT / "runs" / f"W1_s{s:02d}"
    mz, sc = np.load(d / "monitor.npz"), np.load(d / "scores.npz")
    t = {}
    for arm in ["C", "RT", "RE"]:
        active = mz[f"test_{arm}__view"][..., 2] > 0.5  # (1000, 30)
        step = sc[f"test__X-kNN__{arm}__step"]
        argmax = step.argmax(1)
        t[arm] = dict(frac_view_active=float(active.mean()),
                      frac_episode_max_on_active_view=float(active[np.arange(len(argmax)), argmax].mean()),
                      mean_step_score_active=float(step[active].mean()) if active.any() else None,
                      mean_step_score_empty=float(step[~active].mean()))
    pca = sc["test__X-PCA__RE5__step"]
    t["RE5_xpca_argmax_step_index_frac_eq5"] = float((pca.argmax(1) == 5).mean())
    out["teams"][s] = t


def mean(path):
    vals = [v for v in (path(t) for t in out["teams"].values()) if v is not None]
    return dict(mean=float(np.mean(vals)), min=float(np.min(vals)), max=float(np.max(vals)), n=len(vals))


out["summary"] = {f"{arm}.{k}": mean(lambda t, a=arm, k=k: t[a][k]) for arm in ["C", "RT", "RE"]
                  for k in ["frac_view_active", "frac_episode_max_on_active_view", "mean_step_score_active",
                            "mean_step_score_empty"]}
out["summary"]["RE5_xpca_argmax_step_index_frac_eq5"] = mean(lambda t: t["RE5_xpca_argmax_step_index_frac_eq5"])
(ROOT / "posthoc.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out["summary"], indent=2))
