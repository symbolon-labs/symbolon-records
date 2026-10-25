"""EXPLORATORY. Complement basis Q (960 x 896) of the d0 output column space (PLAN.md section B, 4.2).

CPU only, no model. Writes complement.pt and COMPLEMENT.json in this folder (refuses to overwrite).
"""
import hashlib, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
import torch

ROOT = Path(__file__).resolve().parent
R = ROOT.parent
CANDIDATE = R / "latent_binding_shared_id_query_mechanics_20260908/run/candidate.pt"
D0_SHA = "d0f16cb9705ba53daf6cc8f2bfad03a08efa96cbb4f77dd836c9533e948cba87"


def sha(p):
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    assert sha(CANDIDATE) == D0_SHA
    st = torch.load(CANDIDATE, map_location="cpu", weights_only=True)
    alpha = float(st["reader.gate"].double() * st["reader.norm_scale"].double())
    A = alpha * st["reader.output.weight"].double()
    U, S, Vh = torch.linalg.svd(A, full_matrices=True)
    assert U.shape == (960, 960) and S.shape == (64,)
    Q = U[:, 64:].contiguous()
    out = ROOT / "complement.pt"
    assert not out.exists()
    torch.save({"Q": Q, "alpha": alpha, "candidate_sha256": D0_SHA}, out)
    info = {"label": "EXPLORATORY", "shape": list(Q.shape), "alpha": alpha, "sha256": sha(out),
            "qtq_minus_identity_max_abs": float((Q.T @ Q - torch.eye(896, dtype=torch.float64)).abs().max()),
            "column_space_times_q_max_abs": float((U[:, :64].T @ Q).abs().max())}
    with open(ROOT / "COMPLEMENT.json", "x") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
