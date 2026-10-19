"""Non-overlapping recorded job wall times, explicitly not processor CPU-hours."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(relative):
    return json.loads((ROOT / relative).read_text())


def main():
    destination = ROOT / "compute_ledger.json"
    assert not destination.exists()
    jobs = [
        {"name": name, "elapsed_s": read(path)["elapsed_s"], "source": path}
        for name, path in [("tiny_sender_capture", "prepare_tiny_completion.json"),
                           ("tiny_training", "train_tiny_completion.json"),
                           ("tiny_strong_text", "text_tiny/summary.json")]
    ]
    for path in ["development_job/state.json", "confirmation_job/state.json"]:
        state = read(path)
        assert state["status"] == "complete"
        for stage in state["stages"]:
            assert stage["status"] == "complete" and stage["exit_code"] == 0
            jobs.append({"name": stage["name"], "elapsed_s": stage["elapsed_s"], "source": path})
    primary_s = sum(j["elapsed_s"] for j in jobs)
    for name, path in [("sender_reproduction", "sender_reproduction/verification.json"),
                       ("bridge_component_profile", "bridge_profile.json")]:
        jobs.append({"name": name, "elapsed_s": read(path)["elapsed_s"], "source": path})
    total = sum(j["elapsed_s"] for j in jobs)
    text = read("text_final/sender.json")
    receiver = read("text_final/receiver.json")
    result = {"jobs": jobs, "v2_primary_job_wall_s": primary_s,
        "v2_recorded_job_wall_s": total, "v2_recorded_job_wall_h": total/3600,
        "v1_prior_reported_components_s": 838.13,
        "v1_prior_reported_download_s_approx": 374,
        "v1_v2_recorded_components_plus_approx_download_h": (total+838.13+374)/3600,
        "accounting": "Sum stage times, not parent waiting time or nested per-condition times. Tiny text excludes CLI startup. V1 values are the sealed report's rounded subtotal, not a new independent remeasurement.",
        "exclusions": "Unlogged orchestration, some audits/tests/sealing, initial failed V1 pre-training exporter and the older Arabic branch. Processor CPU-time, electricity, assistant-service billing and deployment latency are not measured.",
        "text_sender_prompt_token_lengths": sorted({len(r["input_ids"]) for r in text["outputs"]}),
        "text_receiver_prompt_token_lengths": sorted({len(r["input_ids"]) for r in receiver["outputs"]}),
        "text_payload_utf8_bytes": 1, "latent_payload_bytes": 2048,
        "external_cloud_spend_usd": 0, "budget_ceiling_usd_approx": 30,
        "cloud_resources_created": 0}
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "jobs"}))


if __name__ == "__main__":
    main()
