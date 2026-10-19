# Local development resource gate

Local Gates0–2 passed at tiny step160: correct32/32; shuffle10/32;
random8/32 and6/32; zero/baseline8/32. Actual all-condition deterministic rerun
matched exactly; receiver parameter SHA256 unchanged. The fixed instruction-
formatted text sender/receiver and oracle controls scored32/32, while missing-
source chat scored8/32. These are mechanical/calibration results only.

Measured tiny sender extraction: ~24s including startup/hash/load for32 records.
Measured receiver batch8 train step: approximately1.3–1.5s; each32-item latent
control evaluation approximately3–4s, text relay approximately8–11s. Extrapolated
TRAIN1024+DEV256 sender collection: approximately12–16min; full600-step batch16
training: approximately25–35min; repeated five-arm DEV evaluation plus fixed
baseline/text: approximately25–40min. Total development envelope approximately
60–95min, less if registered early stopping triggers. Estimates are not latency
benchmarks; thermal drift and load affect this CPU.

Hardware: i5-1135G7,4cores/8threads,16GiB physical RAM, no CUDA device; ~14GiB
available before work. Sender float32 weights ~6.9GB plus runtime and at most
~0.4GB accumulated/copying cached states. Receiver float32 weights ~2.4GB plus
autograd, cached TRAIN/DEV states and runtime; estimate <5GB total. Models are
loaded sequentially, not simultaneously. Record actual child-process peak RSS.

Use local CPU,4 PyTorch threads, no cloud provisioning. External spend remains
USD0 of cumulative approximatelyUSD30 ceiling. This run keeps the registered
data sizes, optimizer, step ceiling, architecture, scoring and DEV selection rule.

Development pipeline: prepare TRAIN -> prepare DEV -> fresh full training.
No automatic final-test creation or evaluation in this pipeline. Final requires
an explicit verified checkpoint/code freeze after DEV inspection. On subprocess
failure, retain all artifacts and stop; do not blindly restart partial work.

Measured update after TRAIN capture: parent-recorded peak child RSS was
12,824,012KiB (~12.23GiB), above the initial weight-plus-runtime estimate;
steady-state sampling was approximately7.6GiB. The stage completed successfully.
This higher measured peak, not the estimate, governs subsequent memory planning.
Continue strictly sequential model processes; do not launch another model
alongside the ongoing full receiver training.
