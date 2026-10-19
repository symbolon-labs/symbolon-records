# Frozen-reader randomized binding gate V1

Written before new binding-task model inference, data materialization, or
optimization. Prior synthetic two-bit Qwen→SmolLM2 PASS remains sealed. This
is one harder task with the same architecture, not another bridge search.

## Hypothesis and exact interface

Frozen Qwen/Qwen3-1.7B revision70d244cc86ccca08cf5af4e1e306ecf908b1ad5e,
block13 output → the exact previous bounded resampler/projector/reader classes
→ frozen HuggingFaceTB/SmolLM2-360M-Instruct revision
a10cc1512eabd3dde888204e902eca88bddb4951, reader after block15.
Both indices zero-based.8×64 float32 transport, per-slot RMS1/L2norm8,
569,857 trainable parameters, one width64 reader, both attentions2*cosine(q,k),
gate initialized.01. Fresh bridge initialization, not the prior trained weights.
All endpoint/embedding/LM-head/tokenizer parameters frozen. No identity lane,
anchors, curriculum, extra depth, alternative layer or widened bottleneck.
Native eager float32 CPU, deterministic,4 intra-op/1 inter-op threads.

Source contains four lines `<ID> = <color>`, with a random bijection between
four distinct IDs and red/green/blue/yellow and randomized row order. Source
contains no receiver query. Receiver sees exactly `Which ID is <color>?`.
Each table is evaluated with all four queries, sharing one source memory.
No raw source, record IDs, candidate shortlist or target metadata enters Q.

Alphabet: all128 IDs from A…P × digits0…7. Pool seed120501 independently
permutes letters/digits; parity of permuted letter-index plus digit-index
splits64 TRAIN/DEV IDs from64 FINAL-novel IDs. Each pool covers all components,
four digits per letter/eight letters per digit. Novelty means whole ID strings
absent from bridge-training sources/targets, not unseen component tokens or
absence from endpoint pretraining. All128 answer candidates are fixed globally;
no per-record shortlist, including during scoring. Final pool membership is
metadata fixed now, but final tables/queries/outcomes are not generated now.

Pre-inference tokenizer-only audit found both models use two tokens per ID
(with leading space on plain latent Q), queries5 tokens, sampled source19.
Audit all materialized inputs; fail on incompatible tokenization/padding rather
than silently changing IDs/prompts. Native chat text uses bare ID continuations.

## Data and lexical boundary

Tiny seed121001:8 random tables/32 queries;32 distinct TRAIN-pool IDs sampled
without replacement. This is deliberate overfit only, not balanced coverage
of all64 known identities.
TRAIN seed122001:256 tables/1,024 queries;4 balanced rounds of64 tables.
DEV seed123001:64 tables/256 queries;1 balanced round, same known-ID pool.
FINAL-known seed124001:64 tables/256 queries; same ID pool, new bindings.
FINAL-novel seed125001:128 tables/512 queries;2 rounds using only reserved IDs.

Each balanced round assigns each pool ID once to each queried color across64
tables, rejecting same-ID collisions within a table and duplicate bindings
across partitions. Row order is randomized. Therefore Q alone predicts at most
1/64 on each balanced arm for any deterministic fixed answer per color (possibly
0 if its preferred answer lies outside that arm's pool); uniform128-way chance
is1/128. Do not call a0% baseline a bug or claim controls are uniformly calibrated.

Later data not instantiated before tiny gate. FINAL-known/novel created only
after selected DEV rerun, strong DEV text check and final code/checkpoint freeze.
All observed final outputs burned for development. No architecture tuning on
FINAL; no replacement checkpoint if the selected one fails readiness.

## Two content substitutions and other controls

Every evaluation includes receiver-only, correct, shuffled, wrong-binding,
random71001,random71002,zero. Mandatory actual sender→compact text is evaluated
on tiny before optimization, DEV before final freeze, and both final arms.
An oracle compact-text sanity arm is included for interpreting text mechanics.

SHUFFLED: label-blind table derangement seed61001, receiver query/color fixed
literally; score both original target and donor's binding answer. The donor
comes from the same evaluated pool. Correct and shuffled are actual native
receiver runs, not merely a permuted reported accuracy.
WRONG-BINDING: same four IDs and row order, but colors cyclically permuted by a
seeded nonzero offset; capture this real counterfactual source through sender.
All queried answers change while available identities remain fixed. Report
original-target accuracy and counterfactual donor-answer following separately.
This tests binding specificity rather than absence/presence of an identity.
RANDOM: independent Gaussian transport per table/seed, slot-norm-matched to
correct, shared across that table's four queries. No claim of postprojection
or residual-energy matching. ZERO: actual enabled reader with zero memory.

Correct/shuffle/wrong/random all traverse the same reader. Keep response norms,
attention entropy, per-example ID scores, donor joins and both tokens' scores.
Never inject a decoded sender ID into the latent path. Metadata is used only
for training loss, scoring and controlled source construction.

## Sequence scoring and mechanical gate

Primary ID score = sum of full-vocabulary log probabilities of both ID tokens,
no length normalization. Fixed128-ID ranking. Shared first-token prefixes may
be scored once per unique prefix using the native model (16 branches), not128
duplicated forwards. No KV-cache/parallel inference implementation is added.
Full direct candidate scoring is the oracle: require all labels' score agreement
within1e-4, identical argmax, and within-run deterministic repeats bit-exact.
Stock/disabled/zero logits must be bit-exact for identical full input sequences.
The small tolerance concerns different sequence shapes in prefix reuse only;
do not change it after seeing comparisons. Score both tokens, test candidate
ordering, same-query donor joins, and zero parity. Any mismatch stops training.

Teacher-forced training runs Q+first gold ID token and full-vocabulary CE at
the final two predictive positions, summed over the two answer tokens and
averaged over examples. This is ordinary complete-answer NLL, not a new loss.
Endpoint weights are frozen but autograd traverses the remaining receiver
layers to the bridge. Candidate prefixes are output likelihood conditioning,
not source-specific receiver input or an identity channel.

## Fixed training and readiness

Same optimizer: AdamW lr.003,weight_decay0,clip1,no scheduler/auxiliary loss.
Tiny init51001,minibatches51002,batch8,max300,checks20.
Full fresh init52001,minibatches52002,batch16,max600,DEV50; early stop after
three checks without>=.001 improvement in best mean full-ID DEV NLL. Select
minimum trained DEV NLL, earliest tie, independently re-derived from saved
correct-label probabilities. Do not increase steps or change optimization after
a failure. Numerical batch changes need a documented mechanical necessity and
fresh preregistration before optimization, not a performance-driven retry.

Before tiny optimization: native oracle compact-text accuracy>=90% on all32
tiny queries and every color>=75%, actual sender binding text reported, full-ID
scoring validation, finite nonzero bridge gradients, frozen endpoint checks.
Failure of oracle sanity or mechanics stops this experiment. No prompt search.
Actual sender text failure is reported, not silently replaced by oracle.

Tiny gate: correct>=95%, shuffle/random original-target accuracies<=25%,
shuffled donor following>=95%, zero exact to baseline, baseline<=12.5%, actual
rerun exact and frozen hashes. Wrong-binding following is a diagnostic at tiny
(untrained counterfactuals), not required to make a deliberate overfit pass.
No low-control gate counts unless correct overfit passes.

DEV readiness: correct>=90%, each color>=85%, randomized and wrong-binding
original-target accuracies<=5%, zero exact, baseline<=1/64; shuffled and
wrong-binding donor following>=95%. Strong oracle text>=90%, actual sender text
fully reported; selected checkpoint must actually reproduce. Failure stops
before FINAL. No novel-ID DEV outcomes are inspected to tune the bridge.

## Final pass rule (frozen before training)

On BOTH FINAL-known and FINAL-novel separately:
correct>=90%, each queried color>=85%; shuffled/random/wrong-binding original
accuracy<=5%; baseline<=1/64 and zero exact; paired99% lower bound for
correct-minus-every destructive control >+.75. Shuffled/random minus baseline
paired95% intervals inside±.05. Shuffled and wrong-binding donor following>=95%,
with paired group-bootstrap95% lower bounds>=.90. Exact cross-record shuffled
probability vectors equal matching donor-correct vectors for identical Q.
10,000 table-group resamples, seed81001, all four queries and arms paired.
Report per-ID as well as per-color performance; no per-ID subgroup rescue.

Require deterministic actual first8-table sender original/counterfactual state
rerun and first32-query receiver rerun with ORIGINAL full-set control memories,
frozen endpoints/code/checkpoint, rebuilt donors/memories and independent hashes.
Complete actual sender-text/oracle comparison. No generation or text-efficiency
advantage required/claimed. Later capacity/anchoring/curriculum changes are not
automatically run if the gate fails; diagnose and preserve this attempt first.

Possible scoped outcomes: NOVEL BINDING PASS; KNOWN BINDING NO-GO;
NOVEL IDENTITY NO-GO; BINDING-COUNTERFACTUAL NO-GO; MECHANICS/TRAINING NO-GO.
Do not erase the earlier two-bit PASS or imply all latent channels fail.

## Compute and provenance

Local only before mechanics and tiny overfit/shuffle pass.128 candidates require
16 prefix branches: anticipate substantially more scoring time than the one-
symbol task. Source capture is short (19 tokens); receiver query5, training6.
Keep batch sizes unless actual memory mechanics require otherwise. No cloud
provisioning now; external spendUSD0, original approximatelyUSD30 ceiling remains.
Record actual elapsed times, RSS, code/models/tokenizers, bridge hashes, input
and candidate IDs, prompt/split hashes and raw scores. No accuracy-based runtime
changes. Initial estimate: text/mechanics15–30min, tiny20–60min; full/final costs
must be estimated from observed tiny timing before scaling. No idle instances.

No historical sealed tree, core Rust inference semantics, Arabic task, relational
reasoning task, identity anchors, curriculum or wider architecture is modified.
