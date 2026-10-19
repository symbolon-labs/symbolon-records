# Cross-family text control

Use the identical sender and receiver instruction bodies and forced-choice
one-letter text channel from sealed V2 text_baseline.py. Sender remains Qwen3;
receiver alone changes to the registered Llama3.2 revision. Native tokenizers
render their respective chat templates. Explicit date_string="06 Sep 2026"
prevents the Llama template's calendar default from changing across reruns.
Same actual sender, oracle, and missing-source conditions, caching and exact
receiver reruns. No prompt tuning. Score joint full continuations correctly.
Report text honestly even if it equals or beats the latent channel. No generation.
