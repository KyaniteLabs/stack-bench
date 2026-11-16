# stack-bench

A Python command-line instrument for operators comparing local llama-server inference configurations. It measures server-reported prefill/decode throughput and paired-request variability, and saves inspectable JSON rather than presenting one unlabeled speed number. It does not serve models or decide whether their answers are correct.

## Core operator experience

Run `python3 stack_bench.py --selftest` for the built-in local HTTP mock, or supply a real `--url` and run name. Choose quick/full targets, repetitions, decode context/output length and socket timeouts. Each run creates a no-clobber timestamped directory containing `summary.json` and per-request `rows.jsonl`; inspect errors and sizing flags before comparing runs.

## Existing capabilities and data

- `/tokenize`-based prompt growth with seeded, distinct prefill prompts per repetition; a run-wide flagged word-approximation fallback if tokenization fails.
- Sequential prefill/decode batteries, measured-prompt-size buckets, means/ranges/sample counts, missing-timing and early-stop flags.
- Same-request ambient A/B pairs with output hashes and timing deltas. This reports variability; it does not control ambient load or explain the cause.
- Optional `/props` capture and explicit disclosure when speculative-decoding fields are absent. Decode numbers remain end-to-end server-visible throughput, not isolated base-model speed.
- HTTP/socket error rows, configurable authentication, argument bounds, machine-readable output, built-in mock, and an offline test suite.
- [results](results/) retains a historical live-parity summary/rows and two HQ64k comparison summaries. They are experiment artifacts, not a general dataset or fresh proof of current hardware. Workload text is generated from an embedded word list in [stack_bench.py](stack_bench.py).

There is no GUI, MCP surface, concurrency benchmark, TTFT/quality evaluation, model installation, or speculative-decoding isolation. The endpoint contract is `/tokenize` and `/completion` with llama-server timing fields, not arbitrary OpenAI chat compatibility.

## Honesty and acceptance

Historical performance figures in [README.md](README.md) include corrections and regime caveats; do not reuse them as universal claims. Selftest uses deterministic mock timings and demonstrates instrument flow only. Real benchmark acceptance requires resource-policy checks, the [experiment protocol](EXPERIMENT-PROTOCOL.md), raw artifacts and operator-side evidence. No new model benchmark is claimed by these docs.
