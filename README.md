# stack-bench

**One-command honest local-inference benchmark for llama-server.** Measures prefill and decode throughput against any llama-server-compatible endpoint with true-token prompt sizing, ambient-noise controls, and speculative-decoding state disclosure. Zero dependencies, Python 3.10+, MIT license.

<!-- Agent metadata
tool: stack-bench
version: 0.1.1
language: python
dependencies: none (stdlib only)
license: MIT
compatibility: llama-server / any OpenAI-completion-compatible endpoint with /tokenize
measures: prefill_tps, decode_tps, ambient_noise, spec_dec_state
output: JSON (summary + rows.jsonl)
tests: 19, offline, no network required
-->


One-command honest local-inference battery: measures prefill and decode
throughput against any llama-server-compatible endpoint, with true-token
prompt sizing and an ambient-noise control, emitting machine-readable JSON.

Zero dependencies. Python 3.10+, stdlib only.

## Quickstart

```sh
python3 stack_bench.py --selftest                 # offline, built-in mock server (~10s)
python3 stack_bench.py --url http://127.0.0.1:8080 --name myserver
python3 stack_bench.py --url ... --profile quick  # 8k/16k only
```

Output lands in `./stack-bench-out/<UTC-timestamp>-<name>/`:
`summary.json` (schema v1: env, server props, per-battery summaries, ambient
block) and `rows.jsonl` (every request row, including error rows with HTTP
status and server body).

## What it measures

- **Speculative decoding disclosure:** decode t/s is END-TO-END
  server-visible throughput. If your server runs a draft model
  (speculative decoding / MTP), the number INCLUDES those gains — this
  tool does not isolate or control spec-dec. Summaries record whatever
  spec/draft fields the server exposes in `/props`; when the server
  build does not expose them, the summary says so explicitly. For
  spec-dec-focused benchmarking see
  [llama-benchy](https://github.com/eugr/llama-benchy).

Not measured (v0.1 boundaries): concurrency (single sequential client),
TTFT, quality/accuracy, spec-dec isolation, cold-start/warmup effects.

- **Prefill**: prompts grown until the server's own `/tokenize` reports the
  target token count (8k/16k/32k default), 3 reps with DISTINCT prompt texts
  per rep (reps cannot be flattered by cache reuse), `cache_prompt: false`,
  summary bucketed by MEASURED `prompt_n` — mean/min/max per bucket.
- **Decode**: fixed prompt (`--decode-ctx`, default 2048 tokens),
  `n_predict 128`, `ignore_eos: true`
  (the in-tree `llama-bench` tg convention — adopted from llama.cpp, not
  invented here), per-row `predicted_n` with `early_eos` flagging.
- **Ambient A/B control**: each probe shape issued twice back-to-back;
  reports content-hash equality and timing deltas, because same-boot reruns
  at temperature 0 are not guaranteed bit-identical on real stacks
  (llama.cpp uses different kernels at different batch sizes — measured
  14/23 identical on our own deployment, 2026-08-30).

## Runtime expectations and comparing runs

A quick run (8k/16k, 2 reps + decode + ambient) took ~9 minutes on the
27B reference box; deeper targets scale with your prefill t/s (a 131k
prompt at 150 t/s is ~15 minutes per rep — raise --timeout-completion
accordingly). To compare two configurations, run the tool against each
and diff the `summary.json` files (bucketed means + ambient pairs make
the comparison direct); raw per-request evidence is in `rows.jsonl`.

## Compatibility

Any llama-server-compatible endpoint exposing `/tokenize` + `/completion`
(timing fields in responses); `/props` is captured when present and
gracefully absent otherwise. Without `/tokenize`, the tool falls back to
flagged word-approximation sizing.

## Exit codes

0 = clean run / 1 = hard failure (every request of the prefill or decode
battery failed, or a
battery produced zero usable measurements - a server returning 200 with
garbage exits 1, never 0-with-empty-summary) - 2 = usage error (bad
--reps/--targets/--n-predict).

## Timeouts

`--timeout-tokenize` (default 600 s) and `--timeout-completion` (default
1800 s, sized for 32k prefills) bound SOCKET operations, not total wall
time. A hung endpoint costs one tokenize timeout per run at most: after
an initial `/tokenize` failure, prompt growth stops calling `/tokenize` and
falls through to the flagged word-approx mode instead of retrying into the
hang.

## Sample output

```json
{"schema_version": 1, "prefill": {"by_measured_prompt_n": {"8000":
  {"mean_tps": 223.9, "min": 219.5, "max": 228.4, "n": 2}}}, "decode":
  {"decode_tps": {"mean_tps": 17.6, "min": 12.8, "max": 22.3, "n": 2}}}
```

## Deltas vs `llama-bench` (named prior art, in-tree microbench)

Each delta cites its measurement source:

| Delta | Source |
|---|---|
| Server-level measurement (HTTP against a live endpoint, not a lib-bound microbench) | measured on our Strix Halo box 2026-08-30, this server battery vs llama-bench runs |
| `/tokenize`-true prompt sizing (word-count sizing mislabeled ~1.75× in our own v1) | internal v1-to-v2 instrument fix, our box 2026-08-29 |
| Ambient-noise reporting | same-boot A/B reruns measured 14/23 identical, our box 2026-08-30 |
| Reps-with-ranges JSON (min/max/n per bucket, error rows preserved) | our internal experiment verdict records (not shipped) |

**Claim boundary (honesty first):** prefill measurement is a productization
of a battery with live-validated results (our Strix Halo box, 2026-08-30: production server
215/207/184 t/s at 8/16/32k). Live parity LOGGED 2026-08-30T18:51Z against
our resident production server: this tool measured 223.9/213.4 t/s at 8k/16k
(banked-battery agreement within ~6-9 t/s on means, 3-4%); the decode battery ran
live (mean 17.6 t/s, range 12.8-22.3), and the ambient pair minutes later
read 7.5-10.9 t/s with hash-unequal outputs — a ~2.4x spread the tool
caught and we report verbatim rather than smooth away; treat decode
figures as end-to-end and load-state-dependent (spec-dec disclosure
above). Artifacts: results/LIVE-PARITY-20260830.json (summary; url/argv
sanitized) and results/LIVE-PARITY-20260830-rows.jsonl (the run's raw
rows, unedited).

## Credits

- The PROVENANCE/capture-date/unvendored-numbers conventions this tool
  follows are adopted from
  [Nathanw1014/strix-halo-llamacpp](https://github.com/Nathanw1014/strix-halo-llamacpp)
  (MIT) — its benchmarks/BENCHMARKS.md is the prior art for context-depth
  prefill sweeps (0–64k, multi-GPU).
- The `ignore_eos` decode convention is adopted from `llama-bench`
  (ggml-org/llama.cpp, MIT).

## Tests

```sh
python3 -m unittest discover tests
```

19 tests, all offline (mock server; no network, no model needed).

## License

MIT — see LICENSE. Measurement outputs you produce with the tool are yours.
