# stack-bench architecture

## Parts and data flow

All runtime logic is in [stack_bench.py](stack_bench.py), using Python's standard library. [tests/test_e2e_mock.py](tests/test_e2e_mock.py) supplies unit/HTTP mock tests; [EXPERIMENT-PROTOCOL.md](EXPERIMENT-PROTOCOL.md) governs interpreting real experiments.

```text
CLI validation -> run_all -> no-clobber output directory + environment capture
 -> optional GET /props
 -> prefill: seeded prompt growth via POST /tokenize -> POST /completion
 -> decode: sized prompt -> repeated POST /completion with ignore_eos
 -> ambient: back-to-back pairs -> content-hash/timing comparisons
 -> summaries + raw request rows -> summary.json / rows.jsonl -> exit code
```

`UrllibTransport` owns HTTP, optional Bearer authentication, JSON decoding and socket-error capture. `grow_prompt` appends seeded word chunks until tokenized length reaches the target. A shared `tokenize_dead` flag prevents repeated tokenization attempts after initial failure; approximation mode is labeled rather than disguised as true-token sizing.

`_timed_completion` requests temperature zero and `cache_prompt: false`, extracts server timing fields and a shortened SHA-256 output digest, and records missing timings/errors. Prefill uses different seeds per repetition; decode repeats one prompt. Ambient uses pairs at each prefill target plus a decode-shaped pair. Its decode prompt is 2,048-target tokens in source, independent of the main battery's `--decode-ctx` option.

Summaries bucket by measured prompt count rounded to 8,000-token bands, not simply the requested target. Rows include seeds, sizing, timing metadata, content hashes, and bounded error excerpts; they are not full model-output transcripts. `run_all` writes artifacts even for battery hard failures. Exit 1 covers an entirely failed prefill/decode battery or zero usable measurements; partial errors and ambient failures can coexist with exit 0.

## Risky paths and failure signatures

- Endpoint mismatch: `/completion` or timing fields missing can produce empty buckets even with HTTP 200. Check summary measurements, error rows and exit code, not HTTP success alone.
- Tokenizer failure: word approximation is explicitly unknown-token sizing, not comparable to an exact token target without examining actual server counts. Socket timeouts are per operation, not an overall run deadline; repeated growth and large completions may take much longer.
- Server state: request flags cannot prove caches are cold or speculation is disabled. `/props` only discloses fields the server exposes. Different depth, draft acceptance, thermal/load state, or server builds can make speed comparisons misleading. Ambient pairs are two samples, not statistical control.
- Early stop: the flag uses a tolerance (`predicted_n < n_predict - 8`) and early-stop rows still enter decode statistics. Inspect flags/counts before treating throughput as equivalent work.
- Secrets and artifacts: `env.argv` stores the process arguments without redaction. Passing `--api-key` can therefore persist a credential in `summary.json`; URL arguments and server/error metadata can also be sensitive. Do not publish artifacts without checking/redacting them; prefer an isolated non-sensitive endpoint for documentation verification. This is a documented existing risk, not a fix in this PR.
- Interrupted writes: output-directory creation avoids clobbering another run, but summary/rows are written at the end, not streamed transactionally. A crash can leave an empty or partial run directory. Absence of rows is not a successful empty experiment.

## Verification boundary

`--selftest` starts only the built-in loopback HTTP mock; its synthetic rates validate instrument plumbing, not hardware. The offline suite exercises fallback, bounds, errors, missing timing, early stop and ambient divergence. Tests contain a hardcoded `/tmp/sb-test-out` output location, so runners with stricter workspace rules need an explicitly isolated output-path harness. No inference service changes are needed for docs or mock checks. Revert the docs commit for rollback; do not delete experiment data.
