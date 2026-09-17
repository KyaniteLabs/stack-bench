# Qwen4 quant design — from BF16 source on disk
CS-9, 2026-08-31. The Qwen4-beta was dismissed as "MARGINAL" because the
Q2 quant had a ~16k context ceiling. The MODEL itself measured 331/294/248
t/s prefill at 8/16/32k (q4v2 experiment) — 40-55% faster than the stock
champion. We have the BF16 source (54G) and Q8_0 (29G) on disk.

## Why this matters
- Fastest prefill we've ever measured on this hardware
- The context ceiling was a QUANT artifact (Q2), not a model limitation
- A Q4/Q5 quant from BF16 source could be a legitimate floor challenger
- The Q8_0 is already on disk as a reference for perplexity comparison

## Design (pre-registered before any cut)
### Arms (closed set)
- A1: imatrix Q4_K_M from BF16 (imatrix from BF16, wiki corpus) — the speed challenger
- A2: Q8_0 (already on disk — reference, no new cut)
- A3: stock champion (reference)

(Q5_K_M cut per CEO challenge: this is a speed experiment, Q5 sits in the middle where nobody needs it)

### Measurement plan
- stack-bench prefill at 8/16/32k + 64k (the context question is key)
- Decode t/s
- Context ceiling test: can it serve at 32k? 64k? 128k?
  (The Q2 died at 32k due to memory. Q4 at ~15G weights + KV should fit.)
- Perplexity on wiki.test.raw (deterministic, same corpus as lab-1)

### Safety (S-139/140)
- One arm boot at a time, memory-planned
- Headroom gate before every boot
- OOM-protected floor units
- Governor authority confirmed

### Success criteria (pre-registered)
- Prefill ≥ 250 t/s at 8k (faster than stock champion's 235)
- Context ceiling ≥ 64k without OOM (the Q2 blocker)
- Perplexity within 0.1 of the Q8_0 reference
- Decode ≥ 12 t/s

## What this ISN'T
- Not a floor swap proposal (that's CEO-gated)
- Not a quant-quality lab (that was cancelled — this is a targeted
  floor-challenger assessment)
- Not using the community-favorite criterion (Qwen4 is OUR pick based
  on measured speed, not community adoption)

## Execution plan
1. Compute imatrix from BF16 on wiki corpus (CPU-only, no GPU risk)
2. Cut Q4_K_M and Q5_K_M using the imatrix (CPU-only)
3. Boot each arm, measure prefill + decode + context ceiling
4. Compare against stock champion + Q8_0 reference
5. If any arm passes success criteria → floor-challenger recommendation

## Estimated GPU time
- Imatrix: CPU-only, ~4-14h (overnight)
- Quantize: CPU-only, ~2h per quant
- Measurement: 2-3 research windows, ~2h total GPU
- Total: ~1 evening of GPU time + overnight CPU

## Dependencies
- BF16 source: ON DISK (54G)
- Q8_0 reference: ON DISK (29G)
- Wiki corpus: ON DISK (558KB)
- Tool: llama-quantize + llama-imatrix (BUILT, both trees)
- All verified present
