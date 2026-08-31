# Experiment protocol: pre-mortem and post-mortem for every phase
## CEO standing order (2026-08-31): "build in a pre and a post mortem into
## every single phase of every single one. every time anything fails assume
## user error or AI error. or some process artifact."

## Pre-mortem (before ANY GPU boot or measurement)

Answer these IN WRITING before the command runs:

| # | Question | Answer |
|---|---|---|
| P1 | What exactly will this boot/process do? (binary, flags, memory budget) | |
| P2 | What could MY COMMAND get wrong? (flags, paths, quoting, env vars) | |
| P3 | What could the SCRIPT get wrong? (variable expansion, shell quoting, unit bugs) | |
| P4 | What memory will this allocate? What else is resident? Will it fit with headroom? | |
| P5 | What is the expected signal? What does a NULL/empty result look like vs success? | |
| P6 | If this fails, what is the FIRST command I run to diagnose? | |
| P7 | What could I have measured WRONG last time that makes this run misleading? | |

## Post-mortem (after every phase, pass OR fail)

| # | Question | Answer |
|---|---|---|
| M1 | Did it do what P1 said it would? If not, what diverged? | |
| M2 | If it failed: was it user error (my command), AI error (my reasoning), or a process artifact (quoting, env, script bug)? Name it. | |
| M3 | What did the INSTRUMENT report vs what actually happened? Any discrepancy? | |
| M4 | What memory was actually used vs the P4 plan? | |
| M5 | What would I do differently on the re-run? | |
| M6 | Is there a new law here? (If yes, bank it before the next boot) | |

## The failure-assumption hierarchy (always in this order)
1. **MY command was wrong** (typo, wrong flag, wrong path, wrong quoting)
2. **MY reasoning was wrong** (misunderstood the tool, wrong assumption)
3. **A script/process artifact** (shell quoting, variable expansion, unit bug, environment leakage)
4. **Infrastructure** (OOM, thermal, network) — LAST, not first, because
   blaming infrastructure first is how user errors hide

## Wall-time sanity checks (cheap failure detection)
- A battery taking < 5% of expected wall time = investigate immediately
- Empty summary buckets = the battery never ran real requests
- Zero errors + zero data = the tool didn't connect or the target didn't exist
