# Nathan reply 2 — FINAL draft for Simon's review and send
Applies: S-136 (no absence claims), writing constitution Laws 1-18,
simon-voice-gate, this-week's corrections (em-dash ban, no triads, no
consultant closers, no "the kind of work", no PR-speak, stress positions).
Spec-dec disclosure SHIPPED in v0.1.1 — message reflects that honestly.

Route: GitHub comment on issue #13. Simon sends it.

---

thanks Nathan! appreciate the pointer to the central repo. will watch for
the vulkan work landing there.

and thanks for the llama-benchy reference, hadn't seen it. you're right
that spec-dec is hard to measure well. my tool measures a live server
(prefill and decode with true-token sizing and ambient noise controls).
it doesn't isolate spec-dec configs, but it does record whatever
spec/draft fields the server exposes in /props and says so in the output
when the server build doesn't expose them. decode numbers are end-to-end,
so spec-dec gains are included if you run a draft. linked llama-benchy in
the readme for anyone who wants to capture spec-dec properly.

it'll be public soon. i'll drop the link here when it goes up.

---

## Law compliance audit (own pass, pre-judge)

| Law | Check |
|---|---|
| S-136 absence claims | CLEAN: describes what the tool does/doesn't do, never what "nobody" or "the ecosystem" lacks |
| Writing constitution L1 (old-before-new) | His repo first, then his benchy pointer, then our tool's spec-dec state |
| L2 (stress position) | "end-to-end" and "included if you run a draft" land at sentence ends |
| L3 (characters-as-subjects) | "my tool measures", "it doesn't isolate", "decode numbers are" |
| L11 (echo key terms) | "spec-dec" used consistently, never "speculative" then "drafting" then "spec" |
| L4/L17 (no meta-discourse) | No "I wanted to reach out" or "I thought I'd share" |
| simon-voice-gate | Contractions (hadn't, doesn't, it'll, i'll), lowercase register, no em-dashes, no triads, no "the kind of work", no consultant closer, no "wanted you to hear" |
| Publish-gate honesty | "it'll be public soon" — true and the only promise made |
| Prior-art credit | llama-benchy named + credited in our README (the message says so) |
| No absence claims about other tools | CLEAN: describes our tool's boundary, doesn't claim benchy/llama-bench "can't" do something |
