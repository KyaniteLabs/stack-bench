"""Explicit state machines: tables, legal/illegal transitions, legacy reads,
and a randomized model test. Stdlib only (the tool is zero-dependency)."""
import json
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stack_bench import (SHA_EQUAL, AmbientVerdict, IllegalTransition,
                         MEASURE_KEYS, RowOutcome, Sizing, TokenizeLatch,
                         Transport, ambient_verdict, grow_prompt, row_outcome,
                         run_ambient, sizing_of, summarize_decode,
                         summarize_prefill)

WA = "word-approx-fallback (token count unknown)"


class TestSizing(unittest.TestCase):
    def test_persisted_values_unchanged(self):
        # byte-compatibility with rows.jsonl already on disk
        self.assertEqual({s.value for s in Sizing}, {"tokenized", WA})

    def test_reader_table_exhaustive(self):
        for s in Sizing:  # every state round-trips through the reader
            self.assertIs(sizing_of({"sizing": s.value}), s)

    def test_reader_tolerates_legacy_and_unknown(self):
        self.assertIs(sizing_of({"sizing": "word-approx"}), Sizing.WORD_APPROX)
        self.assertIs(sizing_of({"sizing": "word-approx (old label)"}),
                      Sizing.WORD_APPROX)
        self.assertIsNone(sizing_of({}))
        self.assertIsNone(sizing_of({"sizing": None}))
        self.assertIsNone(sizing_of({"sizing": "something-else"}))


class TestTokenizeLatch(unittest.TestCase):
    def test_states_and_sizing_table(self):
        latch = TokenizeLatch()
        self.assertFalse(latch.dead)
        self.assertIs(latch.sizing(), Sizing.TOKENIZED)
        latch.trip()  # LIVE -> DEAD (the only legal move)
        self.assertTrue(latch.dead)
        self.assertIs(latch.sizing(), Sizing.WORD_APPROX)

    def test_dead_is_terminal(self):
        latch = TokenizeLatch()
        latch.trip()
        with self.assertRaises(IllegalTransition):
            latch.trip()
        self.assertTrue(latch.dead)

    def test_legacy_dict_state(self):
        self.assertTrue(TokenizeLatch({"tokenize_dead": True}).dead)
        self.assertFalse(TokenizeLatch({"tokenize_dead": False}).dead)
        self.assertFalse(TokenizeLatch({}).dead)
        store = {}
        TokenizeLatch.coerce(store).trip()
        self.assertEqual(store, {"tokenize_dead": True})  # same key on write
        latch = TokenizeLatch()
        self.assertIs(TokenizeLatch.coerce(latch), latch)


class ScriptedTransport(Transport):
    """/tokenize answers from a script: True = ok, False = failure."""
    def __init__(self, rng, p_fail):
        self.rng, self.p_fail, self.calls = rng, p_fail, 0

    def post(self, path, payload, timeout):
        assert path == "/tokenize"
        self.calls += 1
        if self.rng.random() < self.p_fail:
            return {"ok": False, "status": 503, "json": None, "body": ""}
        n = -(-len(payload["content"]) // 3)
        return {"ok": True, "status": 200, "json": {"tokens": [0] * n},
                "body": ""}


class TestLatchModel(unittest.TestCase):
    """Randomized model test: drive many grow_prompt calls sharing one latch
    with random /tokenize failures; check invariants after every step."""

    def test_random_sequences(self):
        for seed in range(200):
            rng = random.Random(seed)
            t = ScriptedTransport(rng, p_fail=rng.choice([0.0, 0.05, 0.3, 1.0]))
            latch = TokenizeLatch()
            model_dead = False
            for step in range(rng.randint(1, 8)):
                calls_before = t.calls
                target = rng.choice([512, 800, 2000])
                _, n, mode = grow_prompt(t, target, step, 1, latch)
                # once dead, /tokenize is never called again
                if model_dead:
                    self.assertEqual(t.calls, calls_before, (seed, step))
                model_dead = model_dead or latch.dead
                # monotone: the latch never revives
                self.assertEqual(latch.dead, model_dead)
                # result sizing agrees with the latch state; n iff tokenized
                self.assertIs(sizing_of({"sizing": mode}), latch.sizing())
                if mode == Sizing.TOKENIZED.value:
                    self.assertGreaterEqual(n, target)
                else:
                    self.assertIsNone(n)


class TestRowOutcome(unittest.TestCase):
    MEASURED = {"prefill": {"prompt_tps": 300.0, "prompt_n": 8000},
                "decode": {"predicted_tps": 40.0}}

    def test_table_exhaustive_per_battery(self):
        for battery in MEASURE_KEYS:
            cases = {
                RowOutcome.ERROR: {"error": {"status": 500}},
                RowOutcome.TIMINGS_MISSING: {"timings_missing": True},
                RowOutcome.NO_THROUGHPUT: {},
                RowOutcome.MEASURED: dict(self.MEASURED[battery]),
            }
            self.assertEqual(set(cases), set(RowOutcome))  # new state => fail
            for want, row in cases.items():
                self.assertIs(row_outcome(row, battery), want, (battery, want))

    def test_precedence(self):
        both = {"error": {}, "timings_missing": True, "prompt_tps": 1,
                "prompt_n": 1}
        self.assertIs(row_outcome(both, "prefill"), RowOutcome.ERROR)
        self.assertIs(row_outcome({"error": None}, "prefill"),
                      RowOutcome.ERROR)  # key presence, not truthiness
        tm = {"timings_missing": True, "predicted_tps": 5.0}
        self.assertIs(row_outcome(tm, "decode"), RowOutcome.TIMINGS_MISSING)

    def test_summaries_count_every_state_once(self):
        for battery, summarize in (("prefill", summarize_prefill),
                                   ("decode", summarize_decode)):
            rows = [{"error": {}}, {"timings_missing": True}, {},
                    dict(self.MEASURED[battery])]
            s = summarize(rows)
            self.assertEqual(s["rows"], 4)
            self.assertEqual(s["errors"], 1)
            self.assertEqual(s["timings_missing_rows"], 1)

    def test_illegal_combo_counted_once_in_both_summaries(self):
        # BUG (old code): decode counted an error+timings_missing row in BOTH
        # errors and timings_missing_rows; prefill counted it once.
        row = {"error": {"status": 500}, "timings_missing": True}
        for summarize in (summarize_prefill, summarize_decode):
            s = summarize([row])
            self.assertEqual((s["errors"], s["timings_missing_rows"]), (1, 0))

    def test_word_approx_counts_agree_on_error_rows(self):
        # BUG (old code): a fallback-sized row whose completion then failed
        # counted as word_approx in decode (1) but not in prefill (0).
        row = {"sizing": WA, "error": {"status": 500}}
        self.assertEqual(summarize_prefill([row])["word_approx_rows"], 1)
        self.assertEqual(summarize_decode([row])["word_approx_rows"], 1)

    def test_summary_key_order_unchanged(self):
        self.assertEqual(list(summarize_prefill([])),
                         ["rows", "errors", "timings_missing_rows",
                          "word_approx_rows", "by_measured_prompt_n"])
        self.assertEqual(list(summarize_decode([])),
                         ["rows", "errors", "timings_missing_rows",
                          "word_approx_rows", "early_eos_rows", "decode_tps"])


class TestAmbientVerdict(unittest.TestCase):
    def test_table_exhaustive(self):
        cases = {
            AmbientVerdict.ERROR: ({"error": {}}, {"content_sha": "a"}),
            AmbientVerdict.INDETERMINATE: ({"content_sha": None}, {}),
            AmbientVerdict.EQUAL: ({"content_sha": "a"}, {"content_sha": "a"}),
            AmbientVerdict.DIVERGED: ({"content_sha": "a"}, {"content_sha": "b"}),
        }
        self.assertEqual(set(cases), set(AmbientVerdict))
        self.assertEqual(set(SHA_EQUAL), set(AmbientVerdict))
        for want, (a, b) in cases.items():
            self.assertIs(ambient_verdict(a, b), want)

    def test_persisted_sha_equal_values(self):
        self.assertEqual(SHA_EQUAL, {AmbientVerdict.ERROR: None,
                                     AmbientVerdict.INDETERMINATE: None,
                                     AmbientVerdict.EQUAL: True,
                                     AmbientVerdict.DIVERGED: False})

    def test_error_wins_and_one_side_empty_is_divergence(self):
        self.assertIs(ambient_verdict({"content_sha": "a"}, {"error": {}}),
                      AmbientVerdict.ERROR)
        self.assertIs(ambient_verdict({"content_sha": None},
                                      {"content_sha": "a"}),
                      AmbientVerdict.DIVERGED)

    def test_sizing_exception_pair_has_sha_equal(self):
        # BUG (old code): when prompt sizing raised, the ambient pair had no
        # `sha_equal` key at all, contradicting the documented contract
        # "errors surface as errors (sha_equal stays None)".
        class Boom(Transport):
            def post(self, path, payload, timeout):
                raise ConnectionError("boom")

        pairs, rows = run_ambient(Boom(), [8000],
                                  {"tokenize": 1, "completion": 1},
                                  TokenizeLatch(), log=lambda *a: None)
        self.assertEqual(rows, [])
        for p in pairs:
            self.assertIn("sha_equal", p)
            self.assertIsNone(p["sha_equal"])
            self.assertIn("boom", p["error"])
        json.dumps(pairs)  # still serializable


if __name__ == "__main__":
    unittest.main()
