"""stack-bench test suite — unit + e2e mock, all offline."""
import contextlib
import io
import json
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stack_bench import (GrowthBoundExceeded, MockHandlerConfig, Transport,
                         UrllibTransport, _redact_argv, bucket_of, count_tokens,
                         grow_prompt, growth_iteration_budget, main,
                         make_mock_server, run_all)


def with_mock(fn):
    def wrapper(self):
        cfg = MockHandlerConfig()
        server = make_mock_server(cfg)
        port = server.server_address[1]
        url = f"http://127.0.0.1:{port}"
        try:
            fn(self, cfg, url, port)
        finally:
            server.shutdown()
    return wrapper


class TestBucketMath(unittest.TestCase):
    def test_edges(self):
        self.assertEqual(bucket_of(3999), 0)
        self.assertEqual(bucket_of(4000), 8000)
        self.assertEqual(bucket_of(7999), 8000)
        self.assertEqual(bucket_of(8000), 8000)
        self.assertEqual(bucket_of(8001), 8000)
        self.assertEqual(bucket_of(11999), 8000)
        self.assertEqual(bucket_of(12000), 16000)
        self.assertEqual(bucket_of(32000), 32000)


class TestGrowth(unittest.TestCase):
    @with_mock
    def test_convergence_and_fallback(self, cfg, url, port):
        t = UrllibTransport(url)
        text, n, mode = grow_prompt(t, 8000, 1, 10)
        self.assertEqual(mode, "tokenized")
        self.assertGreaterEqual(n, 8000)
        self.assertEqual(-(-len(text) // 3), n)  # mock tokenizer: ceil(len/3)
        cfg.tokenize_status = 503  # fallback path
        t2 = UrllibTransport(url)
        text2, n2, mode2 = grow_prompt(t2, 800, 7, 10)
        self.assertTrue(mode2.startswith("word-approx-fallback"))
        self.assertIsNone(n2)


class TestE2E(unittest.TestCase):
    @with_mock
    def test_full_run_shapes(self, cfg, url, port):
        summary, out, rc = run_all(
            url, "e2e", targets=[8000], reps=2,
            out_base="/tmp/sb-test-out",
            timeouts={"tokenize": 10, "completion": 10},
            transport=UrllibTransport(url), log=lambda *a: None)
        self.assertEqual(rc, 0)
        # prefill buckets keyed by MEASURED prompt_n, 8k target lands in 8000
        self.assertIn("8000", summary["prefill"]["by_measured_prompt_n"])
        stats = summary["prefill"]["by_measured_prompt_n"]["8000"]
        self.assertEqual(stats["n"], 2)
        self.assertAlmostEqual(stats["mean_tps"], 333.3, delta=0.5)
        # decode: deterministic 41.7 t/s, predicted_n == 128 (ignore_eos held)
        d = summary["decode"]["decode_tps"]
        self.assertEqual(d["n"], 2)
        self.assertAlmostEqual(d["mean_tps"], 41.7, delta=0.3)
        self.assertEqual(summary["decode"]["early_eos_rows"], 0)
        # ambient block: prefill pair per target + decode pair
        kinds = [p["kind"] for p in summary["ambient"]["pairs"]]
        self.assertEqual(kinds, ["prefill@8000", "decode"])
        self.assertTrue(all(p.get("sha_equal") for p in summary["ambient"]["pairs"]))
        # attribution fields present
        from stack_bench import TOOL_VERSION as _TV
        self.assertEqual(summary["env"]["tool_version"], _TV)
        self.assertEqual(summary["server"]["n_ctx"], 32768)
        self.assertEqual(summary["schema_version"], 1)
        # per-rep distinct prompt texts (F2): battery rows carry distinct
        # seeds (ambient probe rows excluded — they carry probe fields)
        rows = [json.loads(l) for l in open(os.path.join(out, "rows.jsonl"))]
        prefill_seeds = {r["seed"] for r in rows
                         if r["battery"] == "prefill" and "probe" not in r}
        self.assertEqual(prefill_seeds, {8000 * 10 + 0, 8000 * 10 + 1})
        for r in rows:
            if "error" not in r:
                for f in ("content_sha", "stop_type", "tokens_cached",
                          "prompt_n"):
                    self.assertIn(f, r)

    @with_mock
    def test_error_rows_capture_status_and_body(self, cfg, url, port):
        cfg.completion_500_next = 1
        summary, out, rc = run_all(
            url, "err", targets=[8000], reps=1,
            out_base="/tmp/sb-test-out",
            timeouts={"tokenize": 10, "completion": 10},
            transport=UrllibTransport(url), log=lambda *a: None)
        rows = [json.loads(l) for l in open(os.path.join(out, "rows.jsonl"))]
        errs = [r for r in rows if "error" in r]
        self.assertEqual(len(errs), 1)
        self.assertEqual(errs[0]["error"]["status"], 500)
        self.assertIn("injected completion failure",
                      errs[0]["error"]["body"])
        self.assertGreaterEqual(summary["prefill"]["errors"], 1)

    @with_mock
    def test_timings_missing_flagged_and_excluded(self, cfg, url, port):
        cfg.timings_missing = True
        summary, out, rc = run_all(
            url, "tm", targets=[8000], reps=1,
            out_base="/tmp/sb-test-out",
            timeouts={"tokenize": 10, "completion": 10},
            transport=UrllibTransport(url), log=lambda *a: None)
        self.assertEqual(summary["prefill"]["timings_missing_rows"], 1)
        self.assertEqual(summary["prefill"]["by_measured_prompt_n"], {})
        rows = [json.loads(l) for l in open(os.path.join(out, "rows.jsonl"))]
        self.assertTrue(rows[0].get("timings_missing"))

    @with_mock
    def test_old_tokenize_shape(self, cfg, url, port):
        cfg.shape = "old"
        t = UrllibTransport(url)
        _, n, mode = grow_prompt(t, 800, 3, 10)
        self.assertEqual(mode, "tokenized")  # tokens-only shape still parses

    @with_mock
    def test_early_eos_flag(self, cfg, url, port):
        cfg.eos_at = 20  # server stops early only when ignore_eos absent
        summary, out, rc = run_all(
            url, "eos", targets=[8000], reps=1,
            out_base="/tmp/sb-test-out",
            timeouts={"tokenize": 10, "completion": 10},
            transport=UrllibTransport(url), log=lambda *a: None)
        # we DO send ignore_eos → predicted_n == 128 → no early flag
        self.assertEqual(summary["decode"]["early_eos_rows"], 0)

    @with_mock
    def test_early_eos_fires_when_server_ignores_flag(self, cfg, url, port):
        cfg.eos_at = 20
        cfg.server_ignores_ignore_eos = True  # M1: the safety catch trips
        summary, out, rc = run_all(
            url, "eos2", targets=[8000], reps=1,
            out_base="/tmp/sb-test-out",
            timeouts={"tokenize": 10, "completion": 10},
            transport=UrllibTransport(url), log=lambda *a: None)
        self.assertEqual(summary["decode"]["early_eos_rows"], 1)
        rows = [json.loads(l) for l in open(os.path.join(out, "rows.jsonl"))]
        decode_rows = [r for r in rows if r["battery"] == "decode"]
        self.assertTrue(decode_rows[0].get("early_eos"))
        self.assertEqual(decode_rows[0].get("predicted_n"), 20)

    @with_mock
    def test_ambient_error_is_not_divergence(self, cfg, url, port):
        # run order: prefill(1 call) + decode(1 call), then ambient pairs —
        # fail every completion from call 3 so BOTH ambient members fail
        cfg.completion_500_from_call = 3
        summary, out, rc = run_all(
            url, "amberr", targets=[8000], reps=1,
            out_base="/tmp/sb-test-out",
            timeouts={"tokenize": 10, "completion": 10},
            transport=UrllibTransport(url), log=lambda *a: None)
        pair = summary["ambient"]["pairs"][0]
        self.assertIsNone(pair["sha_equal"])  # indeterminate, NOT false
        self.assertEqual(pair["error"]["status"], 500)
        # ambient rows persisted with their error context
        rows = [json.loads(l) for l in open(os.path.join(out, "rows.jsonl"))]
        probe_rows = [r for r in rows if "probe" in r and "error" in r]
        self.assertGreaterEqual(len(probe_rows), 2)

    @with_mock
    def test_hung_tokenize_bounded(self, cfg, url, port):
        import time as _t
        cfg.slow_tokenize_s = 5.0  # handler sleeps past the socket timeout
        t = UrllibTransport(url)
        t0 = _t.time()
        text, n, mode = grow_prompt(t, 800, 5, timeout=0.3)
        elapsed = _t.time() - t0
        self.assertTrue(mode.startswith("word-approx-fallback"))
        self.assertLess(elapsed, 5.0)  # ONE timeout, not one per iteration

    @with_mock
    def test_fallback_flags_in_rows_and_summary(self, cfg, url, port):
        cfg.tokenize_status = 503
        summary, out, rc = run_all(
            url, "fb", targets=[8000], reps=1,
            out_base="/tmp/sb-test-out",
            timeouts={"tokenize": 10, "completion": 10},
            transport=UrllibTransport(url), log=lambda *a: None)
        self.assertGreater(summary["prefill"]["word_approx_rows"], 0)
        self.assertGreater(summary["decode"]["word_approx_rows"], 0)
        rows = [json.loads(l) for l in open(os.path.join(out, "rows.jsonl"))]
        for r in rows:
            if "error" not in r:
                self.assertTrue(str(r.get("sizing", "")).startswith(
                    "word-approx-fallback"))

    @with_mock
    def test_ambient_divergence_detected(self, cfg, url, port):
        cfg.vary_content = True  # mock flips content per call → sha mismatch
        summary, out, rc = run_all(
            url, "amb", targets=[8000], reps=1,
            out_base="/tmp/sb-test-out",
            timeouts={"tokenize": 10, "completion": 10},
            transport=UrllibTransport(url), log=lambda *a: None)
        self.assertFalse(summary["ambient"]["pairs"][0]["sha_equal"])


    @with_mock
    def test_hard_fail_exit_code(self, cfg, url, port):
        cfg.completion_500_from_call = 1  # every completion fails
        summary, out, rc = run_all(
            url, "hard", targets=[8000], reps=1,
            out_base="/tmp/sb-test-out",
            timeouts={"tokenize": 10, "completion": 10},
            transport=UrllibTransport(url), log=lambda *a: None)
        self.assertEqual(rc, 1)
        self.assertEqual(summary["prefill"]["errors"], 1)
        self.assertEqual(summary["decode"]["errors"], 1)
        # artifacts still written despite hard failure
        self.assertTrue(os.path.exists(os.path.join(out, "summary.json")))

    def test_garbage_200_is_hard_fail(self):
        """A server returning 200 with garbage must exit 1, never exit 0
        with an empty summary (misleading success)."""
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class BadHandler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                body = b"not-json{{{" if self.path == "/tokenize" else b""
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        srv = HTTPServer(("127.0.0.1", 0), BadHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        try:
            summary, out, rc = run_all(
                url, "garbage", targets=[8000], reps=1,
                out_base="/tmp/sb-test-out",
                timeouts={"tokenize": 5, "completion": 5},
                transport=UrllibTransport(url), log=lambda *a: None)
            self.assertEqual(rc, 1)
            self.assertEqual(summary["prefill"]["by_measured_prompt_n"], {})
            self.assertIsNone(summary["decode"]["decode_tps"])
            # both flags counted despite co-occurring
            self.assertEqual(summary["prefill"]["word_approx_rows"], 1)
            self.assertEqual(summary["prefill"]["timings_missing_rows"], 1)
        finally:
            srv.shutdown()


class _CountingTransport(UrllibTransport):
    """Real urllib transport that counts /tokenize calls and refuses more than
    `cap`: a runaway growth loop fails the test fast and explicitly instead
    of hanging the suite."""

    def __init__(self, url, cap=2):
        super().__init__(url)
        self.cap = cap
        self.tokenize_calls = 0

    def post(self, path, payload, timeout):
        if path == "/tokenize":
            self.tokenize_calls += 1
            if self.tokenize_calls > self.cap:
                raise AssertionError(
                    f"/tokenize called more than {self.cap} times without the "
                    f"run falling back: prompt growth is looping")
        return super().post(path, payload, timeout)


def shape_server(tokenize_body, completion_body=b'{"not_a_benchmark": true}'):
    """Loopback server answering every POST with HTTP 200. Each body is bytes
    or a callable(payload) -> bytes. Returns (server, url)."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    def body_for(spec, payload):
        return spec(payload) if callable(spec) else spec

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            payload = {}
            try:
                payload = json.loads(raw or b"{}")
            except ValueError:
                pass
            if self.path == "/tokenize":
                body = body_for(tokenize_body, payload)
            else:
                body = body_for(completion_body, payload)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


class TestTokenizeShapes(unittest.TestCase):
    """t_bc35a990: a 200 with valid JSON but no real token list is a FAILED
    /tokenize (None), never a count of 0 -- 0 is treated as growth data."""

    class Stub(Transport):
        def __init__(self, response):
            self.response = response

        def post(self, path, payload, timeout):
            return self.response

    @staticmethod
    def _resp(js, ok=True, status=200):
        return {"ok": ok, "status": status, "json": js, "body": ""}

    def test_shape_table(self):
        cases = [
            (self._resp({"not_a_benchmark": True}), None),
            (self._resp({"tokens": None}), None),
            (self._resp({"tokens": []}), None),
            (self._resp({"tokens": "abc"}), None),
            (self._resp({"tokens": [1, 2]}), 2),
            (self._resp({"tokens": [0] * 7, "pieces": ["x"] * 7}), 7),
            (self._resp(None), None),                        # non-JSON body
            (self._resp({"tokens": [1, 2]}, ok=False, status=500), None),
        ]
        for response, want in cases:
            self.assertIs(count_tokens(self.Stub(response), "text", 1), want,
                          response)


class TestValidJson200WithoutTokens(unittest.TestCase):
    """t_bc35a990 regression: a /tokenize 200 whose JSON has no usable
    `tokens` list must terminate and exit 1 with a BOUNDED /tokenize call
    count. Before the fix the growth loop called /tokenize forever."""

    def _run(self, payload):
        srv, url = shape_server(json.dumps(payload).encode())
        transport = _CountingTransport(url)
        t0 = time.time()
        try:
            summary, out, rc = run_all(
                url, "shape", targets=[8000], reps=1,
                out_base="/tmp/sb-test-out",
                timeouts={"tokenize": 5, "completion": 5},
                transport=transport, log=lambda *a: None)
        finally:
            srv.shutdown()
            srv.server_close()
        return summary, rc, transport.tokenize_calls, time.time() - t0

    def test_missing_tokens_key_terminates_exit_1(self):
        summary, rc, calls, elapsed = self._run({"not_a_benchmark": True})
        self.assertEqual(rc, 1)
        self.assertEqual(calls, 1)          # latch: one failed tokenize total
        self.assertLess(elapsed, 20)
        self.assertEqual(summary["prefill"]["by_measured_prompt_n"], {})
        self.assertEqual(summary["prefill"]["word_approx_rows"], 1)
        self.assertIsNone(summary["decode"]["decode_tps"])

    def test_null_tokens_terminates_exit_1(self):
        summary, rc, calls, elapsed = self._run({"tokens": None})
        self.assertEqual(rc, 1)
        self.assertEqual(calls, 1)
        self.assertLess(elapsed, 20)
        self.assertEqual(summary["prefill"]["by_measured_prompt_n"], {})
        self.assertEqual(summary["prefill"]["word_approx_rows"], 1)

    def test_empty_tokens_terminates_exit_1(self):
        summary, rc, calls, elapsed = self._run({"tokens": []})
        self.assertEqual(rc, 1)
        self.assertEqual(calls, 1)
        self.assertLess(elapsed, 20)


class TestGrowthBound(unittest.TestCase):
    """Growth has a finite iteration budget; hitting it is a loud hard
    failure, never an infinite loop or a partial success."""

    class ConstantCount(Transport):
        """Always reports the same small token count: growth can never reach
        the target."""

        def __init__(self, n=100):
            self.n, self.calls = n, 0

        def post(self, path, payload, timeout):
            self.calls += 1
            return {"ok": True, "status": 200,
                    "json": {"tokens": [0] * self.n}, "body": ""}

    def test_budget_finite_and_generous(self):
        self.assertEqual(growth_iteration_budget(800), 4 * 7 + 64)
        # the largest allowed target still fits at one token per appended word
        self.assertGreaterEqual(growth_iteration_budget(131072),
                                131072 // 120)

    def test_grow_prompt_raises_at_the_bound(self):
        t = self.ConstantCount()
        with self.assertRaises(GrowthBoundExceeded) as cm:
            grow_prompt(t, 800, 1, 1)
        self.assertEqual(t.calls, growth_iteration_budget(800))
        self.assertIn("800 tokens", str(cm.exception))
        self.assertIn("last /tokenize count: 100", str(cm.exception))

    def test_partial_run_is_a_hard_failure(self):
        """One target measures, another hits the ceiling: the run must exit 1
        and say so, not report the measured target as success."""

        def tok(payload):
            n = min(-(-len(payload.get("content") or "") // 3), 1000)
            return json.dumps({"tokens": [0] * n}).encode()

        def comp(payload):
            n = min(-(-len(payload.get("prompt") or "") // 3), 1000)
            return json.dumps({
                "content": "ok", "stop_type": "length",
                "timings": {"prompt_n": n, "prompt_ms": n * 3.0,
                            "prompt_per_second": n / (n * 3.0 / 1000.0),
                            "predicted_n": 4, "predicted_ms": 96.0,
                            "predicted_per_second": 4 / 0.096,
                            "tokens_cached": 0}}).encode()

        srv, url = shape_server(tok, comp)
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                summary, out, rc = run_all(
                    url, "bound", targets=[800, 8000], reps=1,
                    out_base="/tmp/sb-test-out",
                    timeouts={"tokenize": 5, "completion": 5},
                    transport=UrllibTransport(url), decode_ctx=800,
                    log=lambda *a: None)
        finally:
            srv.shutdown()
            srv.server_close()
        # a real measurement DID happen for the 800 target ...
        self.assertTrue(summary["prefill"]["by_measured_prompt_n"])
        self.assertIsNotNone(summary["decode"]["decode_tps"])
        # ... but the 8000 target never grew: the run must still exit 1
        self.assertEqual(rc, 1)
        rows = [json.loads(l) for l in open(os.path.join(out, "rows.jsonl"))]
        bound_rows = [r for r in rows
                      if (r.get("error") or {}).get("growth_bound")]
        self.assertEqual(len(bound_rows), 1)
        self.assertIn("8000", bound_rows[0]["error"]["transport_error"])
        pair = [p for p in summary["ambient"]["pairs"]
                if p["kind"] == "prefill@8000"][0]
        self.assertTrue(pair.get("growth_bound"))
        # and the run says so loudly
        self.assertIn("FATAL", err.getvalue())
        self.assertIn("growth", err.getvalue())


class TestCliBounds(unittest.TestCase):
    def test_reps_rejected(self):
        with self.assertRaises(SystemExit) as cm:
            main(["--url", "http://x", "--reps", "11"])
        self.assertEqual(cm.exception.code, 2)

    def test_target_rejected(self):
        with self.assertRaises(SystemExit) as cm:
            main(["--url", "http://x", "--targets", "999999"])
        self.assertEqual(cm.exception.code, 2)

    def test_n_predict_rejected(self):
        with self.assertRaises(SystemExit) as cm:
            main(["--url", "http://x", "--n-predict", "4096"])
        self.assertEqual(cm.exception.code, 2)

    def test_url_required(self):
        with self.assertRaises(SystemExit) as cm:
            main([])
        self.assertEqual(cm.exception.code, 2)

    def test_selftest_cli(self):
        self.assertEqual(main(["--selftest"]), 0)


class TestArgvRedaction(unittest.TestCase):
    SECRET = "sk-test-DO-NOT-SHIP-0123456789"

    def test_redact_forms(self):
        cases = [
            (["stack_bench.py", "--url", "http://x", "--api-key", self.SECRET,
              "--reps", "2"],
             ["stack_bench.py", "--url", "http://x", "--api-key", "***",
              "--reps", "2"]),
            (["stack_bench.py", "--api-key=" + self.SECRET, "--profile",
              "quick"],
             ["stack_bench.py", "--api-key=***", "--profile", "quick"]),
            (["stack_bench.py", "--api", self.SECRET],  # argparse abbreviation
             ["stack_bench.py", "--api", "***"]),
        ]
        for argv, expected in cases:
            self.assertEqual(_redact_argv(argv), expected, argv)
        # untouched when no api-key is present
        plain = ["stack_bench.py", "--url", "http://x"]
        self.assertEqual(_redact_argv(plain), plain)

    @with_mock
    def test_api_key_not_persisted_in_artifacts(self, cfg, url, port):
        backup = sys.argv
        sys.argv = ["stack_bench.py", "--url", url, "--api-key", self.SECRET]
        try:
            summary, out, rc = run_all(
                url, "redact-e2e", targets=[8000], reps=1,
                out_base="/tmp/sb-test-out",
                timeouts={"tokenize": 10, "completion": 10},
                transport=UrllibTransport(url, self.SECRET),
                log=lambda *a: None)
        finally:
            sys.argv = backup
        with open(os.path.join(out, "summary.json")) as f:
            summary_blob = f.read()
        with open(os.path.join(out, "rows.jsonl")) as f:
            rows_blob = f.read()
        self.assertNotIn(self.SECRET, summary_blob)
        self.assertNotIn(self.SECRET, rows_blob)
        argv = summary["env"]["argv"]
        self.assertIn("--api-key", argv)
        self.assertEqual(argv[argv.index("--api-key") + 1], "***")


if __name__ == "__main__":
    unittest.main()
