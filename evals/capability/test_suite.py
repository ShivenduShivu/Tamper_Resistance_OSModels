"""Offline unit tests: no GPU, no network, no dataset downloads.

Covers the parts that decide numbers: GSM8K extraction, custom instruction
checks and their fail-closed validation, Wilson intervals, and the paired
McNemar analysis including every counterexample from the independent review
(identical vectors, opposite vectors, all-discordant, sign convention, empty
packs). Run: python3 test_suite.py
"""
import json
import sys
import types
import unittest

import tasks
import scoring
import run as runner


class TestGsm8k(unittest.TestCase):
    def test_extract_with_marker(self):
        self.assertEqual(tasks.gsm8k_extract("blah blah\nAnswer: 42"), (42.0, "marker"))

    def test_extract_last_number_fallback(self):
        self.assertEqual(tasks.gsm8k_extract("so 3 + 4 = 7."), (7.0, "fallback"))

    def test_extract_none_on_empty(self):
        self.assertEqual(tasks.gsm8k_extract(""), (None, "none"))
        self.assertEqual(tasks.gsm8k_extract("no digits here"), (None, "none"))

    def test_extract_strips_commas(self):
        self.assertEqual(tasks.gsm8k_extract("Answer: 1,200"), (1200.0, "marker"))

    def test_extract_fraction_not_prefix(self):
        # review counterexample: 'Answer: 2/3' must NOT award the prefix 2
        v, route = tasks.gsm8k_extract("Answer: 2/3")
        self.assertAlmostEqual(v, 2 / 3)
        self.assertEqual(route, "marker")
        self.assertEqual(tasks.gsm8k_score({"answer": "2"}, "Answer: 2/3"), 0)

    def test_extract_leading_dot_not_5(self):
        # review counterexample: 'Answer: .5' must be 0.5, not 5
        self.assertEqual(tasks.gsm8k_extract("Answer: .5"), (0.5, "marker"))
        self.assertEqual(tasks.gsm8k_score({"answer": "5"}, "Answer: .5"), 0)
        self.assertEqual(tasks.gsm8k_score({"answer": "0.5"}, "Answer: .5"), 1)

    def test_extract_negative(self):
        self.assertEqual(tasks.gsm8k_extract("Answer: -12"), (-12.0, "marker"))

    def test_exponent_rejected_not_misparsed(self):
        # declared protocol: scientific notation fails extraction
        self.assertEqual(tasks.gsm8k_extract("Answer: 1e5"), (None, "none"))
        self.assertEqual(tasks.gsm8k_extract("Answer: 2.5e3"), (None, "none"))

    def test_marker_after_last_wins(self):
        self.assertEqual(tasks.gsm8k_extract("Answer: 10\nwait, answer: 12"), (12.0, "marker"))

    def test_reject_ambiguous_or_negated_marker_lines(self):
        # retest counterexamples: the marker line must be ONE complete number
        self.assertEqual(tasks.gsm8k_extract("Answer: 42 or 43"), (None, "none"))
        self.assertEqual(tasks.gsm8k_extract("Answer: not 42"), (None, "none"))
        self.assertEqual(tasks.gsm8k_extract("Answer: 42 bananas"), (None, "none"))
        self.assertEqual(tasks.gsm8k_extract("Answer: 42%"), (None, "none"))

    def test_reject_mixed_number(self):
        self.assertEqual(tasks.gsm8k_extract("Answer: 1 1/3"), (None, "none"))

    def test_fallback_exponent_not_awarded(self):
        # retest counterexamples: gold 5 must NOT be awarded via exponent forms,
        # unsigned or signed - the trailing digits are not a separate token
        self.assertEqual(tasks.gsm8k_extract("therefore 1e5"), (None, "none"))
        self.assertEqual(tasks.gsm8k_score({"answer": "5"}, "therefore 1e5"), 0)
        self.assertEqual(tasks.gsm8k_extract("therefore 1e+5"), (None, "none"))
        self.assertEqual(tasks.gsm8k_score({"answer": "5"}, "therefore 1e+5"), 0)
        self.assertEqual(tasks.gsm8k_score({"answer": "5"}, "therefore 2e-3 and then nothing"), 0)
        self.assertEqual(tasks.gsm8k_extract("Answer: 1e+5"), (None, "none"))

    def test_marker_allows_dollar_and_trailing_period(self):
        self.assertEqual(tasks.gsm8k_extract("Answer: $42"), (42.0, "marker"))
        self.assertEqual(tasks.gsm8k_extract("Answer: 42."), (42.0, "marker"))

    def test_exact_rational_match_no_tolerance(self):
        # declared protocol: exact rational equality, no float tolerance
        self.assertEqual(tasks.gsm8k_score({"answer": "0.5"}, "Answer: .5"), 1)
        self.assertEqual(tasks.gsm8k_score({"answer": "1"}, "Answer: 3/3"), 1)
        self.assertEqual(tasks.gsm8k_score({"answer": "41.9999"}, "Answer: 42"), 0)

    def test_score_match_and_miss(self):
        row = {"id": "x", "answer": "42"}
        self.assertEqual(tasks.gsm8k_score(row, "Answer: 42"), 1)
        self.assertEqual(tasks.gsm8k_score(row, "Answer: 41"), 0)
        self.assertEqual(tasks.gsm8k_score(row, "no answer"), 0)

    def test_score_detail_reports_route(self):
        d = tasks.gsm8k_score_detail({"id": "x", "answer": "42"}, "Answer: 42")
        self.assertEqual(d["route"], "marker")
        d = tasks.gsm8k_score_detail({"id": "x", "answer": "42"}, "nothing numeric")
        self.assertEqual(d["route"], "none")
        self.assertEqual(d["correct"], 0)


class TestInstructionChecks(unittest.TestCase):
    def test_keywords_existence(self):
        row = {"instruction_id_list": ["keywords:existence"], "kwargs": [{"keywords": ["apple", "banana"]}]}
        self.assertEqual(tasks.ifeval_score(row, "An apple and a banana."), 1)
        self.assertEqual(tasks.ifeval_score(row, "Only an apple."), 0)

    def test_word_count_less_than(self):
        row = {"instruction_id_list": ["length_constraints:number_words"],
               "kwargs": [{"relation": "less than", "num_words": 5}]}
        self.assertEqual(tasks.ifeval_score(row, "one two three"), 1)
        self.assertEqual(tasks.ifeval_score(row, "one two three four five six"), 0)

    def test_bullets_exact(self):
        row = {"instruction_id_list": ["detectable_format:number_bullet_lists"],
               "kwargs": [{"num_bullets": 2}]}
        self.assertEqual(tasks.ifeval_score(row, "* one\n* two"), 1)
        self.assertEqual(tasks.ifeval_score(row, "* one"), 0)

    def test_all_instructions_must_pass(self):
        row = {"instruction_id_list": ["keywords:existence", "detectable_format:title"],
               "kwargs": [{"keywords": ["x"]}, {}]}
        self.assertEqual(tasks.ifeval_score(row, "<<T>>\nhas x"), 1)
        self.assertEqual(tasks.ifeval_score(row, "has x but no title"), 0)

    def test_empty_generation_fails(self):
        row = {"instruction_id_list": ["keywords:existence"], "kwargs": [{"keywords": ["x"]}]}
        self.assertEqual(tasks.ifeval_score(row, ""), 0)

    def test_kwargs_length_mismatch_fails_closed(self):
        # review counterexample: zip truncation must never silently skip an instruction
        row = {"instruction_id_list": ["keywords:existence", "detectable_format:title"],
               "kwargs": [{"keywords": ["x"]}]}
        self.assertEqual(tasks.ifeval_score(row, "x <<T>>"), 0)

    def test_missing_verifier_params_fail_closed(self):
        row = {"instruction_id_list": ["keywords:existence"], "kwargs": [{}]}
        self.assertEqual(tasks.ifeval_score(row, "anything"), 0)

    def test_end_checker_case_and_quotes(self):
        # official-parity: strips surrounding quotes, case-insensitive endswith
        row = {"instruction_id_list": ["startend:end_checker"], "kwargs": [{"end_phrase": "The End"}]}
        self.assertEqual(tasks.ifeval_score(row, "some text\nthe end"), 1)
        self.assertEqual(tasks.ifeval_score(row, 'some text\n"The End"'), 1)
        self.assertEqual(tasks.ifeval_score(row, "the end is near"), 0)

    def test_params_ok_validation(self):
        self.assertTrue(tasks.params_ok("keywords:existence", {"keywords": ["a"]}))
        self.assertFalse(tasks.params_ok("keywords:existence", {"keywords": []}))
        self.assertFalse(tasks.params_ok("keywords:existence", {}))
        self.assertFalse(tasks.params_ok("keywords:frequency",
                                         {"keyword": "a", "relation": "exactly", "frequency": 2}))
        self.assertTrue(tasks.params_ok("keywords:frequency",
                                        {"keyword": "a", "relation": "at least", "frequency": 2}))
        self.assertFalse(tasks.params_ok("not:a:real_check", {}))

    def test_params_ok_rejects_bad_types_and_counts(self):
        # retest counterexamples: non-string keywords and non-positive counts
        self.assertFalse(tasks.params_ok("keywords:existence", {"keywords": [12]}))
        self.assertFalse(tasks.params_ok("keywords:existence", {"keywords": ["a", 12]}))
        self.assertFalse(tasks.params_ok("length_constraints:number_words",
                                         {"num_words": -2, "relation": "at least"}))
        self.assertFalse(tasks.params_ok("length_constraints:number_words",
                                         {"num_words": 0, "relation": "at least"}))
        self.assertFalse(tasks.params_ok("keywords:frequency",
                                         {"keyword": "a", "relation": "at least", "frequency": 0}))
        self.assertFalse(tasks.params_ok("detectable_format:number_bullet_lists",
                                         {"num_bullets": -1}))

    def test_empty_instruction_list_raises(self):
        with self.assertRaises(ValueError):
            tasks.ifeval_score({"instruction_id_list": [], "kwargs": []}, "anything")


class TestScoring(unittest.TestCase):
    def test_wilson_bounds(self):
        lo, hi = scoring.wilson(520, 1000)
        self.assertAlmostEqual(lo, 0.489, places=2)
        self.assertAlmostEqual(hi, 0.551, places=2)

    def test_wilson_resolution_documented(self):
        # README resolution claim: n=200 at 50% is roughly +/-6.9 points
        lo, hi = scoring.wilson(100, 200)
        self.assertAlmostEqual(lo, 0.431, places=2)
        self.assertAlmostEqual(hi, 0.569, places=2)

    def test_rate_with_ci_empty_raises(self):
        # review: empty packs must fail, not report 0 with CI [0,0]
        with self.assertRaises(ValueError):
            scoring.rate_with_ci([])

    def test_newcombe_published_reference(self):
        # statsmodels devel confint_proportions_paired doc example (verified there
        # against R ratesci/PropCIs): table [[10,5],[2,20]] -> (-0.0643926, 0.2211160)
        cand = [1]*10 + [1]*5 + [0]*2 + [0]*20
        base = [1]*10 + [0]*5 + [1]*2 + [0]*20
        r = scoring.mcnemar(cand, base)
        self.assertAlmostEqual(r["delta"], 3/37, places=4)
        self.assertAlmostEqual(r["delta_ci"][0], -0.0644, places=3)
        self.assertAlmostEqual(r["delta_ci"][1], 0.2211, places=3)

    def test_mcnemar_identical_vectors_narrow_not_zero(self):
        # Newcombe-10 has positive width even with zero discordance: n=100
        # identical answers cannot pin the true difference to exactly 0
        a = [1, 0] * 50
        r = scoring.mcnemar(a, a)
        self.assertEqual(r["delta"], 0)
        lo, hi = r["delta_ci"]
        self.assertTrue(lo < 0 < hi)
        self.assertLess(abs(hi - lo), 0.06)   # narrow, about +/-0.019 at n=100
        self.assertEqual(r["mcnemar_p_exact"], 1.0)
        self.assertFalse(r["detected_loss"])
        self.assertFalse(r["detected_gain"])

    def test_mcnemar_opposite_vectors_wide_ci(self):
        # anti-correlated equal-accuracy vectors must give a WIDE interval
        a = [1] * 50 + [0] * 50
        b = [0] * 50 + [1] * 50
        r = scoring.mcnemar(a, b)
        self.assertEqual(r["delta"], 0)
        lo, hi = r["delta_ci"]
        self.assertAlmostEqual(lo, -0.192, places=2)
        self.assertAlmostEqual(hi, 0.192, places=2)

    def test_mcnemar_all_discordant_boundary_not_degenerate(self):
        # 100 candidate-only wins: calibrated boundary CI ~[0.948, 1.0], NOT [1,1]
        cand = [1] * 100
        base = [0] * 100
        r = scoring.mcnemar(cand, base)
        self.assertEqual(r["delta"], 1.0)
        lo, hi = r["delta_ci"]
        self.assertGreater(lo, 0.9)
        self.assertLess(lo, 0.97)
        self.assertAlmostEqual(hi, 1.0, places=4)
        self.assertTrue(r["detected_gain"])
        self.assertFalse(r["detected_loss"])

    def test_mcnemar_p_full_precision(self):
        # tiny p must not be rounded to 0.0000
        cand = [1] * 100
        base = [0] * 100
        r = scoring.mcnemar(cand, base)
        self.assertGreater(r["mcnemar_p_exact"], 0)
        self.assertLess(r["mcnemar_p_exact"], 1e-20)

    def test_mcnemar_sign_convention_candidate_minus_base(self):
        # review: the delta must be candidate - base, with the candidate FIRST
        cand = [1] * 100
        base = [1] * 50 + [0] * 50
        r = scoring.mcnemar(cand, base)
        self.assertAlmostEqual(r["delta"], 0.5)          # candidate better -> positive
        self.assertEqual(r["candidate_only_correct"], 50)
        self.assertEqual(r["base_only_correct"], 0)
        self.assertTrue(r["detected_gain"])
        self.assertFalse(r["detected_loss"])
        r2 = scoring.mcnemar(base, cand)
        self.assertAlmostEqual(r2["delta"], -0.5)        # symmetric flip
        self.assertTrue(r2["detected_loss"])

    def test_mcnemar_small_discordance_not_detected(self):
        cand = [1, 1, 1, 0] + [1] * 96
        base = [1, 0, 1, 1] + [1] * 96
        r = scoring.mcnemar(cand, base)
        self.assertFalse(r["detected_loss"])
        self.assertFalse(r["detected_gain"])

    def test_mcnemar_positively_correlated_ci_narrow(self):
        a = [1] * 90 + [0] * 10
        b = [1] * 85 + [0] * 5 + [1] * 5 + [0] * 5
        r = scoring.mcnemar(a, b)
        lo, hi = r["delta_ci"]
        self.assertLess(abs(hi - lo), 0.15)  # much tighter than two independent Wilsons

    def test_mcnemar_length_mismatch(self):
        with self.assertRaises(AssertionError):
            scoring.mcnemar([1, 0], [1])


class TestMmluPrompt(unittest.TestCase):
    def test_format(self):
        row = {"question": "Q?", "choices": ["w", "x", "y", "z"], "answer": 2}
        p = tasks.mmlu_prompt(row)
        self.assertIn("A. w", p)
        self.assertIn("D. z", p)
        self.assertTrue(p.endswith("Answer with the letter only."))


class TestStableIds(unittest.TestCase):
    def test_dedupe_check(self):
        tasks._dedupe_check([{"id": "a"}, {"id": "b"}])
        with self.assertRaises(ValueError):
            tasks._dedupe_check([{"id": "a"}, {"id": "a"}])


class TestTaskRegistry(unittest.TestCase):
    def test_mbpp_removed_fails_closed(self):
        with self.assertRaises(ValueError):
            runner.resolve_task("mbpp")

    def test_unknown_task_rejected(self):
        with self.assertRaises(ValueError):
            runner.resolve_task("cyber")
        with self.assertRaises(ValueError):
            runner.resolve_task("global_mmlu_de")  # not in the allow-list

    def test_valid_tasks_resolve(self):
        for t in ["mmlu", "gsm8k", "ifeval_custom", "cybermetric"] + ["global_mmlu_" + l for l in "hi ar fr es".split()]:
            self.assertEqual(runner.resolve_task(t), t)

    def test_global_mmlu_lang_case(self):
        with self.assertRaises(ValueError):
            tasks.global_mmlu_load("de")  # rejected before any load_dataset call


class TestCyberMetric(unittest.TestCase):
    GOOD = {"questions": [
        {"question": f"q{i}", "answers": {"A": "a", "B": "b", "C": "c", "D": "d"},
         "solution": s} for i, s in enumerate("ABCDABCD")]}
    REV = "32759c2fa90274706219e1cf946cadff93da52e3"

    def _with_fixture(self, payload, fn, **kw):
        import tempfile, os
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(payload, f)
            path = f.name
        old = tasks._cybermetric_fetch
        tasks._cybermetric_fetch = lambda rev: path
        try:
            return fn(**kw)
        finally:
            tasks._cybermetric_fetch = old
            os.unlink(path)

    def test_bad_revision_rejected_before_download(self):
        def boom(rev):
            raise AssertionError("fetch must not run on a bad revision")
        old = tasks._cybermetric_fetch
        tasks._cybermetric_fetch = boom
        try:
            with self.assertRaises(ValueError):
                tasks.cybermetric_load(revision="latest")
        finally:
            tasks._cybermetric_fetch = old

    def test_missing_questions_key_rejected(self):
        with self.assertRaises(ValueError):
            self._with_fixture({"rows": []}, tasks.cybermetric_load, revision=self.REV)

    def test_bad_solution_letter_rejected(self):
        bad = {"questions": [{"question": "q", "answers": {"A": "a", "B": "b", "C": "c", "D": "d"},
                              "solution": "E"}]}
        with self.assertRaises(ValueError):
            self._with_fixture(bad, tasks.cybermetric_load, revision=self.REV)

    def test_happy_path_shape_ids_and_determinism(self):
        rows1 = self._with_fixture(self.GOOD, tasks.cybermetric_load, n=4, seed=7, revision=self.REV)
        rows2 = self._with_fixture(self.GOOD, tasks.cybermetric_load, n=4, seed=7, revision=self.REV)
        self.assertEqual(rows1, rows2)  # same seed -> identical item set, audit-recoverable
        self.assertEqual(len(rows1), 4)
        self.assertTrue(all(r["id"].startswith("cm#") for r in rows1))
        self.assertTrue(all(len(r["choices"]) == 4 and 0 <= r["answer"] < 4 for r in rows1))
        # mmlu_prompt must accept the shared MCQ row shape
        p = tasks.mmlu_prompt(rows1[0])
        self.assertIn("A. ", p)
        self.assertIn("Answer with the letter only.", p)
        # n larger than the pool selects the whole pool, never wraps
        rows_all = self._with_fixture(self.GOOD, tasks.cybermetric_load, n=999, seed=0, revision=self.REV)
        self.assertEqual(len(rows_all), 8)

    def test_mcq_classification(self):
        self.assertTrue(runner.is_mcq("cybermetric"))
        self.assertEqual(runner.DEFAULT_N["cybermetric"], 500)


class TestCodingBind(unittest.TestCase):
    """load_and_bind must bind CONTENT, not the stored hash label (reviewer
    repro: genuine fields_sha256 next to altered fields was accepted)."""

    def _run_dir(self, manifest_row, response_id="T"):
        import tempfile, json as j, os as o
        td = tempfile.mkdtemp()
        o.makedirs(o.path.join(td, "responses"))
        with open(o.path.join(td, "items_manifest_humaneval.json"), "w") as f:
            j.dump([manifest_row], f)
        with open(o.path.join(td, "responses", "humaneval__base.jsonl"), "w") as f:
            f.write(j.dumps({"id": response_id, "response": "x"}) + "\n")
        return td

    def _patched(self, cs, row):
        from unittest import mock
        p1 = mock.patch.object(cs.tasks, "humaneval_load",
                               lambda revision=None, meta_out=None: [row])
        p2 = mock.patch.object(cs, "_rev", lambda: "pinned-fixture")
        p1.start(); p2.start()
        self.addCleanup(p1.stop); self.addCleanup(p2.stop)

    def test_altered_fields_with_genuine_hash_rejected(self):
        import coding_score as cs, hashlib
        prompt, test, entry = "def f():\n", "def check(c): pass", "f"
        genuine = hashlib.sha256((prompt + test + entry).encode()).hexdigest()
        self._patched(cs, {"id": "T", "prompt": prompt, "test": test,
                           "entry_point": entry, "canonical_solution": "x",
                           "fields_sha256": genuine})
        td = self._run_dir({"id": "T", "fields_sha256": genuine,
                            "prompt": "ALTERED PROMPT", "test": test,
                            "entry_point": entry})
        with self.assertRaises(SystemExit):
            cs.load_and_bind(td, "base")

    def test_genuine_binding_accepted_and_scores_trusted_rows(self):
        import coding_score as cs, hashlib
        prompt, test, entry = "def f():\n", "def check(c): pass", "f"
        genuine = hashlib.sha256((prompt + test + entry).encode()).hexdigest()
        self._patched(cs, {"id": "T", "prompt": prompt, "test": test,
                           "entry_point": entry, "canonical_solution": "x",
                           "fields_sha256": genuine})
        td = self._run_dir({"id": "T", "fields_sha256": genuine,
                            "prompt": prompt, "test": test, "entry_point": entry})
        rows, gens, binding = cs.load_and_bind(td, "base")
        self.assertEqual(rows[0]["prompt"], prompt)
        self.assertEqual(gens, {"T": "x"})
        self.assertEqual(binding["dataset_revision"], "pinned-fixture")

class TestCodingExec(unittest.TestCase):
    def _frames(self, cx, module="m", inp="i", py="3.13.0", result="passed",
                passed=True):
        st = cx.FRAME_STARTUP + json.dumps({"ok": True, "module_sha256": module,
                                            "input_sha256": inp, "python": py})
        vd = cx.FRAME_VERDICT + json.dumps({"result": result, "passed": passed})
        return st, vd

    def test_parse_driver_output_phases(self):
        import coding_exec as cx, json
        st, vd = self._frames(cx)
        # startup + verdict -> verdict
        kind, payload = cx.parse_driver_output(0, st + "\n" + vd + "\n", "m", "i")
        self.assertEqual(kind, "verdict")
        self.assertEqual(payload[1]["result"], "passed")
        self.assertEqual(payload[0]["module_sha256"], "m")
        # verdict frame wins over nonzero rc (recorded before a later crash)
        kind, _ = cx.parse_driver_output(1, st + "\n" + vd + "\n", "m", "i")
        self.assertEqual(kind, "verdict")
        # no startup frame at all -> INFRA regardless of rc or stray verdicts
        self.assertEqual(cx.parse_driver_output(1, "hello\n", "m", "i")[0], "infra")
        self.assertEqual(cx.parse_driver_output(0, vd + "\n", "m", "i")[0], "infra")
        self.assertEqual(cx.parse_driver_output(None, "", "m", "i")[0], "infra")
        # startup but no verdict -> INFRA (driver/sandbox death is not a
        # candidate outcome - review fix)
        self.assertEqual(cx.parse_driver_output(-9, st + "\n", "m", "i")[0], "infra")
        # conflicting verdict frames -> confused, never trusted
        _, vd2 = self._frames(cx, result="failed: AssertionError", passed=False)
        kind, _ = cx.parse_driver_output(0, st + "\n" + vd + "\n" + vd2 + "\n", "m", "i")
        self.assertEqual(kind, "confused")
        # duplicate IDENTICAL frames collapse (idempotent emit is not confusion)
        kind, _ = cx.parse_driver_output(0, st + "\n" + vd + "\n" + vd + "\n", "m", "i")
        self.assertEqual(kind, "verdict")
        # malformed frames are INFRA, not trusted
        kind, _ = cx.parse_driver_output(0, st + "\n" + cx.FRAME_VERDICT + "0\n", "m", "i")
        self.assertEqual(kind, "infra")

    def test_parse_driver_output_evidence_binding(self):
        # the reviewer's exact v4 repro: forged startup evidence must not pass
        import coding_exec as cx, json
        st, vd = self._frames(cx, module="BAD", inp="OTHER", py="0")
        out = st + "\n" + vd + "\n"
        self.assertEqual(cx.parse_driver_output(0, out, "m", "i")[0], "infra")
        # each evidence field is individually binding
        st_bad, _ = self._frames(cx, module="BAD")
        self.assertEqual(cx.parse_driver_output(0, st_bad + "\n" + vd + "\n", "m", "i")[0], "infra")
        st_bad, _ = self._frames(cx, inp="OTHER")
        self.assertEqual(cx.parse_driver_output(0, st_bad + "\n" + vd + "\n", "m", "i")[0], "infra")
        st_bad, _ = self._frames(cx, py="0")
        self.assertEqual(cx.parse_driver_output(0, st_bad + "\n" + vd + "\n", "m", "i")[0], "infra")
        # order violation: startup after verdict -> infra
        st, vd = self._frames(cx)
        self.assertEqual(cx.parse_driver_output(0, vd + "\n" + st + "\n", "m", "i")[0], "infra")
        # conflicting startup frames -> infra
        st2, _ = self._frames(cx, py="3.13.1")
        self.assertEqual(cx.parse_driver_output(0, st + "\n" + st2 + "\n" + vd + "\n", "m", "i")[0], "infra")
        # verdict schema: result/passed inconsistency is not a verdict
        st, vd_bad = self._frames(cx, result="passed", passed=False)
        self.assertEqual(cx.parse_driver_output(0, st + "\n" + vd_bad + "\n", "m", "i")[0], "infra")
        _, vd_bad = self._frames(cx, result="maybe")
        self.assertEqual(cx.parse_driver_output(0, st + "\n" + vd_bad + "\n", "m", "i")[0], "infra")
        # explicit HEINFRA frame -> infra with the phase/error
        infra = cx.FRAME_INFRA + json.dumps({"phase": "exec", "error": "RuntimeError: boom"})
        kind, detail = cx.parse_driver_output(2, st + "\n" + infra + "\n", "m", "i")
        self.assertEqual(kind, "infra")
        self.assertIn("exec", detail)

    def test_score_completion_phase_mapping(self):
        import coding_exec as cx, json, hashlib
        problem = {"task_id": "T/0", "prompt": "def add(a, b):\n",
                   "test": "def check(c):\n    assert c(1, 2) == 3\n",
                   "entry_point": "add", "canonical_solution": "SECRET"}
        seen = {}
        class Stub:
            out = ""
            rc = 0
            def run_driver(self, drv, mod, item_json, on_create=None):
                seen["item"] = json.loads(item_json)
                seen["mod"] = mod
                return self.rc, self.out
        def startup_for(item_json):
            return cx.FRAME_STARTUP + json.dumps({
                "ok": True, "module_sha256": cx.REFERENCE_SHA256,
                "input_sha256": hashlib.sha256(item_json.encode()).hexdigest(),
                "python": "3.13.0"})
        # verdict mapping: passed / timed out / failed
        for result, want in (("passed", "passed"), ("timed out", "timed_out"),
                             ("failed: AssertionError", "failed")):
            class DynStub(Stub):
                def run_driver(self, drv, mod, item_json, on_create=None):
                    super().run_driver(drv, mod, item_json)
                    self.out = (startup_for(item_json) + "\n" + cx.FRAME_VERDICT +
                                json.dumps({"result": result,
                                            "passed": result == "passed"}) + "\n")
                    return self.rc, self.out
            rec = cx.score_completion(problem, "    return a + b\n", DynStub())
            self.assertEqual(rec["status"], want)
            self.assertEqual(rec["phase_evidence"]["module_sha256"], cx.REFERENCE_SHA256)
        # the trusted module bytes and the problem WITHOUT canonical_solution
        self.assertEqual(seen["mod"], cx._UPSTREAM_BYTES)
        self.assertNotIn("canonical_solution", seen["item"]["problem"])
        self.assertNotIn("SECRET", json.dumps(seen["item"]))
        # no startup -> infra_error, never a candidate outcome
        Stub.out, Stub.rc = "", 1
        rec = cx.score_completion(problem, "    return a + b\n", Stub())
        self.assertEqual(rec["status"], "infra_error")
        self.assertNotIn("correct", rec)
        # backend exception -> infra_error
        class Boom:
            def run_driver(self, *a):
                raise RuntimeError("sandbox create refused")
        rec = cx.score_completion(problem, "    return a + b\n", Boom())
        self.assertEqual(rec["status"], "infra_error")
        # forged startup evidence (reviewer repro) -> infra_error, NOT passed
        class Forged:
            def run_driver(self, *a):
                return 0, ('HESTARTUP::{"ok":true,"module_sha256":"BAD",'
                           '"input_sha256":"OTHER","python":"0"}\n'
                           'HEVERDICT::{"result":"passed","passed":true}\n')
        rec = cx.score_completion(problem, "    return a + b\n", Forged())
        self.assertEqual(rec["status"], "infra_error")
        self.assertNotIn("correct", rec)

    def test_extract_completion_rules(self):
        import coding_exec as cx
        body, why = cx.extract_completion("    return a + b\n")
        self.assertEqual((body, why), ("    return a + b", "raw+no_stop"))
        body, why = cx.extract_completion("```python\n    return a + b\n```\n")
        self.assertEqual((body, why), ("    return a + b", "fence_stripped+no_stop"))
        body, why = cx.extract_completion("    x = 1\n    return x\n\ndef other():\n    pass")
        self.assertEqual(body, "    x = 1\n    return x\n")
        self.assertTrue(why.endswith("stop_cut"))

    def test_completion_payload_shape(self):
        import coding_exec as cx, json
        # v4: assembly is upstream's; the driver receives problem + completion
        # as JSON and calls the pinned module's own check_correctness
        seen = {}
        class Stub:
            def run_driver(self, drv, mod, item_json, on_create=None):
                import hashlib
                seen["item"] = json.loads(item_json)
                return 0, (cx.FRAME_STARTUP + json.dumps({
                    "ok": True, "module_sha256": cx.REFERENCE_SHA256,
                    "input_sha256": hashlib.sha256(item_json.encode()).hexdigest(),
                    "python": "3.13.0"}) + "\n" +
                    cx.FRAME_VERDICT + '{"result": "passed", "passed": true}\n')
        problem = {"task_id": "T/0", "prompt": "def add(a, b):\n",
                   "test": "def check(c):\n    assert c(1, 2) == 3\n",
                   "entry_point": "add", "canonical_solution": "SECRET"}
        rec = cx.score_completion(problem, "    return a + b\n", Stub())
        self.assertEqual(seen["item"]["completion"], "    return a + b")
        self.assertEqual(seen["item"]["problem"]["prompt"], "def add(a, b):\n")
        self.assertNotIn("SECRET", json.dumps(seen["item"]))
        self.assertEqual(rec["status"], "passed")

    def test_raw_cap_applies_before_extraction(self):
        import coding_exec as cx
        huge = "    return 1\n" + " " * (cx.MAX_COMPLETION_CHARS + 10)
        rec = cx.score_completion(
            {"task_id": "T/0", "prompt": "def f():\n", "test": "t", "entry_point": "f"},
            huge, backend=None)  # backend never reached
        self.assertEqual((rec["status"], rec["finish_reason"]),
                         ("failed", "raw_over_cap"))

    def test_cost_bound_is_caps_based(self):
        import coding_exec as cx
        bound = cx.cost_bound_per_item_usd()
        # Modal Sandbox rates (3x function rates): 0.25 core * 30s * 3.942e-5
        # + 0.5 GiB * 30s * 6.67e-7. Compute-time bound only, not a spend
        # ceiling: startup, network and retention overheads come on top.
        self.assertAlmostEqual(bound, 0.25*30*0.00003942 + 0.5*30*0.00000667, places=10)
        self.assertLess(bound * 164 * 3, 0.20)  # 3-model sweep compute bound < 20 cents pre-overheads

    def test_driver_pins_are_stable(self):
        import coding_exec as cx
        self.assertEqual(cx.driver_sha256(), cx.driver_sha256())
        for marker in ("check_correctness", "HESTARTUP::", "HEVERDICT::",
                       "EXPECTED_MODULE_SHA256"):
            self.assertIn(marker, cx.driver_source())

    def test_reference_module_pinned(self):
        import coding_exec as cx, hashlib, os
        self.assertTrue(os.path.exists(cx.REFERENCE_PATH))
        with open(cx.REFERENCE_PATH, "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(),
                             cx.REFERENCE_SHA256)
        # the driver embeds the same pin for in-sandbox verification
        self.assertIn(cx.REFERENCE_SHA256, cx.driver_source())
        # the vendored module is the real upstream: it exposes check_correctness
        with open(cx.REFERENCE_PATH, encoding="utf-8") as f:
            src = f.read()
        for marker in ("def check_correctness(", "def unsafe_execute(",
                       "def reliability_guard(", "multiprocessing"):
            self.assertIn(marker, src)

    def test_image_digest_pinned(self):
        import coding_exec as cx
        self.assertIn("@sha256:", cx.SANDBOX_IMAGE)
        self.assertIn("python:3.13-slim", cx.SANDBOX_IMAGE)

    def test_backend_enforces_sdk_pin_and_no_unpinned_layer(self):
        import coding_exec as cx, inspect
        src = inspect.getsource(cx.SandboxBackend)
        # no add_python layer: an added layer on the digest base would be
        # unpinned - the from_registry call must carry no extra arguments
        self.assertIn("modal.Image.from_registry(SANDBOX_IMAGE)", src)
        # the pinned SDK version is compared against the actual at every use
        self.assertIn("REQUIRED_MODAL_VERSION", src)

    def _resume_ctx(self):
        import coding_exec as cx, coding_score as cs
        return {"model": "base", "model_revision": "rev-model-1",
                "generation_manifest_sha256": "gm1",
                "items_manifest_sha256": "m1",
                "responses_sha256": "r1", "dataset_revision": "rev1",
                "protocol": cx.CODING_PROTOCOL_VERSION,
                "driver_sha256": cx.driver_sha256(),
                "coding_exec_sha256": "ce1", "scorer_sha256": "sc1",
                "measurement_config_sha256": cs.measurement_config_sha256(),
                "sandbox_image": cx.SANDBOX_IMAGE,
                "modal_sdk_required": cx.REQUIRED_MODAL_VERSION,
                "expected_python": "3.13"}

    def _good_row(self, **over):
        import coding_exec as cx
        row = {"id": "a", "status": "passed", "correct": 1, "detail": "",
               "finish_reason": "raw+no_stop",
               "phase_evidence": {"startup": True,
                                  "module_sha256": cx.REFERENCE_SHA256,
                                  "input_sha256": "i", "driver_python": "3.13.0"}}
        row.update(over)
        return row

    def test_validate_scored_row_schema(self):
        import coding_score as cs
        ctx = self._resume_ctx()
        valid = {"a", "b"}
        row = self._good_row(); row.update(ctx)
        self.assertIsNone(cs.validate_scored_row(row, valid, ctx, expected_input_sha="i"))
        # bare {"id": "T"}-style rows are rejected (review repro)
        self.assertIsNotNone(cs.validate_scored_row({"id": "a"}, valid, ctx))
        # foreign id
        r = self._good_row(id="ZZZ"); r.update(ctx)
        self.assertIsNotNone(cs.validate_scored_row(r, valid, ctx))
        # stale/other-model/other-runtime context each rejects
        for k, bad in (("model", "v2"), ("protocol", "he-suite-3-standard"),
                       ("driver_sha256", "0" * 64), ("sandbox_image", "python:3.13-slim"),
                       ("responses_sha256", "other"), ("modal_sdk_required", "0.0.0"),
                       ("model_revision", "other-rev"),
                       ("generation_manifest_sha256", "other"),
                       ("measurement_config_sha256", "other"),
                       ("scorer_sha256", "other"), ("coding_exec_sha256", "other"),
                       ("expected_python", "3.10")):
            r = self._good_row(); r.update(ctx); r[k] = bad
            self.assertIsNotNone(cs.validate_scored_row(r, valid, ctx, expected_input_sha="i"), k)
        # missing a context key rejects
        r = self._good_row(); r.update(ctx); del r["driver_sha256"]
        self.assertIsNotNone(cs.validate_scored_row(r, valid, ctx, expected_input_sha="i"))
        # invalid status rejects
        r = self._good_row(status="maybe"); r.update(ctx)
        self.assertIsNotNone(cs.validate_scored_row(r, valid, ctx, expected_input_sha="i"))
        # the reviewer's exact schema repros: inconsistent correct all reject
        r = self._good_row(status="failed", correct=1); r.update(ctx)
        self.assertIsNotNone(cs.validate_scored_row(r, valid, ctx, expected_input_sha="i"))
        r = self._good_row(status="passed", correct=0); r.update(ctx)
        self.assertIsNotNone(cs.validate_scored_row(r, valid, ctx, expected_input_sha="i"))
        r = self._good_row(status="passed", correct=True); r.update(ctx)
        self.assertIsNotNone(cs.validate_scored_row(r, valid, ctx, expected_input_sha="i"))
        # consistent non-pass candidate rows are accepted
        for st, c in (("failed", 0), ("timed_out", 0)):
            r = self._good_row(status=st, correct=c); r.update(ctx)
            self.assertIsNone(cs.validate_scored_row(r, valid, ctx, expected_input_sha="i"), st)
        # infra row must OMIT correct
        r = self._good_row(status=cs.INFRA, correct=0); r.update(ctx)
        self.assertIsNotNone(cs.validate_scored_row(r, valid, ctx, expected_input_sha="i"))
        r = self._good_row(status=cs.INFRA); r.update(ctx); del r["correct"]; del r["phase_evidence"]
        self.assertIsNone(cs.validate_scored_row(r, valid, ctx))
        # candidate row missing phase evidence rejects (except cap case)
        r = self._good_row(); r.update(ctx); del r["phase_evidence"]
        self.assertIsNotNone(cs.validate_scored_row(r, valid, ctx, expected_input_sha="i"))
        # raw_over_cap: no phase evidence required, and none allowed
        r = self._good_row(finish_reason="raw_over_cap"); r.update(ctx); del r["phase_evidence"]
        self.assertIsNone(cs.validate_scored_row(r, valid, ctx,
                                                 expected_input_sha=None,
                                                 expected_over_cap=True))
        r = self._good_row(finish_reason="raw_over_cap"); r.update(ctx)
        self.assertIsNotNone(cs.validate_scored_row(r, valid, ctx,
                                                    expected_input_sha=None,
                                                    expected_over_cap=True))
        # wrong input hash / wrong driver python reject
        r = self._good_row(); r.update(ctx)
        self.assertIsNotNone(cs.validate_scored_row(r, valid, ctx, expected_input_sha="DIFFERENT"))
        r = self._good_row(); r.update(ctx)
        r["phase_evidence"]["driver_python"] = "3.10.12"
        self.assertIsNotNone(cs.validate_scored_row(r, valid, ctx, expected_input_sha="i"))
        # explicit local-fixture exception honors allow_local_python
        ctx_local = dict(ctx, allow_local_python=True)
        self.assertIsNone(cs.validate_scored_row(r, valid, ctx_local, expected_input_sha="i"))

    def _rows_gens(self):
        import coding_exec as cx
        rows = [{"id": "a", "prompt": "def add(a, b):\n",
                 "test": "def check(c):\n    assert c(1, 2) == 3\n",
                 "entry_point": "add", "canonical_solution": "x",
                 "fields_sha256": "f"},
                {"id": "b", "prompt": "def g():\n",
                 "test": "def check(c):\n    assert c() == 1\n",
                 "entry_point": "g", "canonical_solution": "x",
                 "fields_sha256": "f2"}]
        gens = {"a": "    return a + b\n", "b": "    return 1\n"}
        return rows, gens

    def test_resume_loader_refuses_mixed_state(self):
        import coding_exec as cx, coding_score as cs
        import tempfile, os, json
        ctx = self._resume_ctx()
        rows, gens = self._rows_gens()
        td = tempfile.mkdtemp()
        p = os.path.join(td, "scored.jsonl")
        prob_a = {"task_id": "a", "prompt": rows[0]["prompt"],
                  "test": rows[0]["test"], "entry_point": rows[0]["entry_point"]}
        sha_a = cx.expected_input_sha256(prob_a, gens["a"])
        # clean rows resume
        row = self._good_row(); row.update(ctx)
        row["phase_evidence"]["input_sha256"] = sha_a
        with open(p, "w") as f:
            f.write(json.dumps(row) + "\n")
        ids, recs = cs.load_resume(p, rows, gens, ctx)
        self.assertEqual(ids, {"a"})
        self.assertEqual(len(recs), 1)
        # per-item input hash drift (different bound response) refuses
        bad = self._good_row(); bad.update(ctx)
        bad["phase_evidence"]["input_sha256"] = "0" * 64
        with open(p, "w") as f:
            f.write(json.dumps(bad) + "\n")
        with self.assertRaises(SystemExit):
            cs.load_resume(p, rows, gens, ctx)
        # row from a DIFFERENT model (stale other-model file) refuses closed
        stale = self._good_row(id="b"); stale.update(ctx); stale["model"] = "v2"
        with open(p, "w") as f:
            f.write(json.dumps(stale) + "\n")
        with self.assertRaises(SystemExit):
            cs.load_resume(p, rows, gens, ctx)
        # malformed row refuses closed
        with open(p, "w") as f:
            f.write("not json\n")
        with self.assertRaises(SystemExit):
            cs.load_resume(p, rows, gens, ctx)
        # duplicate id refuses closed
        with open(p, "w") as f:
            f.write(json.dumps(row) + "\n" + json.dumps(row) + "\n")
        with self.assertRaises(SystemExit):
            cs.load_resume(p, rows, gens, ctx)
        # missing file resumes empty
        ids, recs = cs.load_resume(os.path.join(td, "none.jsonl"), rows, gens, ctx)
        self.assertEqual((ids, recs), (set(), []))

    def test_writer_lock_and_interruption_reconciliation(self):
        import coding_score as cs
        import tempfile, os, json
        ctx = self._resume_ctx()
        td = tempfile.mkdtemp()
        out_items = os.path.join(td, "scored.jsonl")
        # fresh acquire; a LIVE writer refuses every later acquire - even
        # under a DIFFERENT context (ack never reaches a live writer)
        tok, fd = cs.acquire_writer_lock(out_items, ctx)
        with self.assertRaises(SystemExit):
            cs.acquire_writer_lock(out_items, ctx)
        with self.assertRaises(SystemExit):
            cs.acquire_writer_lock(out_items, dict(ctx, model="OTHER"))
        # a wrong-token release leaves the live lock untouched
        cs.release_writer_lock(out_items, "WRONG-TOKEN", fd)
        with self.assertRaises(SystemExit):
            cs.acquire_writer_lock(out_items, ctx)
        cs.release_writer_lock(out_items, tok, fd)
        # released: the DURABLE context survives the release and the lock
        self.assertTrue(os.path.exists(out_items + ".context.json"))
        tok, fd = cs.acquire_writer_lock(out_items, ctx)
        cs.release_writer_lock(out_items, tok, fd)
        # durable context mismatch refuses, ack or not
        with self.assertRaises(SystemExit):
            cs.acquire_writer_lock(out_items, dict(ctx, model="OTHER"))
        # dead writer with in-flight items: flock taken, then the process
        # dies after ledgering submissions (x pending, y done)
        tok, fd = cs.acquire_writer_lock(out_items, ctx)
        cs.ledger_append(out_items, {"state": "pending", "id": "x"})
        cs.ledger_append(out_items, {"state": "pending", "id": "y"})
        cs.ledger_append(out_items, {"state": "done", "id": "y"})
        os.close(fd)  # simulate process death: flock gone, state stays active
        tok, fd = cs.acquire_writer_lock(out_items, ctx)  # dead owner: fine
        with self.assertRaises(SystemExit):  # unknown item x, no ack
            cs.reconcile_ledger(out_items, {"y"}, False)
        cs.reconcile_ledger(out_items, {"y"}, True)  # reconciled, ledgered
        cs.ledger_append(out_items, {"state": "done", "id": "x"})  # rerun did it
        cs.release_writer_lock(out_items, tok, fd)
        with open(out_items + ".ledger.jsonl") as f:
            self.assertIn("reconciled", f.read())
        # manual lock-file removal does NOT erase the pending obligation:
        # the ledger is read under the fresh flock regardless
        cs.ledger_append(out_items, {"state": "pending", "id": "z"})
        os.remove(out_items + ".lock")
        tok, fd = cs.acquire_writer_lock(out_items, ctx)
        with self.assertRaises(SystemExit):
            cs.reconcile_ledger(out_items, {"y"}, False)
        cs.reconcile_ledger(out_items, {"y"}, True)
        cs.ledger_append(out_items, {"state": "done", "id": "z"})
        cs.release_writer_lock(out_items, tok, fd)
        # a checkpointed row (in done_ids) is NOT unknown even without a
        # ledger done (crash between row fsync and ledger append)
        cs.ledger_append(out_items, {"state": "pending", "id": "w"})
        cs.reconcile_ledger(out_items, {"y", "w"}, False)  # w checkpointed: ok
        # a malformed ledger line fails CLOSED, never silently skipped
        with open(out_items + ".ledger.jsonl", "a") as f:
            f.write("{not json\n")
        with self.assertRaises(SystemExit):
            cs._ledger_states(out_items + ".ledger.jsonl")

    # ---- fully mocked MAIN tests (review round 4): normal, resume,
    # interrupt+reconcile, live-writer refusal, concurrency ---------------

    def _main_env(self, td, n=2):
        import json, os
        rows = [{"id": f"T{i}",
                 "prompt": f"def f{i}(a, b):\n    \"\"\"op\"\"\"\n",
                 "test": (f"def check(candidate):\n"
                          f"    assert candidate(2, 3) == 5\n"
                          f"    assert candidate(0, 0) == 0\n"),
                 "entry_point": f"f{i}", "canonical_solution": "    return a + b"}
                for i in range(n)]
        gens = {r["id"]: "    return a + b\n" for r in rows}
        binding = {"items_manifest_sha256": "m1", "responses_sha256": "r1",
                   "dataset_revision": "d" * 40}
        rev = "a" * 40
        manifest = {"created_utc": "t", "argv": {"seed": 0},
                    "gen_max_new_tokens": 512,
                    "models": {"base": {"repo": "org/model", "subfolder": None,
                                        "revision": rev,
                                        "effective_eos_ids": [1, 2]}},
                    "tokenizer": {"repo": "org/model", "revision": rev},
                    "environment": {"python": "3.11.0"},
                    "tasks": {"humaneval": {"kind": "generative",
                                            "counters": {"base": {
                                                "skipped_overlong_prompt": 0,
                                                "extraction_failures": 0}}}},
                    "lock": {"datasets": {"openai/openai_humaneval": "d" * 40},
                             "models": {"base": rev}},
                    "suite_source_sha256": {}}
        with open(os.path.join(td, "manifest.json"), "w") as f:
            json.dump(manifest, f)
        return rows, gens, binding

    def _run_main(self, td, rows, gens, binding, extra_argv=(), expect_code=0):
        import coding_exec as cx, coding_score as cs
        from unittest import mock
        argv = ["--run-dir", td, "--model", "base", *extra_argv]
        with mock.patch.dict(sys.modules, {"modal": types.SimpleNamespace(__version__=cx.REQUIRED_MODAL_VERSION)}), \
             mock.patch.object(cs, "load_and_bind",
                               return_value=(rows, gens, binding)), \
             mock.patch.object(cx, "SandboxBackend",
                               lambda *a, **k: cx.LocalBackend(fixture_mode=True)):
            with self.assertRaises(SystemExit) as cm:
                cs.main(argv)
        if expect_code is not None:
            self.assertEqual(cm.exception.code, expect_code)
        return cm.exception.code

    def _cell(self, td):
        import json, os
        with open(os.path.join(td, "cell_humaneval__base.json")) as f:
            return json.load(f)

    def _scored(self, td):
        import json, os
        p = os.path.join(td, "items_humaneval_scored__base.jsonl")
        with open(p) as f:
            return [json.loads(l) for l in f]

    def test_main_mocked_positive_run(self):
        import tempfile, os, json
        td = tempfile.mkdtemp()
        rows, gens, binding = self._main_env(td, n=2)
        self._run_main(td, rows, gens, binding)
        cell = self._cell(td)
        self.assertEqual(cell["state"], "complete_valid")
        self.assertEqual(cell["n_scored"], 2)
        self.assertEqual(cell["run_context"]["model_revision"], "a" * 40)
        self.assertTrue(cell["run_context"]["allow_local_python"])
        recs = self._scored(td)
        self.assertEqual(len(recs), 2)
        self.assertTrue(all(r["status"] == "passed" and r["correct"] == 1
                            for r in recs))
        # final validation accepted the phase-evidenced rows (round-4 bug:
        # the final pass used to reject them via the missing per-item map)
        self.assertIsNone(cell["aborted"])
        # the exact sandbox ID is ledgered per item the moment create
        # returns (review round 6: reconciliation binds to THAT sandbox)
        with open(os.path.join(td, "items_humaneval_scored__base.jsonl"
                               ".ledger.jsonl")) as f:
            entries = [json.loads(l) for l in f]
        created = [e for e in entries if e.get("state") == "created"]
        self.assertEqual(len(created), 2)
        self.assertTrue(all("sandbox_id" in e for e in created))

    def test_main_mocked_resume(self):
        import tempfile
        td = tempfile.mkdtemp()
        rows, gens, binding = self._main_env(td, n=2)
        self._run_main(td, rows, gens, binding)
        self._run_main(td, rows, gens, binding)
        cell = self._cell(td)
        self.assertEqual(cell["state"], "complete_valid")
        self.assertEqual(cell["n_resumed"], 2)
        self.assertEqual(cell["n_executed_this_invocation"], 0)
        self.assertEqual(len(self._scored(td)), 2)  # no duplicate rows

    def test_main_mocked_interrupt_reconcile(self):
        import coding_score as cs, tempfile, os, json
        td = tempfile.mkdtemp()
        rows, gens, binding = self._main_env(td, n=2)
        out_items = os.path.join(td, "items_humaneval_scored__base.jsonl")
        # interrupted prior run: T0 submitted, never checkpointed; the lock
        # file itself was manually removed - the obligation must persist
        cs.ledger_append(out_items, {"state": "pending", "id": "T0"})
        code = self._run_main(td, rows, gens, binding, expect_code=None)
        self.assertNotEqual(code, 0)
        self.assertFalse(os.path.exists(out_items))  # refused before any item
        self._run_main(td, rows, gens, binding,
                       extra_argv=["--ack-interrupted"])
        cell = self._cell(td)
        self.assertEqual(cell["state"], "complete_valid")
        with open(out_items + ".ledger.jsonl") as f:
            self.assertIn("reconciled", f.read())

    def test_main_mocked_live_writer_refuses_even_with_ack(self):
        import coding_score as cs, tempfile, os
        td = tempfile.mkdtemp()
        rows, gens, binding = self._main_env(td, n=1)
        out_items = os.path.join(td, "items_humaneval_scored__base.jsonl")
        tok, fd = cs.acquire_writer_lock(out_items, {"model": "base"})
        try:
            code = self._run_main(td, rows, gens, binding,
                                  extra_argv=["--ack-interrupted"],
                                  expect_code=None)
            self.assertNotEqual(code, 0)  # ack NEVER overrides a live writer
            self.assertFalse(os.path.exists(out_items))
        finally:
            cs.release_writer_lock(out_items, tok, fd)
        os.remove(out_items + ".context.json")  # operator cleanup of our fake
        self._run_main(td, rows, gens, binding)  # and a normal run still works

    def test_main_mocked_concurrency(self):
        import tempfile
        td = tempfile.mkdtemp()
        rows, gens, binding = self._main_env(td, n=4)
        self._run_main(td, rows, gens, binding,
                       extra_argv=["--concurrency", "4"])
        cell = self._cell(td)
        self.assertEqual(cell["state"], "complete_valid")
        self.assertEqual(cell["n_scored"], 4)

    def test_main_mocked_staggered_two_invocations_no_rerun(self):
        # review round 5 TOCTOU regression: B starts while A holds the
        # flock; B must be REFUSED as a live writer (never rerun paid
        # items), and the file keeps exactly one row per id. On the old
        # code B snapshotted pre-lock and duplicated A's rows.
        import coding_exec as cx, coding_score as cs, threading, tempfile, os
        from unittest import mock
        td = tempfile.mkdtemp()
        rows, gens, binding = self._main_env(td, n=3)
        argv = ["--run-dir", td, "--model", "base"]
        codes = []
        first_acquired = threading.Event()
        real_acquire = cs.acquire_writer_lock

        def wrapped(out_items, ctx):
            r = real_acquire(out_items, ctx)
            first_acquired.set()  # A (whoever wins) holds the flock now
            return r

        def run(wait_for_winner):
            if wait_for_winner:
                first_acquired.wait(30)  # B starts only once A holds the lock
            with mock.patch.object(cs, "load_and_bind",
                                   return_value=(rows, gens, binding)), \
                 mock.patch.object(cx, "SandboxBackend",
                                   lambda *a, **k: cx.LocalBackend(fixture_mode=True)):
                try:
                    cs.main(argv)
                except SystemExit as e:
                    codes.append(e.code)

        with mock.patch.dict(sys.modules, {"modal": types.SimpleNamespace(__version__=cx.REQUIRED_MODAL_VERSION)}), \
             mock.patch.object(cs, "acquire_writer_lock", wrapped):
            a = threading.Thread(target=run, args=(False,))
            b = threading.Thread(target=run, args=(True,))
            a.start()
            b.start()
            a.join(60)
            b.join(60)
        self.assertEqual(len(codes), 2)
        self.assertIn(0, codes)  # the winner completed
        refused = [c for c in codes if c != 0]
        self.assertEqual(len(refused), 1)
        self.assertIn("LIVE writer", str(refused[0]))  # B refused, no rerun
        recs = self._scored(td)
        self.assertEqual(len(recs), 3)
        self.assertEqual(len({r["id"] for r in recs}), 3)  # no duplicates
        states = cs._ledger_states(os.path.join(
            td, "items_humaneval_scored__base.jsonl.ledger.jsonl"))
        self.assertEqual(states, {"T0": "done", "T1": "done", "T2": "done"})

    def test_sandbox_created_callback_failure_still_terminates(self):
        # review round 6b: if the on_create ledger callback raises, the
        # explicit terminate in the backend's finally must STILL run (not
        # only the 30s server-side backstop)
        import coding_exec as cx, sys, types, io
        calls = {"terminated": 0, "created_cb": 0}

        class FakeProc:
            returncode = 0
            stdout = io.StringIO("")
            def wait(self):
                return 0

        class FakeFile:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def write(self, *a):
                pass

        class FakeSandbox:
            object_id = "sb-fake-1"
            def exec(self, *a, **k):
                return FakeProc()
            def open(self, *a, **k):
                return FakeFile()
            def terminate(self, wait=False):
                calls["terminated"] += 1

        fake_modal = types.SimpleNamespace(
            __version__=cx.REQUIRED_MODAL_VERSION,
            App=types.SimpleNamespace(lookup=lambda *a, **k: object()),
            Image=types.SimpleNamespace(from_registry=lambda *a, **k: object()),
            Sandbox=types.SimpleNamespace(create=lambda *a, **k: FakeSandbox()))
        old = sys.modules.get("modal")
        sys.modules["modal"] = fake_modal
        try:
            def boom(_sid):
                calls["created_cb"] += 1
                raise RuntimeError("ledger write failed")
            with self.assertRaises(RuntimeError):
                cx.SandboxBackend().run_driver("src", b"mod", "{}",
                                               on_create=boom)
        finally:
            if old is None:
                del sys.modules["modal"]
            else:
                sys.modules["modal"] = old
        self.assertEqual(calls["created_cb"], 1)
        self.assertEqual(calls["terminated"], 1)  # explicit cleanup attempted

    def test_gen_manifest_validation(self):
        import coding_score as cs
        rev = "a" * 40
        good = {"models": {"base": {"repo": "org/model", "revision": rev,
                                    "effective_eos_ids": [1]}},
                "tokenizer": {"repo": "org/model", "revision": rev},
                "environment": {"python": "3.11.0"},
                "gen_max_new_tokens": 512, "argv": {"seed": 0},
                "tasks": {"humaneval": {"counters": {"base": {}}}},
                "lock": {"datasets": {"openai/openai_humaneval": "d" * 40},
                         "models": {"base": rev}}}
        self.assertEqual(cs.validate_gen_manifest(good, "base", "d" * 40), rev)
        bads = [
            {},                                  # empty: unfinalized
            {"models": {}},                      # no model entry
            dict(good, models={"base": {"repo": "org/model",
                                        "revision": "unknown"}}),
            dict(good, models={"base": {"revision": rev}}),       # no repo
            {k: v for k, v in good.items() if k != "tokenizer"},  # no tok pin
            {k: v for k, v in good.items() if k != "environment"},
            dict(good, tasks={}),                # no humaneval coverage
            dict(good, lock={"datasets": {"openai/openai_humaneval": "e" * 40},
                             "models": {"base": rev}}),           # pin drift
            dict(good, lock={"datasets": {"openai/openai_humaneval": "d" * 40},
                             "models": {"base": "b" * 40}}),      # lock != entry
            dict(good, models={"base": {"repo": "org/model",
                                        "revision": rev}}),  # no eos: unfinished
            dict(good, tasks={"humaneval": {"counters": {}}}),  # no counters
            dict(good, gen_max_new_tokens=None),             # config unreconciled
            dict(good, argv={}),                           # no argv.seed
        ]
        for bad in bads:
            with self.assertRaises(SystemExit, msg=str(bad)[:80]):
                cs.validate_gen_manifest(bad, "base", "d" * 40)

    def test_driver_pins_are_stable(self):
        import coding_exec as cx
        self.assertEqual(cx.driver_sha256(), cx.driver_sha256())
        for marker in ("check_correctness", "HESTARTUP::", "HEVERDICT::",
                       "EXPECTED_MODULE_SHA256"):
            self.assertIn(marker, cx.driver_source())

    def test_reference_module_pinned(self):
        import coding_exec as cx, hashlib, os
        self.assertTrue(os.path.exists(cx.REFERENCE_PATH))
        with open(cx.REFERENCE_PATH, "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(),
                             cx.REFERENCE_SHA256)
        # the driver embeds the same pin for in-sandbox verification
        self.assertIn(cx.REFERENCE_SHA256, cx.driver_source())
        # the vendored module is the real upstream: it exposes check_correctness
        with open(cx.REFERENCE_PATH, encoding="utf-8") as f:
            src = f.read()
        for marker in ("def check_correctness(", "def unsafe_execute(",
                       "def reliability_guard(", "multiprocessing"):
            self.assertIn(marker, src)

    def test_image_digest_pinned(self):
        import coding_exec as cx
        self.assertIn("@sha256:", cx.SANDBOX_IMAGE)
        self.assertIn("python:3.13-slim", cx.SANDBOX_IMAGE)

    def test_backend_enforces_sdk_pin_and_no_unpinned_layer(self):
        import coding_exec as cx, inspect
        src = inspect.getsource(cx.SandboxBackend)
        # no add_python layer: an added layer on the digest base would be
        # unpinned - the from_registry call must carry no extra arguments
        self.assertIn("modal.Image.from_registry(SANDBOX_IMAGE)", src)
        # the pinned SDK version is compared against the actual at every use
        self.assertIn("REQUIRED_MODAL_VERSION", src)

    def test_humaneval_loader_validation(self):
        # bad revision rejected before any load
        with self.assertRaises(ValueError):
            tasks.humaneval_load(revision="latest")






# ---- export_check: durable export + verification (offline, no Modal) ----
import export_check as ec
import os, tempfile, json as _json


def _manifest(models=("base", "v2"), tasks=("mmlu", "gsm8k")):
    kinds = {"mmlu": "mcq", "gsm8k": "generative", "ifeval_custom": "generative"}
    return {"argv": {"models": list(models), "tasks": list(tasks)},
            "tasks": {t: {"kind": kinds[t], "selected": 3} for t in tasks}}


def _make_run(dirpath, manifest=None, with_complete=True, with_failed=False,
              omit=(), empty=(), break_summary_cell=None, drop_item_row=None,
              overrides=None):
    """Semantically valid synthetic run mirroring the runner's real layout."""
    if manifest is None:
        manifest = _manifest()
    models = manifest["argv"]["models"]
    tasks = manifest["argv"]["tasks"]
    kinds = {t: manifest["tasks"][t]["kind"] for t in tasks}
    files = {"manifest.json": _json.dumps(manifest), "paired_vs_base.json": "{}"}
    if len(models) <= 1:
        del files["paired_vs_base.json"]
    # summary: one object per line, metric cell per task__model
    lines = []
    for t in tasks:
        for m in models:
            cell = {"acc": 0.5, "n_scored": 3, "skipped_overlong_prompt": 0}
            if break_summary_cell == (t, m):
                cell = {"acc": 0.5}  # n_scored missing
            lines.append(_json.dumps({"%s__%s" % (t, m): cell}))
            lines.append(_json.dumps({"%s__%s_sec" % (t, m): 1.5}))
    files["summary.jsonl"] = "\n".join(lines) + "\n"
    for t in tasks:
        ids = ["%s-%d" % (t, i) for i in range(3)]
        files["items_manifest_%s.json" % t] = _json.dumps(
            [{"id": i, "answer": 1} for i in ids])
        for m in models:
            irows = [{"id": i, "correct": 1} for i in ids]
            if drop_item_row == (t, m):
                irows = irows[:2]
            files["items_%s__%s.jsonl" % (t, m)] = "".join(
                _json.dumps(r) + "\n" for r in irows)
            if kinds[t] == "generative":
                files[os.path.join("responses", "%s__%s.jsonl" % (t, m))] = "".join(
                    _json.dumps({"id": i, "response": "Answer: 1"}) + "\n"
                    for i in ids)
    for rel, content in files.items():
        if rel in omit:
            continue
        if overrides and rel in overrides:
            content = overrides[rel]
        fp = os.path.join(dirpath, rel)
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, "w") as f:
            f.write("" if rel in empty else content)
    if with_complete:
        open(os.path.join(dirpath, "COMPLETE"), "w").write("ok\n")
    if with_failed:
        open(os.path.join(dirpath, "FAILED"), "w").write("boom\n")


class ExportCheckTests(unittest.TestCase):
    def test_export_restore_roundtrip_verified(self):
        with tempfile.TemporaryDirectory() as td:
            src, dst = os.path.join(td, "run-a"), os.path.join(td, "vol", "run-a")
            _make_run(src)  # manifest+summary+paired+2 task manifests+4 items+2 responses+COMPLETE = 12
            rep = ec.export_and_verify(src, dst)
            self.assertEqual(rep["state"], ec.COMPLETE_VALID, rep)
            self.assertTrue(rep["semantic_ok"], rep["semantic_problems"])
            self.assertTrue(rep["export"]["verified"], rep["export"]["problems"])
            self.assertEqual(rep["n_files"], 12)
            self.assertEqual(rep["expected_missing"], [])

    def test_mcq_task_expects_no_responses_file(self):
        exp = ec.expected_files(_manifest(models=("base",), tasks=("mmlu",)))
        self.assertNotIn("responses/mmlu__base.jsonl", exp)
        self.assertIn("items_mmlu__base.jsonl", exp)
        self.assertNotIn("paired_vs_base.json", exp)

    def test_generative_task_expects_responses(self):
        exp = ec.expected_files(_manifest(models=("base", "v2"), tasks=("gsm8k",)))
        self.assertIn(os.path.join("responses", "gsm8k__v2.jsonl"), exp)
        self.assertIn("paired_vs_base.json", exp)

    def test_complete_without_manifest_is_hard_invalid(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "run-a")
            os.makedirs(src)
            open(os.path.join(src, "COMPLETE"), "w").write("ok\n")
            rep = ec.write_index(src)
            self.assertEqual(rep["state"], ec.COMPLETE_INVALID)
            ok, _ = ec.verify_against_index(src, rep["files"])
            self.assertTrue(ok)  # integrity passes; completeness still fails

    def test_malformed_manifest_is_hard_invalid(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "run-a")
            _make_run(src)
            with open(os.path.join(src, "manifest.json"), "w") as f:
                f.write("{not json")
            self.assertEqual(ec.write_index(src)["state"], ec.COMPLETE_INVALID)

    def test_null_task_entry_reports_not_crashes(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "run-a")
            m = _manifest()
            _make_run(src, manifest=m)
            m["tasks"]["mmlu"] = None
            with open(os.path.join(src, "manifest.json"), "w") as f:
                f.write(_json.dumps(m))
            rep = ec.write_index(src)
            self.assertEqual(rep["state"], ec.COMPLETE_INVALID)
            self.assertFalse(rep["manifest_ok"])
            self.assertTrue(any("mmlu" in p for p in rep["manifest_problems"]))

    def test_exactly_one_terminal_marker(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "run-a")
            _make_run(src, with_complete=True, with_failed=True)
            self.assertEqual(ec.write_index(src)["state"], ec.INVALID)
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "run-a")
            _make_run(src, with_complete=False, with_failed=False)
            self.assertEqual(ec.write_index(src)["state"], ec.INVALID)

    def test_missing_expected_makes_complete_invalid(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "run-a")
            _make_run(src, omit=(os.path.join("responses", "gsm8k__v2.jsonl"),))
            rep = ec.write_index(src)
            self.assertEqual(rep["state"], ec.COMPLETE_INVALID)
            self.assertEqual(rep["expected_missing"],
                             [os.path.join("responses", "gsm8k__v2.jsonl")])

    def test_failure_path_is_partial_not_complete(self):
        with tempfile.TemporaryDirectory() as td:
            src, dst = os.path.join(td, "run-a"), os.path.join(td, "dst")
            _make_run(src, with_complete=False, with_failed=True,
                      omit=("paired_vs_base.json",
                            os.path.join("responses", "gsm8k__base.jsonl")))
            rep = ec.export_and_verify(src, dst)
            self.assertEqual(rep["state"], ec.PARTIAL_FAILURE)
            self.assertTrue(rep["export"]["verified"], rep["export"]["problems"])
            self.assertIn("paired_vs_base.json", rep["expected_missing"])

    def test_corruption_detected(self):
        with tempfile.TemporaryDirectory() as td:
            src, dst = os.path.join(td, "run-a"), os.path.join(td, "dst")
            _make_run(src)
            rep = ec.write_index(src)
            ec.export_tree(src, dst)
            with open(os.path.join(dst, "summary.jsonl"), "a") as f:
                f.write("tampered\n")
            ok, problems = ec.verify_against_index(dst, rep["files"])
            self.assertFalse(ok)
            self.assertTrue(any("summary.jsonl" in p for p in problems), problems)

    def test_export_refuses_same_dir(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "run-a")
            _make_run(src)
            ec.write_index(src)
            with self.assertRaises(ValueError):
                ec.export_tree(src, src)

    def test_manifest_missing_kind_invalid(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "run-a")
            m = _manifest()
            _make_run(src, manifest=m)
            del m["tasks"]["mmlu"]["kind"]
            with open(os.path.join(src, "manifest.json"), "w") as f:
                f.write(_json.dumps(m))
            rep = ec.write_index(src)
            self.assertEqual(rep["state"], ec.COMPLETE_INVALID)
            self.assertTrue(any("kind" in p for p in rep["manifest_problems"]))

    def test_empty_expected_files_fail_semantically(self):
        # the reviewer's repro: valid manifest + all expected files present
        # but EMPTY must not pass complete_valid
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "run-a")
            _make_run(src, empty=("summary.jsonl", "items_mmlu__base.jsonl",
                                  "items_manifest_mmlu.json"))
            rep = ec.write_index(src)
            self.assertEqual(rep["state"], ec.COMPLETE_INVALID)
            self.assertFalse(rep["semantic_ok"])
            self.assertTrue(rep["semantic_problems"])

    def test_summary_cell_missing_nscored(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "run-a")
            _make_run(src, break_summary_cell=("gsm8k", "v2"))
            rep = ec.write_index(src)
            self.assertEqual(rep["state"], ec.COMPLETE_INVALID)
            self.assertTrue(any("n_scored" in p for p in rep["semantic_problems"]))

    def test_item_count_mismatch_detected(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "run-a")
            _make_run(src, drop_item_row=("mmlu", "base"))
            rep = ec.write_index(src)
            self.assertEqual(rep["state"], ec.COMPLETE_INVALID)
            self.assertTrue(any("rows" in p for p in rep["semantic_problems"]))

    def test_duplicate_item_id_fails(self):
        # reviewer repro: manifest ids [a,b], items [a,a], responses [a,b],
        # n_scored 2 must NOT pass complete_valid
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "run-a")
            m = {"argv": {"models": ["base"], "tasks": ["gsm8k"]},
                 "tasks": {"gsm8k": {"kind": "generative", "selected": 2}}}
            _make_run(src, manifest=m, overrides={
                "items_manifest_gsm8k.json": _json.dumps(
                    [{"id": "a"}, {"id": "b"}]),
                "items_gsm8k__base.jsonl": "".join(
                    _json.dumps(r) + "\n" for r in
                    [{"id": "a", "correct": 1}, {"id": "a", "correct": 0}]),
                os.path.join("responses", "gsm8k__base.jsonl"): "".join(
                    _json.dumps(r) + "\n" for r in
                    [{"id": "a", "response": "1"}, {"id": "b", "response": "2"}]),
                "summary.jsonl": _json.dumps(
                    {"gsm8k__base": {"acc": 0.5, "n_scored": 2}}) + "\n",
            })
            rep = ec.write_index(src)
            self.assertEqual(rep["state"], ec.COMPLETE_INVALID)
            self.assertTrue(
                any("duplicate" in p or "does not match" in p
                    for p in rep["semantic_problems"]),
                rep["semantic_problems"])

    def test_malformed_id_reports_not_crashes(self):
        # id=[] must produce an invalid report, never a TypeError
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "run-a")
            _make_run(src, overrides={
                "items_manifest_mmlu.json": _json.dumps(
                    [{"id": []}, {"id": "mmlu-1"}, {"id": "mmlu-2"}]),
            })
            rep = ec.write_index(src)
            self.assertEqual(rep["state"], ec.COMPLETE_INVALID)
            self.assertTrue(any("malformed id" in p
                                for p in rep["semantic_problems"]),
                            rep["semantic_problems"])

    def test_scored_row_missing_correctness_fails(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "run-a")
            _make_run(src, overrides={
                "items_mmlu__base.jsonl": "".join(
                    _json.dumps(r) + "\n" for r in
                    [{"id": "mmlu-0", "correct": 1},
                     {"id": "mmlu-1"},  # scored row without correctness
                     {"id": "mmlu-2", "correct": 0}]),
            })
            rep = ec.write_index(src)
            self.assertEqual(rep["state"], ec.COMPLETE_INVALID)
            self.assertTrue(any("correctness" in p
                                for p in rep["semantic_problems"]),
                            rep["semantic_problems"])

    def test_skipped_row_with_correctness_fails(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "run-a")
            _make_run(src, overrides={
                "items_mmlu__base.jsonl": "".join(
                    _json.dumps(r) + "\n" for r in
                    [{"id": "mmlu-0", "correct": 1},
                     {"id": "mmlu-1", "skipped": "prompt_overlong",
                      "correct": 0},
                     {"id": "mmlu-2", "correct": 0}]),
                "summary.jsonl": "\n".join([
                    _json.dumps({"mmlu__base": {"acc": 0.5, "n_scored": 2}}),
                    _json.dumps({"mmlu__base_sec": 1.5}),
                    _json.dumps({"mmlu__v2": {"acc": 0.5, "n_scored": 3}}),
                    _json.dumps({"mmlu__v2_sec": 1.5}),
                    _json.dumps({"gsm8k__base": {"acc": 0.5, "n_scored": 3}}),
                    _json.dumps({"gsm8k__base_sec": 1.5}),
                    _json.dumps({"gsm8k__v2": {"acc": 0.5, "n_scored": 3}}),
                    _json.dumps({"gsm8k__v2_sec": 1.5}),
                ]) + "\n",
            })
            rep = ec.write_index(src)
            self.assertEqual(rep["state"], ec.COMPLETE_INVALID)
            self.assertTrue(any("skipped rows carrying" in p
                                for p in rep["semantic_problems"]),
                            rep["semantic_problems"])

    def test_response_field_missing_fails_empty_string_ok(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "run-a")
            _make_run(src, overrides={
                os.path.join("responses", "gsm8k__base.jsonl"): "".join(
                    _json.dumps(r) + "\n" for r in
                    [{"id": "gsm8k-0"},  # response field missing
                     {"id": "gsm8k-1", "response": "2"},
                     {"id": "gsm8k-2", "response": "3"}]),
            })
            rep = ec.write_index(src)
            self.assertEqual(rep["state"], ec.COMPLETE_INVALID)
            self.assertTrue(any("missing response field" in p
                                for p in rep["semantic_problems"]),
                            rep["semantic_problems"])
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "run-a")
            _make_run(src, overrides={
                os.path.join("responses", "gsm8k__base.jsonl"): "".join(
                    _json.dumps(r) + "\n" for r in
                    [{"id": "gsm8k-0", "response": ""},  # empty is valid
                     {"id": "gsm8k-1", "response": "2"},
                     {"id": "gsm8k-2", "response": "3"}]),
            })
            rep = ec.write_index(src)
            self.assertEqual(rep["state"], ec.COMPLETE_VALID,
                             rep["semantic_problems"])

    def test_runner_done_record_is_valid(self):
        # live-run repro: run.py terminates summary.jsonl with {"DONE": true};
        # that terminal record must not fail semantic validation
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "run-a")
            _make_run(src)
            with open(os.path.join(src, "summary.jsonl"), "a") as f:
                f.write(_json.dumps({"DONE": True}) + "\n")
            rep = ec.write_index(src)
            self.assertEqual(rep["state"], ec.COMPLETE_VALID,
                             rep["semantic_problems"])

    def test_unexpected_summary_cell_flagged(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "run-a")
            _make_run(src)
            with open(os.path.join(src, "summary.jsonl"), "a") as f:
                f.write(_json.dumps({"gsm8k__v3": {"acc": 1.0, "n_scored": 3}}) + "\n")
            rep = ec.write_index(src)
            self.assertEqual(rep["state"], ec.COMPLETE_INVALID)
            self.assertTrue(any("unexpected cell" in p for p in rep["semantic_problems"]))


# ---- rescore: markup-aware POST-HOC parser (gold-independent fixtures) ----
import rescore


class TestMarkupRescore(unittest.TestCase):
    def test_plain_marker_unchanged(self):
        v, route, flags = rescore.markup_extract("2+2=4.\nAnswer: 42")
        self.assertEqual((route, flags, float(v)), ("marker", [], 42.0))

    def test_bold_answer_label_recovered(self):
        v, route, flags = rescore.markup_extract("reasoning\n**Answer:** 20")
        self.assertEqual((route, float(v)), ("marker_markup", 20.0))
        self.assertIn("markup_label", flags)

    def test_escaped_currency_recovered(self):
        v, route, flags = rescore.markup_extract("**Answer:** \\$12.00")
        self.assertEqual((route, float(v)), ("marker_markup", 12.0))
        self.assertIn("currency", flags)

    def test_unfinished_marker_is_none_no_last_number_loosening(self):
        # cutoff after the marker: must NOT grab the earlier number 3
        v, route, flags = rescore.markup_extract(
            "Wilfred is 3 years old.\n**Answer: ")
        self.assertEqual((v, route), (None, "none"))
        self.assertIn("unfinished_marker", flags)

    def test_unparseable_marker_tail_never_falls_back(self):
        # a marker with junk tail must not loosen to a number elsewhere
        v, route, _ = rescore.markup_extract("**Answer:** unknown. Earlier 13.")
        self.assertEqual((v, route), (None, "none"))

    def test_suffix_units_still_rejected(self):
        v, route, _ = rescore.markup_extract("**Answer:** 42 bananas")
        self.assertEqual((v, route), (None, "none"))

    def test_no_marker_keeps_original_fallback(self):
        v, route, _ = rescore.markup_extract("the total is 7 and then 9")
        self.assertEqual((route, float(v)), ("fallback", 9.0))

    def test_last_marker_wins(self):
        v, route, _ = rescore.markup_extract(
            "Answer: 10\nreconsidering\n**Answer:** 20")
        self.assertEqual((route, float(v)), ("marker_markup", 20.0))

    def test_comma_and_fraction_grammar_preserved(self):
        v, _, _ = rescore.markup_extract("**ANSWER:** 1,300")
        self.assertEqual(float(v), 1300.0)
        v, _, _ = rescore.markup_extract("**Answer:** 3/4")
        self.assertEqual(float(v), 0.75)

    def test_exponent_still_rejected(self):
        v, route, _ = rescore.markup_extract("**Answer:** 1e5")
        self.assertEqual((v, route), (None, "none"))

    def test_last_marker_wins_across_plain_and_markup(self):
        # colon OUTSIDE the bold: the frozen parser only saw the plain marker
        # and returned 10; the later emphasized marker must win
        v, route, _ = rescore.markup_extract("Answer: 10 then **Answer**: 20")
        self.assertEqual((route, float(v)), ("marker_markup", 20.0))
        v, route, _ = rescore.markup_extract("Answer: 10 then _answer_ : 20")
        self.assertEqual((route, float(v)), ("marker_markup", 20.0))
        # later PLAIN marker beats an earlier emphasized one too
        v, route, _ = rescore.markup_extract("**Answer:** 10 then Answer: 20")
        self.assertEqual((route, float(v)), ("marker", 20.0))

    def test_balanced_wrappers_only(self):
        v, route, _ = rescore.markup_extract("**Answer:** **20**")
        self.assertEqual((route, float(v)), ("marker_markup", 20.0))
        for bad in ("**Answer:** 42$", "**Answer:** 42\\", "**Answer:** 20 **"):
            v, route, _ = rescore.markup_extract(bad)
            self.assertEqual((v, route), (None, "none"), bad)

    def test_plain_label_with_currency(self):
        v, route, flags = rescore.markup_extract("Answer: $12.00")
        self.assertEqual((route, float(v)), ("marker", 12.0))
        self.assertIn("currency", flags)

    def test_rescore_generation_exact_gold(self):
        row = {"id": "x", "answer": "20"}
        d = rescore.rescore_generation(row, "**Answer:** 20")
        self.assertEqual(d["correct"], 1)
        d = rescore.rescore_generation(row, "**Answer:** 240")
        self.assertEqual(d["correct"], 0)
        d = rescore.rescore_generation(row, "**Answer: ")
        self.assertEqual(d["correct"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
