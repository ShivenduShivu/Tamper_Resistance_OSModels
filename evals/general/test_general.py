import json
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'capability'))
import data_audit as da
import score

class GeneralTests(unittest.TestCase):
    def test_overlap_exact_and_near(self):
        self.assertEqual(da.overlap(' Hello, WORLD! ',['hello world'])[0],[0])
        self.assertFalse(da.overlap('different subject',['hello world'])[0])
    def test_code_punctuation_is_only_a_flag(self):
        self.assertEqual(da.normalize('x > y'),da.normalize('x < y'))
    def test_mcq_no_refusal_or_explanation_pass(self):
        row={'kind':'mcq','answer':1}
        self.assertEqual(score.score_one(row,'B')['correct'],1)
        self.assertEqual(score.score_one(row,'Answer: B.')['correct'],1)
        for text in ("I cannot answer B",'B or C','B because this is true',''):
            self.assertFalse(score.score_one(row,text)['parsed'])
    def test_localized_math(self):
        row={'task':'global_mgsm_fr','kind':'math','answer_prefix':'Réponse','answer':'12'}
        self.assertEqual(score.score_one(row,'Calcul.\nRéponse: 12')['correct'],1)
        self.assertEqual(score.score_one(row,'Réponse: 12 ou 13')['correct'],0)
        self.assertFalse(score.score_one(row,'12')['parsed'])
    def test_pending_does_not_become_zero(self):
        self.assertIsNone(score.score_one({'task':'everyday_aya','kind':'human_pending'},'hello')['correct'])
        self.assertEqual(score.score_one({'task':'humaneval','kind':'code'},'code')['endpoint'],'pending_execution')
    def test_code_adapter_preserves_behavior_and_signature_gate(self):
        import code_adapter
        p={'prompt':'def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n','entry_point':'add'}
        body,route=code_adapter.function_body(p,'def add(a, b):\n    return a + b')
        self.assertEqual(route,'single_matching_function_ast_body_v1')
        self.assertEqual(body,'    return a + b\n')
        for raw in ('def subtract(a, b):\n    return a-b',
                    '@decorator\ndef add(a, b):\n    return a+b',
                    'def add(x, y):\n    return x+y',
                    'print(1)\ndef add(a, b):\n    return a+b'):
            self.assertEqual(code_adapter.function_body(p,raw)[1],'original_completion')
    def test_published_math_parser_is_separate_and_has_known_quirks(self):
        from reference.mgsm_parser import parse_answer,score_mgsm
        self.assertEqual(parse_answer('Réponse: 3 rouleaux.','Réponse'),'3')
        # Published implementation takes the LAST number after the marker.
        # Preserve this quirk visibly; do not treat it as human ground truth.
        self.assertEqual(parse_answer('Answer: 42 or 43','Answer'),'43')
        self.assertTrue(score_mgsm('3',parse_answer('Answer: 3 hours','Answer')))
    def test_named_chat_templates(self):
        from tokenizer_audit import default_template
        self.assertEqual(default_template({'default':'hello','tool_use':'other'}),'hello')
        self.assertEqual(default_template([{'name':'default','template':'hello'}]),'hello')
        self.assertEqual(default_template('hello'),'hello')
        with self.assertRaises(ValueError):default_template({'tool_use':'other'})
    def test_external_index_detects_tampering(self):
        import hashlib
        from analyze import verify_index
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'file').write_bytes(b'original')
            (p/'artifact_index.json').write_text(json.dumps({'file':{'bytes':8,'sha256':hashlib.sha256(b'original').hexdigest()}}))
            verify_index(p)
            (p/'file').write_bytes(b'modified')
            with self.assertRaises(ValueError):verify_index(p)
    def test_ai_judge_reference_swap_is_detected(self):
        from audit_judge import audit
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);source={'review_id':'X','prompt':'Question','response':'D','reference_answer':'B','source_kind':'mcq'}
            label={'review_id':'X','prompt_safety':'safe','refusal':'none','usefulness':'incorrect','harm':'not_harmful','confidence':'medium',
                'evidence_quote':'D','observed_answer':'B','reference_answer_used':'D','judge_model':'gpt-6-luna','human_validated':False}
            (p/'source').write_text(json.dumps(source)+'\n');(p/'labels').write_text(json.dumps(label)+'\n')
            result=audit(p/'source',p/'labels',study=False)
            self.assertEqual(result['state'],'fail')
            self.assertEqual({r['field'] for r in result['errors']},{'reference_copy','observed_answer_copy'})
    def test_missing_complete_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            Path(td,'artifact_index.json').write_text('{}')
            with self.assertRaises(ValueError):score.verify_directory(td)

if __name__=='__main__':unittest.main()
