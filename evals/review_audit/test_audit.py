import json
import tempfile
import unittest
from pathlib import Path
import audit

class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root/'source.jsonl'
        self.rows = [dict(id=str(i), condition=c, prompt='safe question', response='answer',
                         prompt_class='safe', classifier_refusal=True, ai_review_refusal=False)
                     for c in ('base','v2') for i in range(8)]
        audit.write(self.source, self.rows)
    def prepare(self):
        out = self.root/'review'
        audit.prepare(self.source, out, n=2, extra=1)
        for name in ('reviewer_A.jsonl','reviewer_B.jsonl'):
            p=out/name
            rows=audit.read(p)
            for r in rows:
                r.update(refusal='none', usefulness='useful', harm='not_harmful')
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return out
    def summarize(self,out,adj=None):
        return audit.summarize(out/'PRIVATE_key.jsonl',out/'reviewer_A.jsonl',out/'reviewer_B.jsonl',adj)
    def test_blinding_and_strata(self):
        out=self.prepare()
        sheet=audit.read(out/'reviewer_A.jsonl')
        self.assertTrue(all(set(r)=={'blind_id','prompt','response','refusal','usefulness','harm','notes'} for r in sheet))
        result=self.summarize(out)
        self.assertEqual(result['reviewed_n'],6)
        self.assertEqual(sum(c['n_reviewed'] for c in result['cells'] if c['stratum']=='random'),4)
        self.assertEqual(sum(c['classifier_false_refusal_n'] for c in result['cells']),6)
    def test_reproducible_sampling(self):
        a,b=self.root/'a',self.root/'b'
        audit.prepare(self.source,a,2,1,0); audit.prepare(self.source,b,2,1,0)
        self.assertEqual((a/'PRIVATE_key.jsonl').read_bytes(),(b/'PRIVATE_key.jsonl').read_bytes())
    def test_duplicate_rejected(self):
        self.source.write_text(json.dumps(self.rows[0])+'\n'+json.dumps(self.rows[0])+'\n')
        with self.assertRaises(ValueError): audit.prepare(self.source,self.root/'bad')
    def test_missing_review_rejected(self):
        out=self.prepare(); (out/'reviewer_A.jsonl').write_text('')
        with self.assertRaises(ValueError): self.summarize(out)
    def test_modified_text_rejected(self):
        out=self.prepare(); p=out/'reviewer_A.jsonl'; p.write_text(p.read_text().replace('answer','different'))
        with self.assertRaises(ValueError): self.summarize(out)
    def test_adjudication_required(self):
        out=self.prepare(); p=out/'reviewer_B.jsonl'; rows=audit.read(p)
        rows[0]['refusal']='full'; p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaises(ValueError): self.summarize(out)
        adj=out/'adjudication.jsonl';audit.write(adj,[rows[0]])
        self.assertEqual(self.summarize(out,adj)['reviewer_disagreement_n'],1)
    def test_corrupt_key_rejected(self):
        out=self.prepare();p=out/'PRIVATE_key.jsonl';p.write_text(p.read_text().replace('answer','other'))
        with self.assertRaises(ValueError): self.summarize(out)
    def test_no_overwrite(self):
        out=self.prepare()
        with self.assertRaises(FileExistsError): audit.prepare(self.source,out)

if __name__=='__main__': unittest.main()
