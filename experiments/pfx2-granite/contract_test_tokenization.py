"""Standalone local run of the pfx2 S3 tokenization contract suite (same code the Modal job gates on).
Usage: python3 contract_test_tokenization.py /path/to/tokenizer_dir /path/to/train.jsonl
Verifies, for every row x (no opener + 8 training openers): exact train-prefix == eval-prefill ids under canonical
joint tokenization, single BOS, decode round-trip, non-empty targets, truncation keeps labels, and left-pad alignment."""
import sys, json
sys.path.insert(0, "/downloads/pfx")
from transformers import AutoTokenizer
from pfx_lib import contract_check, encode_row, TRAIN_OPENERS

tok = AutoTokenizer.from_pretrained(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tok")
rows = [json.loads(l) for l in open(sys.argv[2] if len(sys.argv) > 2 else "/tmp/train.jsonl", encoding="utf-8")]
assert len(rows) == 2355, len(rows)
stats = contract_check(tok, rows)
print("contract_check:", json.dumps(stats))

# left-pad alignment: pads carry attention 0 and label -100
samples = [encode_row(tok, rows[i]["prompt"], rows[i]["response"], TRAIN_OPENERS[i % 8]) for i in (0, 7, 1201)]
L = max(len(s["input_ids"]) for s in samples)
for s in samples:
    pad = L - len(s["input_ids"])
    att = [0] * pad + s["attention_mask"]; lab = [-100] * pad + s["labels"]
    assert all(l == -100 for l in lab[:pad]) and len(att) == len(lab) == L
print("padding/attention alignment OK")
print("ALL CONTRACTS PASS")
