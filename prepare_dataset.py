import csv
import json
import sys

# Usage:
# python prepare_dataset.py input.csv output.jsonl
# CSV columns: system,user,assistant

inp = sys.argv[1]
out = sys.argv[2]

with open(inp, newline="", encoding="utf-8") as f_in, open(out, "w", encoding="utf-8") as f_out:
    reader = csv.DictReader(f_in)
    for row in reader:
        record = {
            "messages": [
                {"role": "system", "content": row["system"]},
                {"role": "user", "content": row["user"]},
                {"role": "assistant", "content": row["assistant"]},
            ]
        }
        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Wrote {out}")
