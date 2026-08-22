import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

def convert(input_path, output_path):
    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            record = json.loads(line)
            text = tokenizer.apply_chat_template(
                record["messages"], tokenize=False, add_generation_prompt=False
            )
            f_out.write(json.dumps({"text": text}) + "\n")

convert("raw_train.jsonl", "data/train.jsonl")
convert("raw_valid.jsonl", "data/valid.jsonl")
print("Done: wrote data/train.jsonl and data/valid.jsonl")
