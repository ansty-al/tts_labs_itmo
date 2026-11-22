from pathlib import Path

for name in ["metadata_phones", "metadata_phones_gt"]:
    src = Path("data") / f"{name}.csv"
    lines = [l for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    train, test = [], []
    for line in lines:
        idx = int(line.split("|")[0].split("_")[0])
        (test if idx % 5 == 0 else train).append(line)
    (Path("data") / f"{name}_train.csv").write_text("\n".join(train) + "\n", encoding="utf-8")
    (Path("data") / f"{name}_test.csv").write_text("\n".join(test) + "\n", encoding="utf-8")
    print(name, len(train), len(test))

eval_ids = {l.split("|")[0] for l in Path("data/eval_sentences.csv").read_text(encoding="utf-8").splitlines() if l.strip()}
test_ids = {l.split("|")[0] for l in Path("data/metadata_phones_gt_test.csv").read_text(encoding="utf-8").splitlines() if l.strip()}
print("eval ids not in test:", len(eval_ids - test_ids))