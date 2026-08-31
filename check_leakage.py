import pandas as pd, json
from sklearn.model_selection import train_test_split

df = pd.read_csv("tickets_en.csv")
df["txt"] = df.subject.fillna("") + " \n " + df.body.fillna("")
_, xte, _, _ = train_test_split(df.txt, df.priority, test_size=0.25,
                                random_state=0, stratify=df.priority)
test_txt = set(df.loc[xte.index, "txt"])

train_txt = set()
for line in open("finetune/data/train.jsonl"):
    msgs = json.loads(line)["messages"]
    train_txt.add(next(m["content"] for m in msgs if m["role"] == "user"))

print(f"overlap: {len(train_txt & test_txt)} of {len(train_txt)}")
