import pandas as pd
import re

df = pd.read_csv(
    "C:/Users/wyk18/Downloads/S08_question_answer_pairs.txt",
    sep="\t"
)

df.columns = df.columns.str.strip()

print("Original size:", len(df))

df = df.dropna(subset=["Question", "Answer"])

df["Question"] = df["Question"].astype(str).str.strip()
df["Answer"] = df["Answer"].astype(str).str.strip()

df = df[
    (df["Question"] != "") &
    (df["Answer"] != "") &
    (df["Answer"].str.upper() != "NULL")
]

df = df.drop_duplicates(subset=["Question"])

print("After cleaning:", len(df))

def classify_question(q):
    q_lower = q.lower().strip()

    if re.match(r"^(is|are|was|were|do|does|did|can|has|have)\b", q_lower):
        return "binary"

    if (
        q_lower.startswith("when") or
        "what year" in q_lower or
        "in what year" in q_lower or
        "since when" in q_lower or
        "during what period" in q_lower or
        q_lower.startswith("how many") or
        q_lower.startswith("how long") or
        q_lower.startswith("how much") or
        "what percentage" in q_lower or
        q_lower.startswith("how old")
    ):
        return "numerical_temporal"

    return "entity"

df["question_type"] = df["Question"].apply(classify_question)

print("\nFull distribution:")
print(df["question_type"].value_counts())

counts = df["question_type"].value_counts()
min_count = counts.min()

print("\nBalancing each class to:", min_count)

binary_df = df[df["question_type"] == "binary"].sample(n=min_count, random_state=42)
entity_df = df[df["question_type"] == "entity"].sample(n=min_count, random_state=42)
numerical_df = df[df["question_type"] == "numerical_temporal"].sample(n=min_count, random_state=42)

balanced_df = pd.concat([binary_df, entity_df, numerical_df]).reset_index(drop=True)

print("\nBalanced distribution:")
print(balanced_df["question_type"].value_counts())

print("Final dataset size:", len(balanced_df))

balanced_df = balanced_df[["Question", "Answer", "question_type"]]
balanced_df.columns = ["question", "correct_answer", "question_type"]

balanced_df.to_csv("balanced_dataset_STAT496.csv", index=False)

print("\nSaved balanced_dataset_STAT496.csv")