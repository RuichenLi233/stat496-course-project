import requests
import pandas as pd
import numpy as np
import re
from tqdm import tqdm

df = pd.read_csv("C:/Users/wyk18/Downloads/balanced_dataset_STAT496.csv")
df.columns = df.columns.str.strip()

print(f"Phase 1 dataset size: {len(df)} questions")
print(df["question_type"].value_counts(), "\n")

temperatures = [0.0, 0.5, 1.0]
top_ps = [0.7, 0.9, 1.0]
top_ks = [10, 40, 100]

summary_results = []
detailed_results = []

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Starting Phase 1 full parameter sweep...\n")

for temp in temperatures:
    for top_p in top_ps:
        for top_k in top_ks:

            print(f"Running temp={temp}, top_p={top_p}, top_k={top_k}")

            correct_total = 0
            type_correct = {}
            type_total = {}

            for idx, row in tqdm(df.iterrows(), total=len(df), leave=False):

                question = row["question"]
                q_type = row["question_type"]
                true_answer = normalize(str(row["correct_answer"]))

                payload = {
                    "model": "llama3.1:latest",
                    "prompt": question,
                    "temperature": temp,
                    "top_p": top_p,
                    "top_k": top_k,
                    "stream": False
                }

                try:
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json=payload,
                        timeout=120
                    )

                    result = response.json()
                    if "response" not in result:
                        continue

                    output = normalize(result["response"])

                    is_correct = (
                        true_answer in output or
                        output in true_answer
                    )

                    if is_correct:
                        correct_total += 1

                    type_total[q_type] = type_total.get(q_type, 0) + 1
                    if is_correct:
                        type_correct[q_type] = type_correct.get(q_type, 0) + 1

                    detailed_results.append({
                        "temperature": temp,
                        "top_p": top_p,
                        "top_k": top_k,
                        "question_type": q_type,
                        "question": question,
                        "correct_answer": row["correct_answer"],
                        "model_output": result["response"],
                        "is_correct": int(is_correct)
                    })

                except Exception as e:
                    print("API error:", e)
                    continue

            overall_accuracy = correct_total / len(df)

            row_summary = {
                "temperature": temp,
                "top_p": top_p,
                "top_k": top_k,
                "overall_accuracy": overall_accuracy
            }

            for t in type_total:
                row_summary[f"{t}_accuracy"] = (
                    type_correct.get(t, 0) / type_total[t]
                )

            summary_results.append(row_summary)

            print(f"Completed temp={temp}, top_p={top_p}, top_k={top_k}\n")

            pd.DataFrame(summary_results).to_csv(
                "phase1_summary_progress.csv",
                index=False
            )

pd.DataFrame(summary_results).to_csv(
    "New_phase1_summary_final.csv",
    index=False
)

pd.DataFrame(detailed_results).to_csv(
    "phase1_detailed_results.csv",
    index=False
)

print("Phase 1 Completed.")
print("Saved New_phase1_summary_final.csv")
print("Saved phase1_detailed_results.csv")