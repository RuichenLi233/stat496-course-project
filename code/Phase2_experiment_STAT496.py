import requests
import pandas as pd
import numpy as np
import re
from collections import defaultdict
from tqdm import tqdm

df = pd.read_csv("C:/Users/wyk18/Downloads/balanced_dataset_STAT496.csv")
df.columns = df.columns.str.strip()

print(f"Phase 2 dataset size: {len(df)} questions")
print(df["question_type"].value_counts(), "\n")

temperatures = [0.0, 0.5, 1.0, 2.0]
top_p = 0.9
top_k = 40
runs_per_setting = 3

summary_results = []

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Starting Phase 2 temperature sweep...\n")

for temp in temperatures:

    print(f"Running temperature={temp}")

    run_accuracies = []
    all_outputs = defaultdict(list)

    type_correct_runs = defaultdict(list)

    for run_id in range(runs_per_setting):

        correct_total = 0
        type_correct = defaultdict(int)
        type_total = defaultdict(int)

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

                output_raw = result["response"]
                output = normalize(output_raw)

                all_outputs[(idx, q_type)].append(output)

                is_correct = (
                    true_answer in output or
                    output in true_answer
                )

                type_total[q_type] += 1

                if is_correct:
                    correct_total += 1
                    type_correct[q_type] += 1

            except Exception as e:
                print("API error:", e)
                continue

        run_accuracy = correct_total / len(df)
        run_accuracies.append(run_accuracy)

        for t in type_total:
            type_correct_runs[t].append(
                type_correct[t] / type_total[t]
            )

    avg_accuracy = np.mean(run_accuracies)
    accuracy_variance = np.var(run_accuracies)

    stable_count = sum(
        1 for outputs in all_outputs.values()
        if len(set(outputs)) == 1
    )
    stability_score = stable_count / len(all_outputs)

    disagreement_rate = 1 - stability_score

    avg_unique_outputs = np.mean([
        len(set(outputs)) for outputs in all_outputs.values()
    ])

    row_summary = {
        "temperature": temp,
        "avg_accuracy": avg_accuracy,
        "accuracy_variance": accuracy_variance,
        "stability_score": stability_score,
        "disagreement_rate": disagreement_rate,
        "avg_unique_outputs": avg_unique_outputs
    }

    for t in type_correct_runs:
        row_summary[f"{t}_accuracy"] = np.mean(type_correct_runs[t])
        row_summary[f"{t}_variance"] = np.var(type_correct_runs[t])

    summary_results.append(row_summary)

    pd.DataFrame(summary_results).to_csv(
        "phase2_summary_progress.csv",
        index=False
    )

    print(f"Completed temperature={temp}\n")

print("Phase 2 Completed.\n")

pd.DataFrame(summary_results).to_csv(
    "New_phase2_summary_final.csv",
    index=False
)

print("Saved New_phase2_summary_final.csv")