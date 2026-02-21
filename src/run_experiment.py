import os
import csv
import time
from openai import OpenAI

# ======================
# Settings
# ======================
RUN_LABEL = "Beautiful Basilisks"
MODEL_NAME = "gpt-4o-mini"

INPUT_QUESTIONS_CSV = "questions.csv"
OUTPUT_RESULTS_CSV = "results.csv"

PROMPT_STYLES = {
    "short": "Answer with ONLY the final answer. No explanation.\nQuestion: {q}",
    "explain": "Answer the question and briefly explain in 1-2 sentences.\nQuestion: {q}",
}

TEMPS = [0.0, 0.7, 1.2]
TOP_PS = [0.3, 0.9]
RUNS_PER_SETTING = 3

# ======================
# OpenAI client
# ======================
client = OpenAI()

def normalize_text(s: str) -> str:
    # simple normalization for matching
    s = (s or "").strip()
    # remove surrounding quotes
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        s = s[1:-1].strip()
    # unify whitespace
    s = " ".join(s.split())
    # remove ending period for short answers like "Beijing."
    if s.endswith("."):
        s = s[:-1].strip()
    return s.lower()

def ask_model(prompt: str, temperature: float, top_p: float) -> str:
    # Chat Completions style
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
    )
    return resp.choices[0].message.content.strip()

def main():
    # check key exists
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.")
        print("In PowerShell: setx OPENAI_API_KEY \"your_key_here\"  (then reopen PowerShell)")
        return

    # read questions
    rows = []
    with open(INPUT_QUESTIONS_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = ["qid", "category", "answer_type", "question", "correct_answer"]
        for col in required:
            if col not in reader.fieldnames:
                print(f"ERROR: missing column '{col}' in {INPUT_QUESTIONS_CSV}")
                print("Expected columns:", required)
                print("Found columns:", reader.fieldnames)
                return
        for r in reader:
            rows.append(r)

    # write output header
    out_fields = [
        "run_label", "model", "qid", "category", "answer_type",
        "question", "correct_answer",
        "style", "temperature", "top_p", "run",
        "response", "response_len",
        "is_exact_match"
    ]

    with open(OUTPUT_RESULTS_CSV, "w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=out_fields)
        writer.writeheader()

        for r in rows:
            qid = r["qid"]
            category = r["category"]
            answer_type = r["answer_type"]
            question = r["question"]
            gold = r["correct_answer"]

            for style_name, template in PROMPT_STYLES.items():
                prompt_used = template.format(q=question)

                for temp in TEMPS:
                    for top_p in TOP_PS:
                        for run in range(1, RUNS_PER_SETTING + 1):
                            print(f"Q{qid} style={style_name} temp={temp} top_p={top_p} run={run}")

                            try:
                                answer = ask_model(prompt_used, temp, top_p)
                            except Exception as e:
                                answer = f"ERROR: {e}"

                            resp_norm = normalize_text(answer)
                            gold_norm = normalize_text(gold)
                            is_match = 1 if resp_norm == gold_norm else 0

                            record = {
                                "run_label": RUN_LABEL,
                                "model": MODEL_NAME,
                                "qid": qid,
                                "category": category,
                                "answer_type": answer_type,
                                "question": question,
                                "correct_answer": gold,
                                "style": style_name,
                                "temperature": temp,
                                "top_p": top_p,
                                "run": run,
                                "response": answer,
                                "response_len": len(answer),
                                "is_exact_match": is_match
                            }

                            writer.writerow(record)
                            out.flush()

                            # small pause to be polite
                            time.sleep(0.2)

    print(f"Finished. Saved to {OUTPUT_RESULTS_CSV}")

if __name__ == "__main__":
    main()