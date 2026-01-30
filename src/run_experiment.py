import json
from gpt4all import GPT4All

import json
from gpt4all import GPT4All

MODEL_NAME = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
RUN_LABEL = "Beautiful Basilisks"

QUESTIONS = [
    "What is the capital of China?",
    "In what year was Valorant officially released?",
    "What is the tallest mountain on Earth above sea level?",
    "Who created the Python programming language?",
    "Which country won the 2018 FIFA World Cup?",
]

PROMPT_STYLES = {
    "one_word_or_phrase": "Answer with only the final answer (no explanation).\nQuestion: {q}",
    "explain": "Answer and briefly explain in 1-2 sentences.\nQuestion: {q}"
}

TEMPS = [0, 0.7, 1.2]
TOP_PS = [0.3, 0.9]
RUNS_PER_SETTING = 3

OUTPUT_FILE = "outputs.jsonl"

model = GPT4All(MODEL_NAME)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    qid = 0
    for base_question in QUESTIONS:
        qid += 1

        for style_name, template in PROMPT_STYLES.items():
            prompt_used = template.format(q=base_question)

            for temp in TEMPS:
                for top_p in TOP_PS:
                    for run in range(1, RUNS_PER_SETTING + 1):
                        print(f"Running Q{qid} style={style_name} temp={temp} top_p={top_p} run={run}")

                        with model.chat_session():  # 每次独立 session，防止 context full
                            answer = model.generate(
                                prompt_used,
                                temp=temp,
                                top_p=top_p,
                                max_tokens=120
                            )

                        record = {
                            "run_label": RUN_LABEL,
                            "model": MODEL_NAME,
                            "qid": qid,
                            "question": base_question,
                            "style": style_name,
                            "prompt_used": prompt_used,
                            "temperature": temp,
                            "top_p": top_p,
                            "run": run,
                            "response": answer.strip()
                        }

                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()

print("Finished. Saved to outputs.jsonl")
