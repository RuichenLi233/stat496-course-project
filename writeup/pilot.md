# Pilot Experiment Writeup — Sampling Parameters and Factual Consistency

## Project Idea
This pilot experiment explores **Idea 2: Sampling Parameters and Factual Consistency**.

Our main research question is:

**How do sampling parameters (temperature and top-p) influence factual accuracy and consistency in LLM responses?**

The goal of this first pass experiment was not to run a large-scale evaluation, but to confirm that changing sampling settings produces meaningful variation in outputs that can be analyzed later.

---

## What prompts were used

We tested a small set of fact-based questions with verifiable answers, such as:

- What is the capital of China?
- In what year was Valorant officially released?
- What is the tallest mountain on Earth above sea level?
- Who created the Python programming language?
- Which country won the 2018 FIFA World Cup?

We used two prompt styles to see whether formatting affects stability:

1. **One-word / short answer style**

Answer with only the final answer (no explanation).
Question: {q}


2. **Brief explanation style**


Answer and briefly explain in 1-2 sentences.
Question: {q}


Each question was run multiple times under different sampling settings.

---

## What kind of responses were received

We observed clear variation across sampling parameters:

- At **temperature = 0**, responses were generally more consistent and repeated outputs were often identical.
- At higher temperatures (0.7 and 1.2), the model produced more diverse phrasing and occasionally different factual claims.
- Lower **top-p values (0.3)** tended to restrict responses and reduce variation.
- Higher **top-p values (0.9)** allowed more randomness and sometimes increased inconsistency.

In some cases, higher randomness settings led to small factual errors or contradictory answers across repeated runs, demonstrating that sampling parameters directly affect factual stability.

---

## How might you improve your experiment?

This pilot experiment was intentionally small, but several improvements are possible:

- Use a larger and more diverse factual dataset (50–200+ questions).
- Add automated scoring for correctness instead of manual inspection.
- Include more models for comparison (e.g., GPT-style APIs vs. open-source models).
- Expand evaluation metrics beyond observation, such as:
- accuracy rate
- inconsistency rate
- number of unique answers per question
- embedding similarity between outputs

---

## What variables do you intend to vary?

In future experiments, we plan to systematically vary:

- **Temperature** (finer grid, e.g., 0.0 to 1.5)
- **Top-p** (more values between 0.1 and 1.0)
- **Prompt format/style**
- **Model type** (different open-source and API-based LLMs)
- **Number of repeated samples per setting**

This will allow more robust conclusions about how randomness impacts factual consistency.

---

## How will you expand on your starting experiment?

To scale this project into a larger study, we would:

- Build a benchmark-style dataset of factual QA prompts with gold answers
- Run experiments across multiple models and parameter combinations
- Produce summary tables and plots showing:
- accuracy vs. temperature
- diversity vs. top-p
- stability differences across prompt styles

This would provide a clearer understanding of when LLM outputs become unreliable.

---

## How might you automate large-scale data collection?

Large-scale automation could be done by:

- Writing scripts that batch-run thousands of prompt evaluations
- Logging outputs automatically into structured JSONL files
- Adding evaluation pipelines that compute correctness and diversity metrics
- Using APIs (OpenAI/Anthropic) or local inference tools (GPT4All/Ollama) for scalable runs
- Running parameter sweeps over full grids of temperature × top-p

This would enable reproducible experiments and more statistically meaningful results.