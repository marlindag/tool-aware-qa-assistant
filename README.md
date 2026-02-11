Tool-Aware QA Assistant (Take-Home)

Overview

This project implements a question-answering assistant that treats Wikipedia as a reference tool.

The system is designed to avoid two common failure modes in QA assistants:
- Over-searching, which adds latency and cost while signaling uncertainty
- Under-searching, which leads to hallucinated or stale answers when verification matters

Tool use is governed by explicit behavioral constraints in the system prompt and validated through a targeted evaluation suite.

System Design

Assistant Behavior

The assistant follows a structured decision flow:

1. Classify the question
   - Common knowledge
   - Requires specific factual lookup
   - Ambiguous
   - Unanswerable (subjective, future prediction, or personal judgment)

2. Decide whether search is necessary
   - Wikipedia search is used only when verification materially improves correctness
   - Common knowledge, arithmetic, and subjective questions do not trigger search
   - Ambiguity must be resolved before searching

3. Respond
   - Answers are synthesized in the assistant’s own words
   - Tool usage and internal reasoning are never narrated
   - Exactly one clarifying question is asked when required, then the assistant stops

Tooling

The assistant has access to a single tool: search_wikipedia(query).

Tool calls are limited to at most one per request. Retrieved sources are captured by the application layer and are not displayed in the assistant’s response.

Evaluation

The evaluation suite validates behavioral correctness rather than raw factual recall.

Each eval case specifies:
- Whether a tool call is expected
- Maximum allowed tool calls
- Forbidden phrases (e.g., narrating tool use)
- Output shape constraints (sentence count, question termination)

Results Summary

10 out of 10 evaluation cases pass.

The passing cases demonstrate that the assistant:
- Avoids searching for common knowledge
- Uses Wikipedia for precise factual queries that require verification
- Handles fictional entities without treating them as real
- Asks a single clarifying question before answering ambiguous prompts
- Produces concise answers without narrating tool usage

Known Limitation

Tool-use is intentionally constrained to a maximum of one Wikipedia search call per question to keep behavior deterministic and easy to evaluate.

In production, a second retrieval can sometimes improve answer quality when the first result is incomplete or off-target. This implementation prioritizes predictable tool discipline over recovery behavior, which is an explicit tradeoff.

How to Run

1. Set your Anthropic API key:
   ANTHROPIC_API_KEY=your_key_here

2. Run the evaluation suite:
   python evals/run_evals.py

3. Results are written to:
   evals/out/eval_results.json

   The CLI prints Tool used: True/False (N call(s)) to indicate whether Wikipedia search was invoked
