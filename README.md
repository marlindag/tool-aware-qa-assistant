Prompt eng take-home assignment

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

Tool calls are limited per request. Retrieved sources are captured by the application layer and are not displayed in the assistant’s response.

Evaluation

The evaluation suite validates behavioral correctness rather than raw factual recall.

Each eval case specifies:
- Whether a tool call is expected
- Maximum allowed tool calls
- Forbidden phrases (e.g., narrating tool use)
- Output shape constraints (sentence count, question termination)

Results Summary

9 out of 10 evaluation cases pass.

The passing cases demonstrate that the assistant:
- Avoids searching for common knowledge
- Uses Wikipedia appropriately for precise factual queries
- Handles fictional entities without treating them as real
- Asks clarifying questions before answering ambiguous prompts
- Produces concise answers without narrating tool usage

Known Limitation

For the query “How tall is Mount Kilimanjaro?”, the assistant makes two Wikipedia search calls instead of one.

The final answer is correct, but the evaluation enforces a strict maximum of one tool call for single-entity factual queries, causing this case to fail.

How to Run

1. Set your Anthropic API key:
   ANTHROPIC_API_KEY=your_key_here

2. Run the evaluation suite:
   python evals/run_evals.py

3. Results are written to:
   evals/out/eval_results.json
