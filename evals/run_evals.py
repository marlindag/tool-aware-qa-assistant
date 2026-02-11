import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from anthropic import Anthropic

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

sys.path.append(SRC_DIR)
from wikipedia import search_wikipedia  # noqa: E402

load_dotenv()

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")

SYSTEM_PROMPT_PATH = os.path.join(PROJECT_ROOT, "prompt", "system_prompt.txt")
EVAL_CASES_PATH = os.path.join(PROJECT_ROOT, "evals", "eval_cases.json")
OUT_DIR = os.path.join(PROJECT_ROOT, "evals", "out")
OUT_PATH = os.path.join(OUT_DIR, "eval_results.json")

TOOLS = [
    {
        "name": "search_wikipedia",
        "description": "Search Wikipedia for relevant pages and return top results with titles and snippets.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The Wikipedia search query."},
                "limit": {
                    "type": "integer",
                    "description": "Max number of results to return.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    }
]


@dataclass
class RunResult:
    id: str
    passed: bool
    tool_used: bool
    tool_call_count: int
    sources: List[str]
    assistant_text: str
    failures: List[str]


def load_system_prompt() -> str:
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def extract_text(resp) -> str:
    parts: List[str] = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()


def strip_model_sources(text: str) -> str:
    """Defensive: remove any model-generated Sources line(s). App owns sources display."""
    clean_lines: List[str] = []
    for line in text.splitlines():
        if line.strip().lower().startswith("sources:"):
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines).strip()


def run_one(client: Anthropic, system_prompt: str, user_text: str) -> Tuple[bool, int, List[str], str]:
    """
    Returns: (tool_used, tool_call_count, sources, assistant_text)

    Tool protocol:
    - If assistant returns tool_use blocks, the next user message MUST contain matching tool_result blocks.
    - We therefore execute ALL tool calls returned in a single assistant message, then send tool_results together.
    """
    messages: List[Dict[str, Any]] = [{"role": "user", "content": user_text}]

    tool_used = False
    tool_call_count = 0
    sources: List[str] = []

    for _round in range(3):  # max 3 tool rounds
        resp = client.messages.create(
            model=MODEL,
            max_tokens=800,
            system=system_prompt,
            tools=TOOLS,
            messages=messages,
        )

        # Collect tool calls first (do NOT return early just because there's some text)
        tool_calls = [
            b for b in resp.content
            if getattr(b, "type", None) == "tool_use" and b.name == "search_wikipedia"
        ]

        if tool_calls:
            tool_used = True
            messages.append({"role": "assistant", "content": resp.content})

            tool_results_blocks: List[Dict[str, Any]] = []

            for block in tool_calls:
                tool_call_count += 1

                tool_input = block.input or {}
                query = tool_input.get("query")
                limit = tool_input.get("limit", 5)

                tool_result = search_wikipedia(query=query, limit=limit)

                # Update sources from this tool_result (keep top 2 titles overall)
                new_sources = [r.get("title") for r in tool_result.get("results", []) if r.get("title")]
                if new_sources:
                    merged = sources + new_sources
                    # de-dupe while preserving order
                    deduped = list(dict.fromkeys([s for s in merged if s]))
                    sources = deduped[:2]

                tool_results_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(tool_result),
                    }
                )

            # Must be the NEXT message after the tool_use message
            messages.append({"role": "user", "content": tool_results_blocks})
            continue

        # No tool calls, so we expect final text
        assistant_text = strip_model_sources(extract_text(resp))
        if assistant_text:
            return tool_used, tool_call_count, sources, assistant_text

        # No tool calls and no text
        return tool_used, tool_call_count, sources, "I couldn't produce a response based on the available information."

    return tool_used, tool_call_count, sources, "I couldn't complete the request within the tool-use limit."

def check_case(case: Dict[str, Any], tool_used: bool, tool_call_count: int, assistant_text: str) -> List[str]:
    failures: List[str] = []
    exp = case.get("expect", {})

    # Check should_search
    should_search = exp.get("should_search")
    if should_search is True and not tool_used:
        failures.append("Expected tool use (should_search=true), but tool was not used.")
    if should_search is False and tool_used:
        failures.append("Expected no tool use (should_search=false), but tool was used.")

    # Check max_tool_calls
    max_tool_calls = exp.get("max_tool_calls")
    if max_tool_calls is not None and tool_call_count > max_tool_calls:
        failures.append(f"Expected at most {max_tool_calls} tool call(s), but {tool_call_count} were made.")

    # Check response_style
    style = exp.get("response_style")
    if style == "clarify":
        # Should be exactly one question, ending with '?'
        if not assistant_text.endswith("?"):
            failures.append("Expected a clarifying question ending with '?', but response did not end with '?'.")
        qmarks = assistant_text.count("?")
        if qmarks != 1:
            failures.append(f"Expected exactly one question mark for a single clarifying question, found {qmarks}.")

    # Check must_contain_any
    must_any = exp.get("must_contain_any", [])
    if must_any and not any(s.lower() in assistant_text.lower() for s in must_any):
        failures.append(f"Expected response to contain one of {must_any}, but none were found.")

    # Check must_also_contain_any (additional requirement on top of must_contain_any)
    must_also_any = exp.get("must_also_contain_any", [])
    if must_also_any and not any(s.lower() in assistant_text.lower() for s in must_also_any):
        failures.append(f"Expected response to also contain one of {must_also_any}, but none were found.")

    # Check must_not_contain
    must_not = exp.get("must_not_contain", [])
    for phrase in must_not:
        if phrase.lower() in assistant_text.lower():
            failures.append(f"Expected response NOT to contain '{phrase}', but it was found.")

    # Check min_length
    min_length = exp.get("min_length")
    if min_length is not None and len(assistant_text) < min_length:
        failures.append(f"Expected response to be at least {min_length} characters, but got {len(assistant_text)}.")

    # Check must_end_with_question_mark (for clarify style)
    if exp.get("must_end_with_question_mark") and not assistant_text.endswith("?"):
        failures.append("Expected response to end with '?', but it did not.")

    return failures


def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY. Set it in .env")

    client = Anthropic(api_key=api_key)
    system_prompt = load_system_prompt()

    with open(EVAL_CASES_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    results: List[RunResult] = []

    for case in cases:
        cid = case["id"]
        user_text = case["input"]

        tool_used, tool_call_count, sources, assistant_text = run_one(client, system_prompt, user_text)
        failures = check_case(case, tool_used, tool_call_count, assistant_text)
        passed = len(failures) == 0

        results.append(
            RunResult(
                id=cid,
                passed=passed,
                tool_used=tool_used,
                tool_call_count=tool_call_count,
                sources=sources,
                assistant_text=assistant_text,
                failures=failures,
            )
        )

        status = "PASS" if passed else "FAIL"
        print(f"\n[{status}] {cid}")
        print(f"Q: {user_text}")
        print(f"Tool used: {tool_used} ({tool_call_count} call(s))")
        if tool_used:
            print(f"App Sources: {', '.join(sources) if sources else '(none)'}")
        print("Answer:")
        print(assistant_text)
        if failures:
            print("Failures:")
            for fmsg in failures:
                print(f" - {fmsg}")

    summary = {
        "model": MODEL,
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "total": len(results),
        "results": [
            {
                "id": r.id,
                "passed": r.passed,
                "tool_used": r.tool_used,
                "tool_call_count": r.tool_call_count,
                "sources": r.sources,
                "failures": r.failures,
                "assistant_text": r.assistant_text,
            }
            for r in results
        ],
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Wrote {OUT_PATH}")
    print("Summary:", summary["passed"], "passed,", summary["failed"], "failed,", summary["total"], "total")


if __name__ == "__main__":
    main()
