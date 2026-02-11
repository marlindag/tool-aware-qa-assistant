import os
import sys

from dotenv import load_dotenv
from anthropic import Anthropic

# Reuse the exact same logic as the eval runner
from evals.run_evals import load_system_prompt, run_one


def main() -> int:
    load_dotenv()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Missing ANTHROPIC_API_KEY. Set it in .env or your environment.")
        return 1

    # Accept question from argv, otherwise prompt
    if len(sys.argv) >= 2:
        question = " ".join(sys.argv[1:]).strip()
    else:
        question = input("Question: ").strip()

    if not question:
        print("No question provided.")
        return 1

    client = Anthropic(api_key=api_key)
    system_prompt = load_system_prompt()

    tool_used, tool_call_count, sources, assistant_text = run_one(client, system_prompt, question)

    print(f"Q: {question}")
    print(f"Tool used: {tool_used} ({tool_call_count} call(s))")
    if tool_used:
        print(f"App Sources: {', '.join(sources) if sources else '(none)'}")
    print("Answer:")
    print(assistant_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

