import json
import os
from dotenv import load_dotenv
from anthropic import Anthropic

from wikipedia import search_wikipedia

load_dotenv()

def load_system_prompt() -> str:
    with open("prompt/system_prompt.txt", "r", encoding="utf-8") as f:
        return f.read()

def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY. Set it in .env")

    client = Anthropic(api_key=api_key)
    system_prompt = load_system_prompt()

    tools = [
        {
            "name": "search_wikipedia",
            "description": "Search Wikipedia for relevant pages and return top results with titles and snippets.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The Wikipedia search query."},
                    "limit": {"type": "integer", "description": "Max number of results to return.", "default": 5},
                },
                "required": ["query"],
            },
        }
    ]

    print("Wikipedia QA (Claude). Type a question, or 'exit' to quit.")
    while True:
        user_text = input("\nYou: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        messages = [{"role": "user", "content": user_text}]

        resp = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=800,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )

        tool_used = False
        sources = []

        for block in resp.content:
            if block.type == "tool_use" and block.name == "search_wikipedia":
                tool_used = True

                tool_input = block.input
                query = tool_input.get("query")
                limit = tool_input.get("limit", 5)

                tool_result = search_wikipedia(query=query, limit=limit)

                # Capture sources (page titles) for printing later
                sources = [r.get("title") for r in tool_result.get("results", []) if r.get("title")]
                sources = sources[:2]  # keep it tight for reviewers

                messages.append({"role": "assistant", "content": resp.content})
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(tool_result),
                            }
                        ],
                    }
                )

                resp = client.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=800,
                    system=system_prompt,
                    tools=tools,
                    messages=messages,
                )
                break

        print("\nAssistant:")
        for block in resp.content:
            if block.type == "text":
                print(block.text)

        if tool_used and sources:
            print("\nSources: " + ", ".join(sources))

if __name__ == "__main__":
    main()
