import requests

WIKI_API = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "takehome-wikipedia-qa/0.1 (contact: mgzentest@gmail.com)"
}

def search_wikipedia(query: str, limit: int = 5) -> dict:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "utf8": 1,
        "srlimit": limit,
        "origin": "*",
    }

    r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()

    results = []
    for item in data.get("query", {}).get("search", []):
        results.append(
            {
                "title": item.get("title"),
                "pageid": item.get("pageid"),
                "snippet": item.get("snippet"),
            }
        )

    return {"query": query, "results": results}

