def main(query: str) -> str:
    """
    Search for a topic and return short text snippets.

    :param query: str: What to search for.
    :returns: str: Text snippets about the topic.
    """
    import json
    import urllib.parse
    import urllib.request

    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 3}
    )
    req = urllib.request.Request(url, headers={"User-Agent": "ace-local-test/0.1 (personal project)"})
    with urllib.request.urlopen(req, timeout=10) as r:
        hits = json.load(r)["query"]["search"]
    return {"result": " | ".join(h["title"] + ": " + h["snippet"] for h in hits)}