import requests

class FetchTool:
    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    def as_mcp(self) -> dict:
        return {
            "name": "search",
            "description": "Fetch a web page or search result",
            "parameters": {"type": "string", "name": "query"},
            "call": self.run,
        }

    def run(self, query: str) -> str:
        if not self.api_key:
            return "offline"
        r = requests.get("https://api.duckduckgo.com", params={"q": query, "format": "json"}, timeout=5)
        data = r.json()
        return data.get("Abstract", "")
