import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

def search(
    query: str,
    num: int = 10,
    dateRestrict: str = None,
    exactTerms: str = None,
    excludeTerms: str = None,
    fileType: str = None,
    lr: str = None,
    safe: str = "off",
    siteSearch: str = None,
    sort: str = None,
):
    """
    Performs a web search using Google's Programmable Search Engine.
    """
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        return "Error: Google API key or CSE ID is not set."

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": num,
        "dateRestrict": dateRestrict,
        "exactTerms": exactTerms,
        "excludeTerms": excludeTerms,
        "fileType": fileType,
        "lr": lr,
        "safe": safe,
        "siteSearch": siteSearch,
        "sort": sort,
    }
    # Remove None values from params
    params = {k: v for k, v in params.items() if v is not None}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json()

        if "items" not in results:
            return "No results found."

        formatted_results = []
        for item in results["items"]:
            formatted_results.append(
                {
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet"),
                }
            )
        return formatted_results
    except requests.exceptions.RequestException as e:
        return f"Error performing search: {e}"
