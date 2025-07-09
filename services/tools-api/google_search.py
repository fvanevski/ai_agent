import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

def search(query: str, **kwargs):
    """
    Performs a web search using Google's Programmable Search Engine.
    """
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        return "Error: Google API key or CSE ID is not set."

    url = "https://www.googleapis.com/customsearch/v1"

    # Define valid API parameters to filter kwargs
    valid_params = {
        "c2coff", "cr", "dateRestrict", "exactTerms", "excludeTerms",
        "fileType", "filter", "gl", "highRange", "hl", "hq", "imgColorType",
        "imgDominantColor", "imgSize", "imgType", "linkSite", "lowRange",
        "lr", "num", "orTerms", "rights", "safe", "searchType",
        "siteSearch", "siteSearchFilter", "sort", "start"
    }

    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
    }

    # Add valid optional parameters from kwargs
    for key, value in kwargs.items():
        if key in valid_params and value is not None:
            params[key] = value

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
