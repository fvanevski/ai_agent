import requests
import os
from dotenv import load_dotenv

load_dotenv()

def search_file_content(pattern: str) -> str:
    """Searches for a regular expression pattern within the content of files."""
    try:
        url = f"{os.getenv('FASTAPI_URL')}/search_file_content?pattern={pattern}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"
