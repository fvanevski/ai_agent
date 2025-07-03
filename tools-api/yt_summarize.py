"""run_agent.py – LangChain orchestration client with HF‑style chat roles
=====================================================================
This client:
1. Sends a URL or local file to Whisper via the GPU‑aware proxy (port 8000).
2. Receives diarised, speaker‑labelled **plain‑text** (`output=text`).
3. Feeds that transcript to an LLM served behind the same proxy **using the
   HuggingFace llama‑3 chat template contraints** (conversation must begin with
   a `user` role; only user/assistant roles; roles must alternate).

It logs major milestones and automatically retries once on chunked‑encoding
errors.
"""
from __future__ import annotations

import json, logging, mimetypes, os, pathlib, re, sys
from typing import Literal, Dict, Any

import requests
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import time
import textwrap # argparse and textwrap are used in __main__ but not imported at top level

# ---------------------------------------------------------------------------
# Logging setup – INFO by default; DEBUG for request/response bodies
# ---------------------------------------------------------------------------
log_level = os.getenv("RUN_AGENT_LOG", "INFO").upper()
logging.basicConfig(
    stream=sys.stderr,
    level=getattr(logging, log_level, logging.INFO),
    format="%(levelname).1s %(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# REST helpers – auto‑route to the correct Whisper endpoint via the proxy
# ---------------------------------------------------------------------------
TRANSCRIBE_ENDPOINT     = f"http://localhost:8003/transcribe"
TRANSCRIBE_URL_ENDPOINT = f"http://localhost:8003/transcribe_url"

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)

def _is_url(src: str) -> bool:
    return bool(_URL_RE.match(src))


def _call_whisper(source: str, *, output: str = "text", timeout: int = 900) -> Dict[str, Any]:
    """POST helper with logging + single retry for occasional chunked errors."""

    def _post(url: str, **kwargs):
        headers = kwargs.pop("headers", {})
        headers.setdefault("Accept-Encoding", "identity")  # avoid gzip truncation
        LOGGER.debug("POST %s output=%s", url, output)
        resp = requests.post(url, headers=headers, timeout=timeout, **kwargs)
        LOGGER.debug("<- %s %s", resp.status_code, url)
        return resp

    def _do_request() -> requests.Response:
        if _is_url(source):
            payload = {"url": source, "output": output}
            return _post(TRANSCRIBE_URL_ENDPOINT, json=payload)
        # local path
        path = pathlib.Path(source)
        if not path.exists():
            raise FileNotFoundError(path)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        with path.open("rb") as fh:
            files = {"file": (path.name, fh, mime)}
            data = {"output": output}
            return _post(TRANSCRIBE_ENDPOINT, files=files, data=data)

    max_retries = 3
    retry_delay_seconds = 5

    for attempt in range(max_retries):
        try:
            resp = _do_request()

            # If model is waking up, proxy returns 202 Accepted.
            if resp.status_code == 202:
                LOGGER.warning(
                    "Whisper model is waking up (attempt %d/%d). Retrying in %d s...",
                    attempt + 1, max_retries, retry_delay_seconds
                )
                time.sleep(retry_delay_seconds)
                continue  # Move to the next attempt

            resp.raise_for_status()  # Raise for other HTTP errors (4xx, 5xx)
            return resp.json()

        except requests.exceptions.ChunkedEncodingError as e:
            LOGGER.warning("ChunkedEncodingError (attempt %d/%d), retrying: %s", attempt + 1, max_retries, e)
            if attempt + 1 >= max_retries:
                raise  # Re-raise the last exception if all retries fail
            time.sleep(retry_delay_seconds)
        except requests.HTTPError as e:
            LOGGER.error("Whisper service HTTPError %s: %s", e.response.status_code, e)
            raise

    raise ConnectionError(f"Failed to get a valid response from Whisper after {max_retries} retries.")



# ---------------------------------------------------------------------------
# LangChain tool – exposes transcription to the LLM chain
# ---------------------------------------------------------------------------
@tool("transcribe_audio", return_direct=True)
def transcribe_audio_tool(
    source: str,
    output: Literal["segments", "words", "sentences", "text"] = "text",
) -> str:  # noqa: D401
    """Transcribe a YouTube URL or local file and return the chosen output view."""
    LOGGER.info("Transcribing %s (view=%s)", source, output)
    whisper_json = _call_whisper(source, output=output)
    LOGGER.info("Transcription complete – %s chars", len(whisper_json.get("text", "")))
    return whisper_json[output] if output != "text" else whisper_json["text"]

# ---------------------------------------------------------------------------
# Investigative‑journalism prompt – HuggingFace roles: user then assistant
# ---------------------------------------------------------------------------
SYSTEM_MSG = (
    "You are an experienced investigative journalist and non‑partisan fact‑checker. "
    "Your mission is to deconstruct the conversation, probe for bias, and surface what an informed, skeptical viewer still needs to know. "
    "Write with clarity, concision, explicit sourcing, and the Markdown structure provided below."
)

STRUCTURED_RUBRIC = """
### 1. Speaker Roster
| Label | Probable Identity | Transcript Evidence (≤15 w) | Public Bio Snippet (+cite) | Certainty (H/M/L) | Role & Likely Bias |
|-------|------------------|-----------------------------|---------------------------|-------------------|--------------------|
| …     | …                | …                           | …                         | …                 | …                  |

*Rules*
* If uncertain, write “Unknown”.
* Bio snippet must cite a public source: e.g. “NYT profile 2023-05-01”.
* Bias = partisan ties, professional incentives, or ideology (≤10 w).

### 2. Core Topics & Positions
List up to **6 topics**.  For each:
* **Topic (≤8 w)** – first timestamp.
    * **SpeakerName:** one-sentence stance (≤25 w).

### 3. Critical Analysis
#### 3.A Intent & Framing (per speaker)
* Speaker – Objective · How background/incentives shape framing (2 lines each).

#### 3.B Key Claims & Fact Check Table
| # | Speaker | Claim (≤20 w quote) | Type (Fact/Opinion/Forecast) | Source & Link/Doc | Verdict (True/Mixed/Unverified/False) | Credibility Tier* |
|---|---------|---------------------|------------------------------|-------------------|---------------------------------------|------------------|

*Credibility tiers: **A** = Documented; **B** = Likely (multi-source); **C** = Unclear; **D** = Dubious; **E** = Refuted.

#### 3.C Persuasion & Rhetoric
| Time | Quote (≤15 w) | Technique | Intended Effect |
|------|--------------|-----------|-----------------|

List at least three concrete examples; flag logical fallacies.

#### 3.D Missing Context & Follow-ups
Bullet unresolved questions, data gaps, or contradictions a skeptic should probe.

### 4. 60-Second Executive Brief
* **Why this video matters** (1 sentence)
* **Most substantiated fact** (1 bullet)
* **Biggest unanswered question** (1 bullet)
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MSG),
    ("user", "Here is the transcript (plain text):\n{transcript}\n\n" + STRUCTURED_RUBRIC),
])

LLM = ChatOpenAI(
    model="chat",                     # served by vLLM
    openai_api_base=f"http://localhost:8002/v1",
    openai_api_key="not-needed",
    temperature=0.2,
)

OUTPUT_PARSER = StrOutputParser()
LLM_CHAIN = PROMPT | LLM | OUTPUT_PARSER

# ---------------------------------------------------------------------------
# Convenience: end‑to‑end helper
# ---------------------------------------------------------------------------

def run_critical_summary(source: str) -> str:
    """Transcribe → feed to LLM → return markdown summary."""
    LOGGER.info("Starting critical summary for %s", source)
    # Wake up the LLM agent before starting transcription
    try:
        wakeup_url = f"http://localhost:8002/wake_up"
        resp = requests.post(wakeup_url, timeout=10)
        if resp.ok:
            LOGGER.info("Sent wake_up to vllm-agent: %s", resp.json())
        else:
            LOGGER.warning("Wake_up call to vllm-agent failed: %s %s", resp.status_code, resp.text)
    except Exception as e:
        LOGGER.warning("Exception during wake_up call to vllm-agent: %s", e)

    transcript_text = transcribe_audio_tool.run({"source": source})
    LOGGER.debug("Transcript length: %d chars", len(transcript_text))

    # Add a retry loop to handle the proxy's "waking up" response.
    max_retries = 3
    retry_delay_seconds = 5

    for attempt in range(max_retries):
        try:
            summary = LLM_CHAIN.invoke({"transcript": transcript_text})
            LOGGER.info("LLM summary complete (%d chars)", len(summary))
            return summary
        except TypeError:
            LOGGER.warning(
                "LLM call failed on attempt %d of %d. This can happen if the model "
                "was sleeping. Retrying in %d seconds...",
                attempt + 1,
                max_retries,
                retry_delay_seconds
            )
            if attempt + 1 >= max_retries:
                LOGGER.error("LLM call failed after all retries.")
                raise
            time.sleep(retry_delay_seconds)
    # This line is unreachable but added for linters/type-checkers
    return ""

# ---------------------------------------------------------------------------
# CLI entry‑point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, textwrap

    parser = argparse.ArgumentParser(description="Transcribe + critical summary")
    parser.add_argument("source", help="YouTube URL | http/https URL | local file path")
    parser.add_argument("--debug", action="store_true", help="verbose logging")
    args = parser.parse_args()

    if args.debug:
        LOGGER.setLevel(logging.DEBUG)

    md_out = run_critical_summary(args.source)
    print(textwrap.fill(md_out, width=1000))
