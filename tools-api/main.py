from fastapi import FastAPI
from pydantic import BaseModel
from yt_summarize import run_critical_summary

app = FastAPI(title="LangChain Tools API")

class SummarizeRequest(BaseModel):
    source: str

@app.post("/yt_summarize")
def summarize(req: SummarizeRequest):
    result = run_critical_summary(req.source)
    return {"summary": result}
