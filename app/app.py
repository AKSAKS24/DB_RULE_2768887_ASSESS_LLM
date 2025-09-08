from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os, json

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required.")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

app = FastAPI(title="SAP Note 2768887 SELECT-DRAFT Assessment (system message style)")

class Finding(BaseModel):
    pgm_name: Optional[str] = None
    inc_name: Optional[str] = None
    type: Optional[str] = None
    name: Optional[str] = None
    class_implementation: Optional[str] = None
    issue_type: Optional[str] = None
    severity: Optional[str] = None
    message: Optional[str] = None
    suggestion: Optional[str] = None
    snippet: Optional[str] = None

class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: str
    findings: List[Finding] = Field(default_factory=list)

SYSTEM_MSG = """
You are a senior ABAP expert. Output STRICT JSON.

If findings is not empty, provide:
- assessment: paragraph about risk of missing DRAFT=SPACE in SELECTs.
- llm_prompt: for every finding, a bullet (snippet and suggestion).

If findings is empty, return nothing.

FORMAT (do NOT explain):
{{{{"assessment": "...", "llm_prompt": "..."}}}}
""".strip()

USER_TEMPLATE = """
Program: {pgm_name}
Include: {inc_name}
Type: {unit_type}
Name: {unit_name}
Findings JSON:
{findings_json}

Instructions:
If findings_json is not empty: summarize risk and return a bullet for each finding (snippet and suggestion).
If findings_json is empty: return nothing.

FORMAT:
{{{{"assessment": "...", "llm_prompt": "..."}}}}
""".strip()

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MSG),
    ("user", USER_TEMPLATE),
])

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.0)
parser = JsonOutputParser()
chain = prompt | llm | parser

def llm_assess_and_prompt(unit: Unit) -> Dict[str, str]:
    findings_json = json.dumps([f.model_dump() for f in unit.findings], ensure_ascii=False, indent=2)
    print("FINDINGS:", findings_json)  # for debugging, can remove later
    try:
        return chain.invoke({
            "pgm_name": unit.pgm_name,
            "inc_name": unit.inc_name,
            "unit_type": unit.type,
            "unit_name": unit.name,
            "findings_json": findings_json
        })
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

@app.post("/assess-selects")
async def assess_selects(units: List[Unit]) -> List[Dict[str, Any]]:
    out = []
    for u in units:
        if not u.findings:
            continue  # Only return units with findings!
        obj = {
            "pgm_name": u.pgm_name,
            "inc_name": u.inc_name,
            "type": u.type,
            "name": u.name
        }
        llm_out = llm_assess_and_prompt(u)
        obj["assessment"] = llm_out.get("assessment", "")
        obj["llm_prompt"] = llm_out.get("llm_prompt", "")
        out.append(obj)
    return out

@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}