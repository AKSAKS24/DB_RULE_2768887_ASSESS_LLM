# app_assess_selects_draftcheck.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any
import os, json

# ---- Env setup ----
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required.")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

# LangChain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

app = FastAPI(title="SAP Note 2768887 Aware SELECT Assessment")

# ===== Models =====
class SelectItem(BaseModel):
    table: str
    target_type: str
    target_name: str
    used_fields: List[str]
    suggested_fields: List[str]
    suggested_statement: str

    @field_validator("used_fields", "suggested_fields")
    @classmethod
    def no_none_elems(cls, v: List[str]) -> List[str]:
        return [x for x in v if x is not None]

class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: str
    selects: List[SelectItem] = Field(default_factory=list)

# ===== Summariser that flags SAP Note 2768887 risks =====
def summarize_selects(unit: Unit) -> Dict[str, Any]:
    """
    Extended summarisation — detects SELECT on VBRK/VBRP without DRAFT filter.
    """
    tables_count: Dict[str, int] = {}
    total = len(unit.selects)
    flagged = []
    for s in unit.selects:
        tables_count[s.table] = tables_count.get(s.table, 0) + 1
        if s.table.upper() in ("VBRK", "VBRP"):
            draft_fields = [f.upper() for f in s.used_fields + s.suggested_fields]
            stmt = (s.suggested_statement or "").upper()
            if not any("DRAFT" in f for f in draft_fields) and "DRAFT" not in stmt:
                flagged.append({
                    "table": s.table,
                    "target": s.target_name,
                    "reason": "No filtering on DRAFT = SPACE detected for table " + s.table
                })

    return {
        "program": unit.pgm_name,
        "include": unit.inc_name,
        "unit_type": unit.type,
        "unit_name": unit.name,
        "stats": {
            "total_selects": total,
            "tables_frequency": tables_count,
            "note_2768887_flags": flagged
        }
    }

# ===== Prompt for SAP Note–specific fix =====
SYSTEM_MSG = "You are a precise ABAP reviewer familiar with SAP Note 2768887. Output strict JSON only."

USER_TEMPLATE = """
You are assessing ABAP SELECT usage in light of SAP Note 2768887 (SD Billing Document Drafts).

From S/4HANA 1709 onwards, tables VBRK/VBRP have a DRAFT boolean field.
- Draft document versions must not be processed in normal logic.
- Custom code should filter: `VBRK-DRAFT = SPACE` or `VBRP-DRAFT = SPACE`.

We provide program/include/unit metadata, and SELECT analysis.
Your tasks:
1) Produce a concise **assessment** highlighting:
   - If SELECTs on VBRK/VBRP lack DRAFT filtering.
   - Potential impact (erroneous reports, extra records, logic errors).
   - Performance / functional considerations.
2) Produce an **LLM remediation prompt** to:
   - Scan ABAP code in this unit for SELECTs on VBRK/VBRP without DRAFT filter.
   - Add `DRAFT = SPACE` in WHERE clauses where missing.
   - Preserve functional logic, ECC/S4-safe syntax.
   - Output strictly in JSON with: original_code, remediated_code, changes[].

Return ONLY strict JSON:
{{
  "assessment": "<concise note 2768887 impact>",
  "llm_prompt": "<prompt for LLM code fixer>"
}}

Unit metadata:
- Program: {pgm_name}
- Include: {inc_name}
- Unit type: {unit_type}
- Unit name: {unit_name}

Analysis:
{plan_json}

selects (JSON):
{selects_json}
""".strip()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MSG),
        ("user", USER_TEMPLATE),
    ]
)

llm = ChatOpenAI(model=OPENAI_MODEL)
parser = JsonOutputParser()
chain = prompt | llm | parser

# ===== LLM Call =====
def llm_assess_and_prompt(unit: Unit) -> Dict[str, str]:
    plan = summarize_selects(unit)
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
    selects_json = json.dumps([s.model_dump() for s in unit.selects], ensure_ascii=False, indent=2)

    try:
        return chain.invoke({
            "pgm_name": unit.pgm_name,
            "inc_name": unit.inc_name,
            "unit_type": unit.type,
            "unit_name": unit.name,
            "plan_json": plan_json,
            "selects_json": selects_json
        })
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

# ===== API =====
@app.post("/assess-selects")
def assess_selects(units: List[Unit]) -> List[Dict[str, Any]]:
    out = []
    for u in units:
        obj = u.model_dump()
        llm_out = llm_assess_and_prompt(u)
        obj["assessment"] = llm_out.get("assessment", "")
        obj["llm_prompt"] = llm_out.get("llm_prompt", "")
        obj.pop("selects", None)
        out.append(obj)
    return out

@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}