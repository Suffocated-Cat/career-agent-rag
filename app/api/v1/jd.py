from fastapi import APIRouter

from app.api import deps
from app.models.jd import JdParseRequest, JdParseResponse
from app.services.jd_parser import parse_jd as parse_jd_text

router = APIRouter()


@router.post("/jd/parse", response_model=JdParseResponse)
async def parse_jd(request: JdParseRequest):
    """Parse a job description text.

    Extracts job title, company, required skills, responsibilities, and
    nice-to-have qualifications. Uses LLM extraction when an LLM is configured,
    falling back to rule-based parsing otherwise.
    """
    jd = parse_jd_text(request.raw_text, llm=deps.get_llm())
    return JdParseResponse(data=jd)
