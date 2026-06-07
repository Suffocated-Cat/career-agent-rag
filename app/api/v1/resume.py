from fastapi import APIRouter

from app.api import deps
from app.models.resume import ResumeParseRequest, ResumeParseResponse
from app.services.resume_parser import parse_resume as parse_resume_text

router = APIRouter()


@router.post("/resume/parse", response_model=ResumeParseResponse)
async def parse_resume(request: ResumeParseRequest):
    """Parse a resume text.

    Extracts skills, projects, education, and work experience. Uses LLM
    extraction when an LLM is configured, falling back to rule-based section
    detection and keyword matching otherwise.
    """
    resume = parse_resume_text(request.raw_text, llm=deps.get_llm())
    return ResumeParseResponse(data=resume)
