from fastapi import APIRouter

from app.models.resume import ResumeParseRequest, ResumeParseResponse
from app.services.resume_parser import parse_resume as parse_resume_text

router = APIRouter()


@router.post("/resume/parse", response_model=ResumeParseResponse)
async def parse_resume(request: ResumeParseRequest):
    """Parse a resume text.

    Extracts skills, projects, education, and work experience
    using rule-based section detection and keyword matching.
    """
    resume = parse_resume_text(request.raw_text)
    return ResumeParseResponse(data=resume)
