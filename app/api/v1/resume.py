from fastapi import APIRouter

from app.models.resume import ResumeParseRequest, ResumeParseResponse, Resume

router = APIRouter()


@router.post("/resume/parse", response_model=ResumeParseResponse)
async def parse_resume(request: ResumeParseRequest):
    """Parse a resume text.

    (Placeholder — full parsing logic arrives on Day 3.)
    """
    resume = Resume(raw_text=request.raw_text)
    return ResumeParseResponse(data=resume)
