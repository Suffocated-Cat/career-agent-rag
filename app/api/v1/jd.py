from fastapi import APIRouter

from app.models.jd import JdParseRequest, JdParseResponse, JobDescription

router = APIRouter()


@router.post("/jd/parse", response_model=JdParseResponse)
async def parse_jd(request: JdParseRequest):
    """Parse a job description text.

    (Placeholder — full parsing logic arrives on Day 2.)
    """
    jd = JobDescription(raw_text=request.raw_text)
    return JdParseResponse(data=jd)
