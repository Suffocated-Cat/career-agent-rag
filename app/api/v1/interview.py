from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.api import deps
from app.services.interview_prep import InterviewPrep, generate_interview_prep
from app.services.jd_parser import parse_jd
from app.services.resume_parser import parse_resume

router = APIRouter()


class InterviewPrepRequest(BaseModel):
    """Raw JD + resume text for interview preparation."""

    jd_text: str = Field(..., min_length=1, description="Raw job description text")
    resume_text: str = Field(..., min_length=1, description="Raw resume text")


class InterviewPrepResponse(BaseModel):
    """Response for the interview-prep endpoint."""

    status: str = "success"
    data: InterviewPrep


@router.post("/interview-prep", response_model=InterviewPrepResponse)
async def interview_prep_endpoint(request: InterviewPrepRequest):
    """Generate RAG-grounded interview prep from raw JD + resume text.

    Parses both, retrieves relevant questions from the knowledge base for the
    JD's skills, and generates a prep guide grounded on them (highlighting the
    candidate's skill gaps).
    """
    embedding_service = deps.get_embedding_service()
    llm = deps.get_llm()
    jd = parse_jd(request.jd_text, embedding_service=embedding_service, llm=llm)
    resume = parse_resume(request.resume_text, embedding_service=embedding_service, llm=llm)
    prep = generate_interview_prep(jd, resume, deps.get_kb_retriever(), llm=llm)
    return InterviewPrepResponse(data=prep)
