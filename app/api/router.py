from fastapi import APIRouter

from app.api.v1 import jd, resume, match

api_router = APIRouter()
api_router.include_router(jd.router, tags=["JD"])
api_router.include_router(resume.router, tags=["Resume"])
api_router.include_router(match.router, tags=["Match"])
