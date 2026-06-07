from fastapi import APIRouter

from app.api.v1 import jd, resume, match, audit, career

api_router = APIRouter()
api_router.include_router(jd.router, tags=["JD"])
api_router.include_router(resume.router, tags=["Resume"])
api_router.include_router(match.router, tags=["Match"])
api_router.include_router(audit.router, tags=["Audit"])
api_router.include_router(career.router, tags=["Career"])
