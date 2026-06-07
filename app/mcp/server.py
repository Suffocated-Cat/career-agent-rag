"""
CareerAgent MCP server.

Exposes the CareerAgent capabilities as MCP tools over a FastMCP server, so an
MCP client (or any MCP-aware host, e.g. Claude Desktop) can discover and call
them. The tool logic lives in ``tools.py``; this module wires those functions
into FastMCP and handles the (lazy, optional) embedding service.

Run as a stdio MCP server:

    python -m app.mcp.server
"""

from functools import lru_cache

from mcp.server.fastmcp import FastMCP

from app.mcp import tools
from app.services.embedding import EmbeddingService


@lru_cache(maxsize=1)
def _get_embeddings() -> EmbeddingService | None:
    """Create the embedding service once; fall back to None if unavailable."""
    try:
        return EmbeddingService()
    except Exception:
        return None


mcp = FastMCP("CareerAgent")


@mcp.tool()
def parse_jd(raw_text: str) -> dict:
    """Parse a job description into skills, responsibilities, and nice-to-haves."""
    return tools.parse_jd_tool(raw_text, embedding_service=_get_embeddings())


@mcp.tool()
def parse_resume(raw_text: str) -> dict:
    """Parse a resume into skills, experience, projects, and education."""
    return tools.parse_resume_tool(raw_text, embedding_service=_get_embeddings())


@mcp.tool()
def match_resume(jd: dict, resume: dict) -> dict:
    """Match a parsed job description against a parsed resume and score the fit."""
    return tools.match_tool(jd, resume, embedding_service=_get_embeddings())


@mcp.tool()
def audit_resume(resume: dict) -> dict:
    """Audit a parsed resume for unsupported or vague claims and risks."""
    return tools.audit_tool(resume)


@mcp.tool()
def rank_projects(jd: dict, resume: dict) -> list[dict]:
    """Rank a resume's experiences/projects by relevance to the job description."""
    return tools.rank_projects_tool(jd, resume, embedding_service=_get_embeddings())


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":  # pragma: no cover
    main()
