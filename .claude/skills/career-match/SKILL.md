---
name: career-match
description: Analyze a job description against a resume — extract skills, score the fit, rank the most relevant experience, audit for unsupported/exaggerated claims, and produce a match report. Use when the user provides (or points to) a JD and a resume and wants a fit analysis, skill-gap breakdown, authenticity check, or interview-prep report.
---

# Career Match

Runs the full CareerAgent pipeline over a job description and a resume:

```
parse JD + resume → match (skills + semantic) → rank projects by relevance
                  → audit for authenticity risks → generate a report
```

Scores, rankings, and risk findings are computed deterministically; an LLM (if
configured) adds field extraction, risk advice, and a natural-language report,
always with a deterministic fallback.

## How to run

From the project root, with the two texts saved to files:

```bash
python -m app.skills.career_match --jd JD.txt --resume RESUME.txt
```

Add `--no-llm` to force the deterministic (rule + retrieval) path.

For programmatic use, call `run_career_match(jd_text, resume_text, ...)` from
`app.skills.career_match`; it returns a `CareerMatchResult` with the parsed
`jd` / `resume`, the `match` result (including `project_relevance` and
`project_audit`), and the `report`.

## Output

- A Markdown match report (overall score/rating, skill analysis, gaps,
  most-relevant experience, project risk audit, recommendations).
- The risk audit summary, when findings exist.

## Relationship to MCP

The same capabilities are exposed as MCP tools via `python -m app.mcp.server`
(`parse_jd`, `parse_resume`, `match_resume`, `audit_resume`, `rank_projects`),
for use from MCP-aware hosts. This skill is the in-process, end-to-end version
that chains those steps into one report.
