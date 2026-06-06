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

## Required inputs

- Job description: text or a file path.
- Resume: text or a file path.

Both are required. If only one is available, ask for the other before running —
do not run the skill with a single input.

## When not to use

Do not use this skill when the user only wants to:

- parse a JD (use `POST /api/v1/jd/parse` or the `parse_jd` MCP tool),
- parse a resume (use `POST /api/v1/resume/parse` or `parse_resume`),
- audit a resume on its own (use `POST /api/v1/audit` or `audit_resume`), or
- call any single MCP tool.

Reach for those individual endpoints/tools instead. This skill is for the full
JD-vs-resume analysis end to end.

## Guardrails

Do not let the LLM modify scores, rankings, or risk findings — these come from
the deterministic services (matching, retrieval ranking, rule-based audit). The
LLM may only extract fields, add risk advice, or write the final narrative
report, and every LLM step has a deterministic fallback. This separation is what
keeps results reproducible and the evaluation harness meaningful.

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
