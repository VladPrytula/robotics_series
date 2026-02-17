# Manning Book Directory

This directory contains the Manning book draft. It is a SEPARATE product
from the tutorials in `tutorials/`. Do not cross-contaminate.

## Directory Structure

```
Manning/
  chapters/       -- Book chapter drafts (chNN_<topic>.md)
  scaffolds/      -- Phase 1 scaffold files (chNN_scaffold.md)
  reviews/        -- Phase 3 review files (chNN_review.md)
  output/         -- Built PDF and DOCX files (generated, do not edit)
  reference.docx  -- DOCX style template (optional, customizable)
```

## How to Produce Book Content

Five specialized agents produce book chapters. Their prompts live in
`manning_proposal/agents/`. Always follow the chapter production protocol
in `manning_proposal/chapter_production_protocol.md`.

**Agents:**

| Agent | Prompt | What it does |
|-------|--------|-------------|
| Scaffolder | `agents/scaffolder.md` | Reads tutorial + protocol -> produces scaffold |
| Lab Engineer | `agents/lab_engineer.md` | Reads scaffold + CLAUDE.md -> produces lab code |
| Book Writer | `agents/writer.md` | Reads scaffold + persona -> produces chapter prose |
| Reviewer | `agents/reviewer.md` | Reads chapter + scaffold + persona -> produces review |
| Publisher | `agents/publisher.md` | Validates + builds chapter -> PDF and DOCX |

**Workflow:**

```
1. Scaffolder agent  -->  Manning/scaffolds/chNN_scaffold.md
          |
    [User reviews scaffold]
          |
2. Lab Engineer agent  -->  scripts/labs/*.py (--verify, --demo, --bridge)
   Book Writer agent   -->  Manning/chapters/chNN_<topic>.md
          |                   (can run in parallel if lab regions exist)
          |
3. Reviewer agent  -->  Manning/reviews/chNN_review.md
          |
    [Address review findings, re-run Writer if needed]
          |
4. Verify: run commands, check artifacts, final read-through
          |
5. Publisher agent  -->  Manning/output/chNN_<topic>.pdf
                         Manning/output/chNN_<topic>.docx
                         Manning/output/build_report.json
```

**Build commands (used by Publisher agent):**

```bash
# Build all chapters as PDF + DOCX
python scripts/build_book.py

# Build specific chapter
python scripts/build_book.py --chapters 1

# Validate without building
python scripts/build_book.py --validate-only --verbose

# Build combined manuscript
python scripts/build_book.py --combined
```

**Key rules:**
- Never modify `tutorials/` when writing book chapters
- The scaffold is the handoff artifact between agents
- Lab code (scripts/labs/) must have regions ready before Writer finalizes listings
- Book voice follows `manning_proposal/manning_writer_persona.md`, NOT the
  tutorial voice in root `CLAUDE.md`
- Always review before manual verification (cheap checks before expensive ones)
