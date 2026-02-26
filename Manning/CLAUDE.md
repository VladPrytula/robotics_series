# Manning Book Directory

This directory contains the Manning book draft. It is a SEPARATE product
from the tutorials in `tutorials/`. Do not cross-contaminate.

## Directory Structure

```
Manning/
  chapters/       -- Book chapter drafts (chNN_<topic>.md)
  scaffolds/      -- Phase 1 scaffold files (chNN_scaffold.md)
  reviews/        -- Phase 3 review files (chNN_review.md)
  revisions/      -- Phase 3.5 revision logs (chNN_revision_NNN.md)
  output/         -- Built PDF and DOCX files (generated, do not edit)
  reference.docx  -- DOCX style template (optional, customizable)
```

## How to Produce Book Content

Six specialized agents produce book chapters. Their prompts live in
`manning_proposal/agents/`. Always follow the chapter production protocol
in `manning_proposal/chapter_production_protocol.md`.

**Agents:**

| Agent | Prompt | What it does |
|-------|--------|-------------|
| Scaffolder | `agents/scaffolder.md` | Reads tutorial + protocol -> produces scaffold |
| Lab Engineer | `agents/lab_engineer.md` | Reads scaffold + CLAUDE.md -> produces lab code |
| Book Writer | `agents/writer.md` | Reads scaffold + persona -> produces chapter prose |
| Reviewer | `agents/reviewer.md` | Reads chapter + scaffold + persona -> produces review |
| Revisor | `agents/revisor.md` | Reads chapter + review + scope -> targeted chapter updates |
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
          |                   NOTE: if scaffold estimates > 6,000 words,
          |                   the Writer phase splits into sequential spans
          |                   (see chapter_production_protocol.md "Writer Span Protocol")
          |
3. Reviewer agent  -->  Manning/reviews/chNN_review.md
          |
    [Address review findings]
          |
3.5 (optional) Revisor agent  -->  Targeted chapter updates
                                    Manning/revisions/chNN_revision_NNN.md
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

**Pipeline design note:** Agents 1-3.5 (Scaffolder through Revisor) are pure
text-in/text-out -- they read and write markdown only. The Publisher (Phase 5)
is the only agent that runs external tools (pandoc, xelatex). This separation
means the Publisher catches build-specific issues invisible to the Reviewer:
math that renders in MkDocs but breaks in pandoc, images that resolve on the
web but not locally, code blocks that overflow page width in PDF.

**Key rules:**
- Never modify `tutorials/` when writing book chapters
- The scaffold is the handoff artifact between agents
- Lab code (scripts/labs/) must have regions ready before Writer finalizes listings
- Book voice follows `manning_proposal/manning_writer_persona.md`, NOT the
  tutorial voice in root `CLAUDE.md`
- Always review before manual verification (cheap checks before expensive ones)

## Build Dependencies

The Publisher agent and `scripts/build_book.py` require:

```bash
# Ubuntu/Debian (DGX or CI)
sudo apt-get install pandoc texlive-xetex texlive-latex-extra \
    texlive-fonts-recommended fonts-dejavu fonts-dejavu-extra

# Optional: for GIF image conversion
sudo apt-get install imagemagick
```

These are already installed in CI (`.github/workflows/pdfs.yml`).

## DOCX Reference Template

If Manning provides a Word template with their house styles, drop it at
`Manning/reference.docx`. All future DOCX builds pick it up automatically
(heading fonts, body spacing, code block styling). To generate a default
starting template:

```bash
python3 scripts/build_book.py --init-reference-doc
```

Open in Word or LibreOffice, customize styles, save in place.
