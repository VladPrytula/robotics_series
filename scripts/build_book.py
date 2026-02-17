#!/usr/bin/env python3
"""Build Manning book chapters as PDF and/or DOCX via pandoc.

Renders chapters from Manning/chapters/ into Manning/output/ as PDF, DOCX,
or both. Can also produce a combined single-document manuscript.

Usage:
    # Build all chapters as PDF + DOCX
    python scripts/build_book.py

    # Build specific chapters
    python scripts/build_book.py --chapters 1 5

    # DOCX only (for Manning editorial submission)
    python scripts/build_book.py --format docx

    # Build combined manuscript (all chapters in one file)
    python scripts/build_book.py --combined

    # Validate chapters without building (check ASCII, code length, etc.)
    python scripts/build_book.py --validate-only

    # Generate a reference DOCX template for style customization
    python scripts/build_book.py --init-reference-doc
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import date, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
import urllib.request


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
CHAPTERS_DIR = REPO_ROOT / "Manning" / "chapters"
OUTPUT_DIR = REPO_ROOT / "Manning" / "output"
REFERENCE_DOC = REPO_ROOT / "Manning" / "reference.docx"

BOOK_TITLE = "Robotics Reinforcement Learning in Action"
BOOK_SUBTITLE = (
    "Build reproducible goal-conditioned manipulation agents"
)
BOOK_AUTHOR = "Vlad Prytula"

# Part structure mirroring manning_proposal/toc.md
BOOK_PARTS: list[tuple[str, list[int]]] = [
    ("Part 1 -- Start Running, Start Measuring", [1, 2]),
    ("Part 2 -- Baselines That Debug Your Pipeline", [3, 4]),
    ("Part 3 -- Sparse Goals, Real Progress", [5, 6]),
    ("Part 4 -- Engineering-Grade Robotics RL", [7, 8, 9]),
    ("Part 5 -- Pixels and the Reality Gap", [10, 11, 12, 13]),
]

# Regex patterns
_REMOTE_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\((https?://[^)\s]+)\)")
_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_CODE_BLOCK_RE = re.compile(r"^```.*?\n(.*?)^```", re.MULTILINE | re.DOTALL)

# Manning guidelines: 30-40 lines per listing; we allow a small buffer
MAX_CODE_BLOCK_LINES = 45

# Phrases the writer persona flags (warning, not error -- context matters)
BANNED_PHRASES = [
    "simply",
    "trivially",
    "obviously",
    "it is easy to see",
    "beyond the scope",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Chapter:
    number: int
    path: Path
    title: str = ""

    def __post_init__(self) -> None:
        if not self.title:
            text = self.path.read_text(encoding="utf-8")
            m = _HEADING_RE.search(text)
            self.title = m.group(2).strip() if m else self.path.stem


@dataclass
class ValidationIssue:
    chapter: int
    severity: str  # "error", "warning", "info"
    category: str
    message: str
    line: Optional[int] = None


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    chapters_checked: int = 0

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def to_dict(self) -> dict:
        return {
            "chapters_checked": self.chapters_checked,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [
                {
                    "chapter": i.chapter,
                    "severity": i.severity,
                    "category": i.category,
                    "message": i.message,
                    "line": i.line,
                }
                for i in self.issues
            ],
        }


# ---------------------------------------------------------------------------
# Chapter discovery
# ---------------------------------------------------------------------------

def discover_chapters(
    chapters_dir: Path,
    numbers: Optional[list[int]] = None,
) -> list[Chapter]:
    """Find Manning chapter files, optionally filtered by number."""
    chapters = []
    for path in sorted(chapters_dir.glob("ch*.md")):
        m = re.match(r"ch(\d+)", path.stem)
        if not m:
            continue
        num = int(m.group(1))
        if numbers and num not in numbers:
            continue
        chapters.append(Chapter(number=num, path=path))
    return chapters


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_chapter(chapter: Chapter) -> list[ValidationIssue]:
    """Run build-readiness checks on a single chapter."""
    issues: list[ValidationIssue] = []
    text = chapter.path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # 1. ASCII-only check
    for i, line in enumerate(lines, 1):
        non_ascii = _NON_ASCII_RE.findall(line)
        if non_ascii:
            chars = ", ".join(repr(c) for c in non_ascii[:5])
            issues.append(ValidationIssue(
                chapter=chapter.number, severity="warning",
                category="ascii", line=i,
                message=f"Non-ASCII characters: {chars}",
            ))

    # 2. Code block length
    for m in _CODE_BLOCK_RE.finditer(text):
        block_lines = m.group(1).splitlines()
        if len(block_lines) > MAX_CODE_BLOCK_LINES:
            line_num = text[: m.start()].count("\n") + 1
            issues.append(ValidationIssue(
                chapter=chapter.number, severity="warning",
                category="code_length", line=line_num,
                message=(
                    f"Code block has {len(block_lines)} lines "
                    f"(guideline: <={MAX_CODE_BLOCK_LINES})"
                ),
            ))

    # 3. Heading structure
    first_heading = _HEADING_RE.search(text)
    if not first_heading:
        issues.append(ValidationIssue(
            chapter=chapter.number, severity="error",
            category="structure",
            message="No heading found in chapter",
        ))
    elif first_heading.group(1) != "#":
        line_num = text[: first_heading.start()].count("\n") + 1
        issues.append(ValidationIssue(
            chapter=chapter.number, severity="warning",
            category="structure", line=line_num,
            message=(
                f"First heading is level {len(first_heading.group(1))}, "
                f"expected level 1"
            ),
        ))

    # 4. Remote images (informational)
    remote_imgs = _REMOTE_IMAGE_RE.findall(text)
    if remote_imgs:
        issues.append(ValidationIssue(
            chapter=chapter.number, severity="info",
            category="images",
            message=f"{len(remote_imgs)} remote image(s) will be downloaded during build",
        ))

    # 5. Banned phrases (context-dependent, so warning not error)
    for phrase in BANNED_PHRASES:
        for i, line in enumerate(lines, 1):
            if phrase in line.lower():
                issues.append(ValidationIssue(
                    chapter=chapter.number, severity="warning",
                    category="voice", line=i,
                    message=f'Potentially banned phrase: "{phrase}"',
                ))
                break  # one warning per phrase per chapter

    # 6. Word count (informational)
    words = len(text.split())
    if words < 4000:
        issues.append(ValidationIssue(
            chapter=chapter.number, severity="info",
            category="length",
            message=f"Chapter is {words:,} words (short; Manning target: 6,000-10,000)",
        ))
    elif words > 12000:
        issues.append(ValidationIssue(
            chapter=chapter.number, severity="warning",
            category="length",
            message=f"Chapter is {words:,} words (long; Manning target: 6,000-10,000)",
        ))
    else:
        issues.append(ValidationIssue(
            chapter=chapter.number, severity="info",
            category="length",
            message=f"Chapter is {words:,} words",
        ))

    return issues


# ---------------------------------------------------------------------------
# Image preprocessing (shared with build_pdfs.py logic)
# ---------------------------------------------------------------------------

def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, out_path.open("wb") as f:
        f.write(resp.read())


def _convert_gif_to_png(gif_path: Path, png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    for cmd_name in ("magick", "convert", "ffmpeg"):
        cmd = shutil.which(cmd_name)
        if not cmd:
            continue
        if cmd_name == "ffmpeg":
            subprocess.run(
                [cmd, "-y", "-i", str(gif_path), "-frames:v", "1", str(png_path)],
                check=True,
            )
        else:
            subprocess.run(
                [cmd, f"{gif_path}[0]", str(png_path)], check=True,
            )
        return
    raise RuntimeError("GIF conversion requires one of: magick, convert, ffmpeg")


def _local_filename(url: str) -> str:
    name = Path(urlparse(url).path).name
    return name or "image"


def rewrite_remote_images(text: str, asset_dir: Path, md_dir: Path) -> str:
    """Download remote images and rewrite markdown URLs to local paths."""
    cache: dict[str, str] = {}

    def _repl(m: re.Match[str]) -> str:
        alt, url = m.group(1), m.group(2)
        if url not in cache:
            fname = _local_filename(url)
            downloaded = asset_dir / fname
            _download(url, downloaded)
            final = downloaded
            if downloaded.suffix.lower() == ".gif":
                final = downloaded.with_suffix(".png")
                _convert_gif_to_png(downloaded, final)
            cache[url] = os.path.relpath(final, md_dir).replace(os.sep, "/")
        return f"![{alt}]({cache[url]})"

    return _REMOTE_IMAGE_RE.sub(_repl, text)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _require_cmd(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(
            f"Required command not found: {name}\n"
            f"Install with: sudo apt-get install pandoc texlive-xetex "
            f"texlive-latex-extra fonts-dejavu"
        )
    return path


def render_file(
    pandoc: str,
    source_md: Path,
    output: Path,
    resource_path: Path,
    fmt: str,
    pdf_engine: str = "xelatex",
    reference_doc: Optional[Path] = None,
    metadata: Optional[dict[str, str]] = None,
) -> None:
    """Render a single markdown file to PDF or DOCX."""
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        pandoc,
        str(source_md),
        "-f", "gfm+tex_math_dollars",
        "--resource-path", str(resource_path),
        "--toc",
        "--toc-depth=3",
        "--standalone",
        "-o", str(output),
    ]

    if fmt == "pdf":
        cmd += [
            "--pdf-engine", pdf_engine,
            "-V", "geometry:margin=1in",
            "-V", "mainfont=DejaVu Sans",
            "-V", "monofont=DejaVu Sans Mono",
            "-V", "fontsize=11pt",
            "-V", "colorlinks=true",
            "-V", "linkcolor=NavyBlue",
            "--highlight-style=tango",
        ]
    elif fmt == "docx":
        if reference_doc and reference_doc.exists():
            cmd += ["--reference-doc", str(reference_doc)]
        cmd += ["--highlight-style=tango"]

    # Note: we do NOT pass --number-sections because Manning chapter
    # headings already contain manual section numbers (e.g., "## 1.2 Title").

    if metadata:
        for key, val in metadata.items():
            cmd += ["-M", f"{key}={val}"]

    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Per-chapter builds
# ---------------------------------------------------------------------------

def build_chapter(
    pandoc: str,
    chapter: Chapter,
    build_dir: Path,
    output_dir: Path,
    fmt: str,
    pdf_engine: str,
    reference_doc: Optional[Path],
    verbose: bool = False,
) -> Path:
    """Build a single chapter. Returns the output path."""
    ext = ".pdf" if fmt == "pdf" else ".docx"
    output = output_dir / f"{chapter.path.stem}{ext}"

    # Preprocess: copy markdown and download remote images
    build_md = build_dir / f"{chapter.path.stem}.md"
    build_md.parent.mkdir(parents=True, exist_ok=True)

    text = chapter.path.read_text(encoding="utf-8")
    asset_dir = build_dir / "_assets" / f"ch{chapter.number:02d}"
    text = rewrite_remote_images(text, asset_dir=asset_dir, md_dir=build_md.parent)
    build_md.write_text(text, encoding="utf-8")

    if verbose:
        print(f"  Rendering {chapter.path.name} -> {output.name}")

    render_file(
        pandoc=pandoc,
        source_md=build_md,
        output=output,
        resource_path=build_dir,
        fmt=fmt,
        pdf_engine=pdf_engine,
        reference_doc=reference_doc,
        metadata={
            "title": chapter.title,
            "author": BOOK_AUTHOR,
            "date": date.today().isoformat(),
        },
    )
    return output


# ---------------------------------------------------------------------------
# Combined book build
# ---------------------------------------------------------------------------

def build_combined(
    pandoc: str,
    chapters: list[Chapter],
    build_dir: Path,
    output_dir: Path,
    fmt: str,
    pdf_engine: str,
    reference_doc: Optional[Path],
    verbose: bool = False,
) -> Path:
    """Assemble all chapters into a single manuscript file."""
    ext = ".pdf" if fmt == "pdf" else ".docx"
    output = output_dir / f"manuscript{ext}"

    parts: list[str] = []

    # YAML front matter for pandoc metadata
    parts.append(
        f'---\n'
        f'title: "{BOOK_TITLE}"\n'
        f'subtitle: "{BOOK_SUBTITLE}"\n'
        f'author: "{BOOK_AUTHOR}"\n'
        f'date: "{date.today().isoformat()}"\n'
        f'---\n'
    )

    chapter_by_num = {ch.number: ch for ch in chapters}

    for part_title, ch_nums in BOOK_PARTS:
        part_chapters = [chapter_by_num[n] for n in ch_nums if n in chapter_by_num]
        if not part_chapters:
            continue

        # Part divider -- centered bold title with page break
        parts.append(f"\n\\newpage\n")
        parts.append(f"\n---\n\n**{part_title}**\n\n---\n")

        for ch in part_chapters:
            text = ch.path.read_text(encoding="utf-8")
            # Download remote images
            asset_dir = build_dir / "_assets" / f"ch{ch.number:02d}"
            text = rewrite_remote_images(
                text, asset_dir=asset_dir, md_dir=build_dir,
            )
            parts.append(f"\n\\newpage\n\n{text}\n")

    combined_md = build_dir / "manuscript.md"
    combined_md.write_text("\n".join(parts), encoding="utf-8")

    if verbose:
        ch_list = ", ".join(f"Ch{ch.number}" for ch in chapters)
        print(f"  Building combined manuscript ({ch_list}) -> {output.name}")

    render_file(
        pandoc=pandoc,
        source_md=combined_md,
        output=output,
        resource_path=build_dir,
        fmt=fmt,
        pdf_engine=pdf_engine,
        reference_doc=reference_doc,
    )
    return output


# ---------------------------------------------------------------------------
# Reference doc generation
# ---------------------------------------------------------------------------

def init_reference_doc(pandoc: str, output: Path) -> None:
    """Generate a default DOCX reference template for style customization."""
    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [pandoc, "--print-default-data-file", "reference.docx"],
        capture_output=True,
        check=True,
    )
    output.write_bytes(result.stdout)
    print(f"Reference doc created: {output}")
    print(
        "Open in Word or LibreOffice to customize heading styles, fonts, "
        "and spacing. Future builds will use it automatically."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Manning book chapters as PDF and/or DOCX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/build_book.py                      # All chapters, PDF + DOCX\n"
            "  python scripts/build_book.py --format pdf          # PDF only\n"
            "  python scripts/build_book.py --chapters 1 5        # Specific chapters\n"
            "  python scripts/build_book.py --combined             # Single manuscript\n"
            "  python scripts/build_book.py --validate-only        # Check without building\n"
            "  python scripts/build_book.py --init-reference-doc   # Create DOCX template\n"
        ),
    )
    parser.add_argument(
        "--chapters",
        type=int,
        nargs="+",
        metavar="N",
        help="Chapter numbers to build (default: all).",
    )
    parser.add_argument(
        "--format",
        choices=["pdf", "docx", "both"],
        default="both",
        help="Output format (default: both).",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Build a single combined manuscript instead of per-chapter files.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run validation checks without building.",
    )
    parser.add_argument(
        "--init-reference-doc",
        action="store_true",
        help="Generate a DOCX reference template at Manning/reference.docx.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR.relative_to(REPO_ROOT)}).",
    )
    parser.add_argument(
        "--pdf-engine",
        default="xelatex",
        help="Pandoc PDF engine (default: xelatex).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        metavar="PATH",
        help="Write validation report as JSON to this path.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress details.",
    )
    args = parser.parse_args()

    # --- Reference doc generation (standalone action) ---
    if args.init_reference_doc:
        pandoc = _require_cmd("pandoc")
        init_reference_doc(pandoc, REFERENCE_DOC)
        return 0

    # --- Discover chapters ---
    if not CHAPTERS_DIR.is_dir():
        print(f"Error: chapters directory not found: {CHAPTERS_DIR}", file=sys.stderr)
        return 1

    chapters = discover_chapters(CHAPTERS_DIR, numbers=args.chapters)
    if not chapters:
        nums = args.chapters or "any"
        print(f"Error: no chapters found matching {nums} in {CHAPTERS_DIR}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Found {len(chapters)} chapter(s):")
        for ch in chapters:
            print(f"  Ch{ch.number}: {ch.path.name} ({ch.title})")

    # --- Validation ---
    report = ValidationReport(chapters_checked=len(chapters))
    for ch in chapters:
        report.issues.extend(validate_chapter(ch))

    # Print validation summary
    if report.issues:
        print(f"\nValidation: {len(report.errors)} error(s), {len(report.warnings)} warning(s)")
        for issue in report.issues:
            if issue.severity == "info" and not args.verbose:
                continue
            loc = f"Ch{issue.chapter}"
            if issue.line:
                loc += f":{issue.line}"
            print(f"  [{issue.severity.upper():7s}] {loc} ({issue.category}) {issue.message}")
    else:
        print("\nValidation: all checks passed")

    # Write JSON report if requested
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8",
        )
        if args.verbose:
            print(f"Report written to {args.report}")

    # Stop here if validate-only
    if args.validate_only:
        return 1 if report.errors else 0

    # Abort on errors
    if report.errors:
        print("\nBuild aborted due to validation errors.", file=sys.stderr)
        return 1

    # --- Build ---
    pandoc = _require_cmd("pandoc")
    formats = ["pdf", "docx"] if args.format == "both" else [args.format]
    out_dir = args.out_dir.resolve()
    ref_doc = REFERENCE_DOC if REFERENCE_DOC.exists() else None

    outputs: list[Path] = []

    with tempfile.TemporaryDirectory(prefix="book-build-") as tmp:
        build_dir = Path(tmp)

        for fmt in formats:
            if args.combined:
                out = build_combined(
                    pandoc=pandoc,
                    chapters=chapters,
                    build_dir=build_dir,
                    output_dir=out_dir,
                    fmt=fmt,
                    pdf_engine=args.pdf_engine,
                    reference_doc=ref_doc,
                    verbose=args.verbose,
                )
                outputs.append(out)
            else:
                for ch in chapters:
                    out = build_chapter(
                        pandoc=pandoc,
                        chapter=ch,
                        build_dir=build_dir,
                        output_dir=out_dir,
                        fmt=fmt,
                        pdf_engine=args.pdf_engine,
                        reference_doc=ref_doc,
                        verbose=args.verbose,
                    )
                    outputs.append(out)

    # --- Summary ---
    print(f"\nBuild complete. {len(outputs)} file(s) produced:")
    for p in outputs:
        size_kb = p.stat().st_size / 1024
        print(f"  {p.relative_to(REPO_ROOT)}  ({size_kb:.0f} KB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
