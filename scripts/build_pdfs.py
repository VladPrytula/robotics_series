#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse


@dataclass(frozen=True)
class PdfSource:
    source_path: Path
    output_path: Path


_REMOTE_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\((https?://[^)\s]+)\)")


def _require_cmd(cmd: str) -> str:
    resolved = shutil.which(cmd)
    if not resolved:
        raise RuntimeError(f"Required command not found in PATH: {cmd}")
    return resolved


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, out_path.open("wb") as f:
        f.write(response.read())


def _convert_gif_to_png(gif_path: Path, png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)

    magick = shutil.which("magick")
    if magick:
        subprocess.run([magick, str(gif_path), str(png_path)], check=True)
        return

    convert = shutil.which("convert")
    if convert:
        subprocess.run([convert, str(gif_path), str(png_path)], check=True)
        return

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run([ffmpeg, "-y", "-i", str(gif_path), str(png_path)], check=True)
        return

    raise RuntimeError("GIF conversion requires one of: magick, convert, ffmpeg")


def _local_filename_for_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    if not name:
        return "image"
    return name


def _rewrite_remote_images(markdown_text: str, asset_dir: Path, build_dir: Path) -> str:
    url_to_relpath: dict[str, str] = {}

    def repl(match: re.Match[str]) -> str:
        alt_text = match.group(1)
        url = match.group(2)
        if url not in url_to_relpath:
            filename = _local_filename_for_url(url)
            suffix = Path(filename).suffix.lower()
            downloaded = asset_dir / filename
            _download(url, downloaded)

            final_path = downloaded
            if suffix == ".gif":
                final_path = downloaded.with_suffix(".png")
                _convert_gif_to_png(downloaded, final_path)

            url_to_relpath[url] = str(final_path.relative_to(build_dir))
        return f"![{alt_text}]({url_to_relpath[url]})"

    return _REMOTE_IMAGE_RE.sub(repl, markdown_text)


def _render_pdf(pandoc: str, source_md: Path, output_pdf: Path, resource_path: Path, pdf_engine: str) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            pandoc,
            str(source_md),
            "-f",
            "gfm+tex_math_dollars",
            "--pdf-engine",
            pdf_engine,
            "--resource-path",
            str(resource_path),
            "--toc",
            "--toc-depth=3",
            "--number-sections",
            "-V",
            "geometry:margin=1in",
            "-o",
            str(output_pdf),
        ],
        check=True,
    )


def _default_sources(repo_root: Path, out_dir: Path) -> list[PdfSource]:
    sources: list[Path] = [
        repo_root / "README.md",
        repo_root / "tutorials" / "README.md",
    ]
    sources.extend(sorted((repo_root / "tutorials").glob("ch*.md")))

    result: list[PdfSource] = []
    for src in sources:
        if not src.exists():
            continue
        rel = src.relative_to(repo_root)
        result.append(PdfSource(source_path=src, output_path=(out_dir / rel).with_suffix(".pdf")))
    return result


def _iter_sources(paths: Iterable[str], repo_root: Path, out_dir: Path) -> list[PdfSource]:
    resolved: list[Path] = []
    for raw in paths:
        p = (repo_root / raw).resolve() if not os.path.isabs(raw) else Path(raw).resolve()
        if p.is_dir():
            resolved.extend(sorted(p.rglob("*.md")))
        else:
            resolved.append(p)

    result: list[PdfSource] = []
    for src in resolved:
        if not src.exists():
            raise FileNotFoundError(str(src))
        try:
            rel = src.relative_to(repo_root)
        except ValueError:
            raise ValueError(f"Source must be inside repo: {src}") from None
        result.append(PdfSource(source_path=src, output_path=(out_dir / rel).with_suffix(".pdf")))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PDFs from README/tutorial markdown via pandoc.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/pdfs"),
        help="Output directory for generated PDFs (default: docs/pdfs).",
    )
    parser.add_argument(
        "--pdf-engine",
        default="xelatex",
        help="Pandoc PDF engine (default: xelatex).",
    )
    parser.add_argument(
        "sources",
        nargs="*",
        help="Optional source files/dirs (repo-relative). Defaults to README.md + tutorials/*.md.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = (repo_root / args.out_dir).resolve()

    pandoc = _require_cmd("pandoc")

    pdf_sources = (
        _iter_sources(args.sources, repo_root=repo_root, out_dir=out_dir)
        if args.sources
        else _default_sources(repo_root=repo_root, out_dir=out_dir)
    )
    if not pdf_sources:
        raise RuntimeError("No markdown sources found.")

    with tempfile.TemporaryDirectory(prefix="pandoc-build-") as build_tmp:
        build_dir = Path(build_tmp)
        for item in pdf_sources:
            rel = item.source_path.relative_to(repo_root)
            build_md = build_dir / rel
            build_md.parent.mkdir(parents=True, exist_ok=True)

            text = item.source_path.read_text(encoding="utf-8")
            asset_dir = build_dir / "_assets" / rel.parent
            rewritten = _rewrite_remote_images(text, asset_dir=asset_dir, build_dir=build_dir)
            build_md.write_text(rewritten, encoding="utf-8")

            _render_pdf(
                pandoc=pandoc,
                source_md=build_md,
                output_pdf=item.output_path,
                resource_path=build_dir,
                pdf_engine=args.pdf_engine,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

