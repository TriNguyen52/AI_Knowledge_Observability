"""Markdown document parsing and normalization into AI-Ready models."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ai_ready.models import CodeBlock, DocumentContent, Heading, KnowledgeArtifact, Link, Paragraph, Section

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
RST_HEADING_RE = re.compile(r"^([^\n]+)\n([=\-~`'\"^_*+#])\2+\s*$", re.MULTILINE)
MD_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
CODE_FENCE_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)


class MarkdownDocumentParser:
    """Parse markdown-ish documents into normalized AI-Ready document models."""

    def parse(self, filepath: Path, source_root: Path | None = None) -> KnowledgeArtifact:
        text = filepath.read_text(encoding="utf-8", errors="replace")
        relative_path = self._relative_path(filepath, source_root)

        artifact_id = hashlib.sha256(relative_path.encode()).hexdigest()[:16]
        metadata = self._extract_frontmatter(text)
        body = self._strip_frontmatter(text)

        headings = self._extract_headings(body, filepath.suffix.lower())
        sections = self._extract_sections(body, headings)
        paragraphs = self._extract_paragraphs(sections)
        links = self._extract_links(body)
        code_blocks = self._extract_code_blocks(body)

        title = self._extract_title(filepath, metadata, headings)

        content = DocumentContent(
            headings=headings,
            sections=sections,
            paragraphs=paragraphs,
            links=links,
            code_blocks=code_blocks,
        )

        return KnowledgeArtifact(
            id=artifact_id,
            uri=relative_path,
            title=title,
            content=content,
            metadata=metadata,
        )

    def _relative_path(self, filepath: Path, source_root: Path | None) -> str:
        if source_root and source_root.is_dir():
            try:
                return str(filepath.relative_to(source_root)).replace("\\", "/")
            except ValueError:
                pass
        return str(filepath).replace("\\", "/")

    def _extract_frontmatter(self, text: str) -> dict:
        match = FRONTMATTER_RE.match(text)
        if not match:
            return {}

        raw = match.group(1)
        result: dict = {}
        for line in raw.split("\n"):
            if ":" in line:
                key, _, value = line.partition(":")
                result[key.strip()] = value.strip()
        return result

    def _strip_frontmatter(self, text: str) -> str:
        match = FRONTMATTER_RE.match(text)
        if match:
            return text[match.end():]
        return text

    def _extract_headings(self, body: str, ext: str) -> list[Heading]:
        headings: list[Heading] = []
        if ext in {".md", ".markdown", ".mdx"}:
            for line_number, line in enumerate(body.split("\n"), 1):
                match = re.match(r"^(#{1,6})\s+(.+)$", line)
                if match:
                    headings.append(
                        Heading(level=len(match.group(1)), text=match.group(2).strip(), line=line_number)
                    )
        elif ext == ".rst":
            for match in RST_HEADING_RE.finditer(body):
                line_number = body[: match.start()].count("\n") + 1
                underline_char = match.group(2)[0]
                level = {"=": 1, "-": 2, "~": 3, "`": 4, "'": 5, '"': 6}.get(underline_char, 2)
                headings.append(Heading(level=level, text=match.group(1).strip(), line=line_number))
        return headings

    def _extract_sections(self, body: str, headings: list[Heading]) -> list[Section]:
        if not headings:
            return [Section(heading=None, text=body.strip(), line_start=1)]

        sections: list[Section] = []
        lines = body.split("\n")

        for index, heading in enumerate(headings):
            start = heading.line
            end = headings[index + 1].line - 1 if index + 1 < len(headings) else len(lines)
            section_lines = lines[start:end]
            sections.append(
                Section(
                    heading=heading,
                    text="\n".join(section_lines).strip(),
                    line_start=start,
                    line_end=end,
                )
            )

        if headings[0].line > 1:
            pre_text = "\n".join(lines[: headings[0].line - 1]).strip()
            if pre_text:
                sections.insert(
                    0,
                    Section(heading=None, text=pre_text, line_start=1, line_end=headings[0].line - 1),
                )

        return sections

    def _extract_paragraphs(self, sections: list[Section]) -> list[Paragraph]:
        paragraphs: list[Paragraph] = []
        for section in sections:
            heading_text = section.heading.text if section.heading else ""
            for paragraph in section.text.split("\n\n"):
                paragraph = paragraph.strip()
                if paragraph and not paragraph.startswith("```") and not paragraph.startswith("|"):
                    paragraphs.append(
                        Paragraph(
                            text=paragraph,
                            section_heading=heading_text,
                            line=section.line_start,
                        )
                    )
        return paragraphs

    def _extract_links(self, body: str) -> list[Link]:
        links: list[Link] = []
        for line_number, line in enumerate(body.split("\n"), 1):
            for match in MD_LINK_RE.finditer(line):
                target = match.group(2).strip()
                is_internal = not target.startswith(("http://", "https://"))
                links.append(Link(text=match.group(1), target=target, line=line_number, is_internal=is_internal))
        return links

    def _extract_code_blocks(self, body: str) -> list[CodeBlock]:
        blocks: list[CodeBlock] = []
        for match in CODE_FENCE_RE.finditer(body):
            line_number = body[: match.start()].count("\n") + 1
            blocks.append(CodeBlock(language=match.group(1) or "text", text=match.group(2), line=line_number))
        return blocks

    def _extract_title(self, filepath: Path, metadata: dict, headings: list[Heading]) -> str:
        if "title" in metadata:
            return metadata["title"]
        if headings:
            return headings[0].text
        return filepath.stem