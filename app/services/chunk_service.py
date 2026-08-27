import hashlib
import re
import uuid
from dataclasses import dataclass

from app.core.config import settings
from app.models.document_chunk import DocumentChunk


@dataclass
class TextBlock:
    content: str
    block_type: str = "paragraph"
    section_title: str | None = None
    heading_path: str | None = None
    page_no: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None


class ChunkService:
    """
    文档分块服务。

    第一版目标：
    1. Markdown 标题结构保留
    2. 表格整体保留
    3. 代码块整体保留
    4. 段落 / 句子优先切分
    5. 超长文本按长度兜底
    6. overlap 保留跨块上下文

    后续增强：
    1. token 级切分
    2. PDF 页码级分块
    3. 表格结构化解析
    4. embedding 前去重
    """

    def __init__(self):
        self.chunk_size_chars = settings.chunk_size_chars
        self.overlap_chars = settings.chunk_overlap_chars
        self.min_chunk_chars = settings.chunk_min_chars
        self.max_chunk_chars = settings.chunk_max_chars

        self._hanlp_sentence_splitter = None

    def split_text(
        self,
        document_uid: str,
        project_uid: str,
        text: str,
        version: int = 1,
        page_no: int | None = None,
    ) -> list[DocumentChunk]:
        normalized = self._normalize_text(text)
        if not normalized:
            return []

        blocks = self._parse_markdown_like_blocks(
            text=normalized,
            default_page_no=page_no,
        )

        chunk_blocks = self._build_chunk_blocks(blocks)

        chunks: list[DocumentChunk] = []
        for idx, block in enumerate(chunk_blocks):
            content = block.content.strip()
            if not content:
                continue

            content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
            chunk_uid = "chunk_" + uuid.uuid4().hex[:24]

            chunks.append(
                DocumentChunk(
                    chunk_uid=chunk_uid,
                    document_uid=document_uid,
                    project_uid=project_uid,
                    version=version,
                    chunk_index=len(chunks),
                    content=content,
                    content_hash=content_hash,
                    section_title=block.section_title,
                    heading_path=block.heading_path,
                    block_type=block.block_type,
                    page_no=block.page_no,
                    start_offset=block.start_offset,
                    end_offset=block.end_offset,
                    token_count=self._rough_token_count(content),
                    embedding_status="pending",
                    index_status="pending",
                )
            )

        return chunks

    def _normalize_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.rstrip() for line in text.split("\n")]
        return "\n".join(lines).strip()

    def _parse_markdown_like_blocks(
        self,
        text: str,
        default_page_no: int | None = None,
    ) -> list[TextBlock]:
        """
        解析 Markdown-like 文本，保留标题、表格、代码块。

        注意：
        PDF / Word 解析出来的纯文本也可以走这里；
        如果没有 Markdown 标题，就退化为普通段落分块。
        """
        lines = text.split("\n")
        blocks: list[TextBlock] = []

        heading_stack: list[tuple[int, str]] = []
        current_paragraph: list[str] = []
        current_start_offset = 0
        offset = 0

        in_code_block = False
        code_lines: list[str] = []
        code_start_offset = 0

        i = 0
        while i < len(lines):
            line = lines[i]
            raw_line = line
            stripped = line.strip()
            line_start_offset = offset
            offset += len(raw_line) + 1

            # 代码块开始 / 结束
            if stripped.startswith("```"):
                if not in_code_block:
                    self._flush_paragraph(
                        blocks,
                        current_paragraph,
                        current_start_offset,
                        line_start_offset,
                        heading_stack,
                        default_page_no,
                    )
                    current_paragraph = []

                    in_code_block = True
                    code_lines = [raw_line]
                    code_start_offset = line_start_offset
                else:
                    code_lines.append(raw_line)
                    in_code_block = False
                    blocks.append(
                        TextBlock(
                            content="\n".join(code_lines),
                            block_type="code",
                            section_title=self._current_section_title(heading_stack),
                            heading_path=self._heading_path(heading_stack),
                            page_no=default_page_no,
                            start_offset=code_start_offset,
                            end_offset=offset,
                        )
                    )
                    code_lines = []

                i += 1
                continue

            if in_code_block:
                code_lines.append(raw_line)
                i += 1
                continue

            # 标题
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
            if heading_match:
                self._flush_paragraph(
                    blocks,
                    current_paragraph,
                    current_start_offset,
                    line_start_offset,
                    heading_stack,
                    default_page_no,
                )
                current_paragraph = []

                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()

                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))

                blocks.append(
                    TextBlock(
                        content=stripped,
                        block_type="heading",
                        section_title=title,
                        heading_path=self._heading_path(heading_stack),
                        page_no=default_page_no,
                        start_offset=line_start_offset,
                        end_offset=offset,
                    )
                )

                i += 1
                continue

            # Markdown 表格：连续多行包含 |
            if self._is_table_line(stripped):
                self._flush_paragraph(
                    blocks,
                    current_paragraph,
                    current_start_offset,
                    line_start_offset,
                    heading_stack,
                    default_page_no,
                )
                current_paragraph = []

                table_lines = [raw_line]
                table_start_offset = line_start_offset
                i += 1

                while i < len(lines):
                    next_line = lines[i]
                    next_stripped = next_line.strip()
                    if not self._is_table_line(next_stripped):
                        break

                    table_lines.append(next_line)
                    offset += len(next_line) + 1
                    i += 1

                blocks.append(
                    TextBlock(
                        content="\n".join(table_lines),
                        block_type="table",
                        section_title=self._current_section_title(heading_stack),
                        heading_path=self._heading_path(heading_stack),
                        page_no=default_page_no,
                        start_offset=table_start_offset,
                        end_offset=offset,
                    )
                )
                continue

            # 空行：结束段落
            if not stripped:
                self._flush_paragraph(
                    blocks,
                    current_paragraph,
                    current_start_offset,
                    line_start_offset,
                    heading_stack,
                    default_page_no,
                )
                current_paragraph = []
                i += 1
                continue

            # 普通段落
            if not current_paragraph:
                current_start_offset = line_start_offset
            current_paragraph.append(raw_line)
            i += 1

        if in_code_block and code_lines:
            blocks.append(
                TextBlock(
                    content="\n".join(code_lines),
                    block_type="code",
                    section_title=self._current_section_title(heading_stack),
                    heading_path=self._heading_path(heading_stack),
                    page_no=default_page_no,
                    start_offset=code_start_offset,
                    end_offset=offset,
                )
            )

        self._flush_paragraph(
            blocks,
            current_paragraph,
            current_start_offset,
            offset,
            heading_stack,
            default_page_no,
        )

        return [b for b in blocks if b.content.strip()]

    def _flush_paragraph(
        self,
        blocks: list[TextBlock],
        paragraph_lines: list[str],
        start_offset: int,
        end_offset: int,
        heading_stack: list[tuple[int, str]],
        page_no: int | None,
    ) -> None:
        if not paragraph_lines:
            return

        content = "\n".join(paragraph_lines).strip()
        if not content:
            return

        blocks.append(
            TextBlock(
                content=content,
                block_type="paragraph",
                section_title=self._current_section_title(heading_stack),
                heading_path=self._heading_path(heading_stack),
                page_no=page_no,
                start_offset=start_offset,
                end_offset=end_offset,
            )
        )

    def _is_table_line(self, line: str) -> bool:
        if not line:
            return False
        if "|" not in line:
            return False
        # 至少两个竖线，更像表格
        return line.count("|") >= 2

    def _current_section_title(self, heading_stack: list[tuple[int, str]]) -> str | None:
        if not heading_stack:
            return None
        return heading_stack[-1][1]

    def _heading_path(self, heading_stack: list[tuple[int, str]]) -> str | None:
        if not heading_stack:
            return None
        return " > ".join(title for _, title in heading_stack)

    def _build_chunk_blocks(self, blocks: list[TextBlock]) -> list[TextBlock]:
        """
        将结构块组合成最终 chunk。
        表格和代码块尽量整体保留。
        普通段落按句子和长度继续切。
        """
        result: list[TextBlock] = []
        buffer_blocks: list[TextBlock] = []

        def flush_buffer():
            nonlocal buffer_blocks
            if not buffer_blocks:
                return

            merged = self._merge_blocks(buffer_blocks)
            for chunk in self._split_if_too_large(merged):
                result.append(chunk)

            buffer_blocks = []

        for block in blocks:
            # 标题本身不单独成为很小 chunk，而是并入后续内容
            if block.block_type == "heading":
                flush_buffer()
                buffer_blocks.append(block)
                continue

            # 表格、代码块如果不大，整体保留
            if block.block_type in {"table", "code"}:
                flush_buffer()
                if len(block.content) <= self.max_chunk_chars:
                    result.append(block)
                else:
                    for chunk in self._split_if_too_large(block):
                        result.append(chunk)
                continue

            # 普通段落
            candidate_blocks = buffer_blocks + [block]
            candidate_text = "\n\n".join(b.content for b in candidate_blocks)

            if len(candidate_text) <= self.chunk_size_chars:
                buffer_blocks.append(block)
            else:
                flush_buffer()
                if len(block.content) <= self.chunk_size_chars:
                    buffer_blocks.append(block)
                else:
                    for chunk in self._split_if_too_large(block):
                        result.append(chunk)

        flush_buffer()

        result = self._merge_tiny_chunks(result)
        result = self._apply_overlap(result)

        return result

    def _merge_blocks(self, blocks: list[TextBlock]) -> TextBlock:
        content = "\n\n".join(b.content for b in blocks if b.content.strip())

        first = blocks[0]
        last = blocks[-1]

        block_types = {b.block_type for b in blocks}
        block_type = "mixed" if len(block_types) > 1 else first.block_type

        section_title = last.section_title or first.section_title
        heading_path = last.heading_path or first.heading_path

        return TextBlock(
            content=content,
            block_type=block_type,
            section_title=section_title,
            heading_path=heading_path,
            page_no=first.page_no,
            start_offset=first.start_offset,
            end_offset=last.end_offset,
        )

    def _split_if_too_large(self, block: TextBlock) -> list[TextBlock]:
        if len(block.content) <= self.chunk_size_chars:
            return [block]

        sentences = self._split_sentences(block.content)

        chunks: list[str] = []
        buf = ""

        for sent in sentences:
            if not sent.strip():
                continue

            if not buf:
                buf = sent
                continue

            if len(buf) + len(sent) <= self.chunk_size_chars:
                buf += sent
            else:
                chunks.extend(self._hard_split_with_overlap(buf))
                buf = sent

        if buf:
            chunks.extend(self._hard_split_with_overlap(buf))

        result = []
        cursor = block.start_offset or 0

        for text in chunks:
            result.append(
                TextBlock(
                    content=text.strip(),
                    block_type=block.block_type,
                    section_title=block.section_title,
                    heading_path=block.heading_path,
                    page_no=block.page_no,
                    start_offset=cursor,
                    end_offset=cursor + len(text),
                )
            )
            cursor += len(text)

        return result

    def _hard_split_with_overlap(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size_chars:
            return [text]

        result = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size_chars, len(text))
            chunk = text[start:end].strip()

            if chunk:
                result.append(chunk)

            if end >= len(text):
                break

            start = max(0, end - self.overlap_chars)

        return result

    def _split_sentences(self, text: str) -> list[str]:
        """
        优先 HanLP 分句，失败则正则兜底。
        """
        hanlp_result = self._split_sentences_by_hanlp(text)
        if hanlp_result:
            return hanlp_result

        # 中文和英文标点后切分
        parts = re.split(r"(?<=[。！？；.!?;])", text)
        return [p for p in parts if p.strip()]

    def _split_sentences_by_hanlp(self, text: str) -> list[str]:
        """
        HanLP 是可选增强。由 settings.hanlp_enabled 控制开关。
        网络不通或未安装时设为 false，自动 fallback 到正则分句。
        """
        if not settings.hanlp_enabled:
            return []

        try:
            if self._hanlp_sentence_splitter is None:
                import hanlp

                self._hanlp_sentence_splitter = hanlp.load(
                    hanlp.pretrained.eos.UD_CTB_EOS_MUL
                )

            result = self._hanlp_sentence_splitter(text)
            if isinstance(result, list):
                return [s for s in result if isinstance(s, str) and s.strip()]
        except Exception:
            return []

        return []

    def _merge_tiny_chunks(self, chunks: list[TextBlock]) -> list[TextBlock]:
        if not chunks:
            return []

        merged: list[TextBlock] = []
        buffer = chunks[0]

        for chunk in chunks[1:]:
            if len(buffer.content) < self.min_chunk_chars:
                buffer = self._merge_blocks([buffer, chunk])
            else:
                merged.append(buffer)
                buffer = chunk

        if buffer.content:
            merged.append(buffer)

        return merged

    def _apply_overlap(self, chunks: list[TextBlock]) -> list[TextBlock]:
        """
        给相邻文本块增加 overlap。
        表格和代码块不强行加 overlap，避免破坏结构。
        """
        if not chunks or self.overlap_chars <= 0:
            return chunks

        result: list[TextBlock] = []

        for idx, chunk in enumerate(chunks):
            if idx == 0:
                result.append(chunk)
                continue

            prev = result[-1]

            if chunk.block_type in {"table", "code"}:
                result.append(chunk)
                continue

            if prev.block_type in {"table", "code"}:
                result.append(chunk)
                continue

            overlap_text = prev.content[-self.overlap_chars:].strip()
            if overlap_text and overlap_text not in chunk.content:
                chunk = TextBlock(
                    content=overlap_text + "\n\n" + chunk.content,
                    block_type=chunk.block_type,
                    section_title=chunk.section_title,
                    heading_path=chunk.heading_path,
                    page_no=chunk.page_no,
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                )

            result.append(chunk)

        return result

    def _rough_token_count(self, text: str) -> int:
        """
        第一版仍然保留粗略 token 估计，但比 return len(text) 稍微合理：
        - 中文字符按 1 token 粗估
        - 英文单词按 1 token 粗估
        - 数字和符号粗略计入
        后续接 tokenizer 后替换这里。
        """
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        english_words = len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))
        numbers = len(re.findall(r"\d+(?:\.\d+)?", text))
        other_chars = len(re.findall(r"[^\s\u4e00-\u9fffA-Za-z0-9]", text))

        return chinese_chars + english_words + numbers + max(1, other_chars // 4)