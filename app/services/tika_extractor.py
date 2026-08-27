import codecs
import os
from dataclasses import dataclass
from typing import Iterable

import requests

from app.core.config import settings


@dataclass
class TikaTextBlock:
    content: str
    source_type: str
    page_no: int | None = None
    slide_no: int | None = None
    sheet_name: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None


class TikaExtractor:
    """
    基于 Apache Tika Server 的统一文档解析器。

    作用：
    1. 把 PDF / Word / PPT / Excel / HTML / TXT 等统一交给 Tika 解析；
    2. Python 侧使用 stream=True 流式接收 Tika 返回文本；
    3. 按 parse_parent_block_size_bytes 累计 parent block；
    4. parent block 再交给 ChunkService 切成最终 document_chunk。
    """

    def __init__(self):
        self.tika_url = settings.tika_server_url.rstrip("/")
        self.timeout = settings.tika_timeout_seconds
        self.stream_chunk_size = settings.tika_stream_chunk_size_bytes
        self.parent_block_size_bytes = settings.parse_parent_block_size_bytes

    def extract(self, local_path: str) -> Iterable[TikaTextBlock]:
        ext = os.path.splitext(local_path)[1].lower().lstrip(".") or "unknown"
        url = f"{self.tika_url}/tika"

        headers = {
            "Accept": "text/plain",
        }

        with open(local_path, "rb") as f:
            response = requests.put(
                url,
                data=f,
                headers=headers,
                timeout=self.timeout,
                stream=True,
            )

            if response.status_code >= 400:
                raise ValueError(
                    f"Tika 解析失败 status={response.status_code}, "
                    f"body={response.text[:500]}"
                )

            yield from self._iter_parent_blocks(
                response=response,
                source_type=ext,
            )

    def _iter_parent_blocks(
        self,
        response: requests.Response,
        source_type: str,
    ) -> Iterable[TikaTextBlock]:
        """
        流式接收 Tika 返回文本，并累计到约 1MB parent block。
        """
        decoder = codecs.getincrementaldecoder("utf-8")("replace")

        buffer = ""
        start_offset = 0

        for raw_chunk in response.iter_content(chunk_size=self.stream_chunk_size):
            if not raw_chunk:
                continue

            text_part = decoder.decode(raw_chunk)
            if not text_part:
                continue

            buffer += text_part

            while len(buffer.encode("utf-8")) >= self.parent_block_size_bytes:
                emit_text, buffer = self._split_buffer_by_paragraph(buffer)

                if emit_text.strip():
                    end_offset = start_offset + len(emit_text)

                    yield TikaTextBlock(
                        content=emit_text.strip(),
                        source_type=source_type,
                        start_offset=start_offset,
                        end_offset=end_offset,
                    )

                    start_offset = end_offset

        tail = decoder.decode(b"", final=True)
        if tail:
            buffer += tail

        if buffer.strip():
            yield TikaTextBlock(
                content=buffer.strip(),
                source_type=source_type,
                start_offset=start_offset,
                end_offset=start_offset + len(buffer),
            )

    def _split_buffer_by_paragraph(self, buffer: str) -> tuple[str, str]:
        """
        buffer 已经超过 parent_block_size_bytes。
        尽量在靠近 1MB 的位置找段落边界切分。
        如果找不到段落边界，再按字符位置硬切。
        """
        max_bytes = self.parent_block_size_bytes
        current_bytes = buffer.encode("utf-8")

        if len(current_bytes) <= max_bytes:
            return buffer, ""

        byte_count = 0
        cut_pos = 0

        for idx, ch in enumerate(buffer):
            byte_count += len(ch.encode("utf-8"))
            if byte_count > max_bytes:
                break
            cut_pos = idx + 1

        candidate = buffer[:cut_pos]

        # 优先在候选区域后 40% 找段落边界，避免切得太早
        search_start = max(0, int(len(candidate) * 0.6))

        split_pos = candidate.rfind("\n\n", search_start)
        if split_pos > 0:
            return buffer[:split_pos].strip(), buffer[split_pos:].lstrip()

        split_pos = candidate.rfind("\n", search_start)
        if split_pos > 0:
            return buffer[:split_pos].strip(), buffer[split_pos:].lstrip()

        return buffer[:cut_pos].strip(), buffer[cut_pos:].lstrip()