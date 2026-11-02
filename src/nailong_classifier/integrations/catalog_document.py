from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

from lark_channel.client import Client
from lark_channel.core.enum import AccessTokenType, HttpMethod
from lark_channel.core.model import BaseRequest
from requests_toolbelt import MultipartEncoder

DOCUMENT_EDIT_INTERVAL_SECONDS = 0.4
IMAGE_BLOCK_TYPE = 27
GRID_BLOCK_TYPE = 24
GRID_COLUMN_BLOCK_TYPE = 25
HEADING_1_BLOCK_TYPE = 3
CLIENT_TOKEN_NAMESPACE = uuid.UUID("ea1d63de-e022-4c79-86d9-0935f6c77ceb")


class CatalogDocumentError(RuntimeError):
    def __init__(
        self,
        operation: str,
        *,
        status_code: int | None,
        code: int | None,
        message: str,
    ) -> None:
        super().__init__(
            f"{operation} failed: status={status_code} code={code} message={message}"
        )
        self.operation = operation
        self.status_code = status_code
        self.code = code


class CatalogDocumentClient:
    def __init__(
        self,
        client: Client,
        *,
        document_id: str,
        section_block_id: str,
    ) -> None:
        self.client = client
        self.document_id = document_id
        self.section_block_id = section_block_id
        self._edit_lock = asyncio.Lock()
        self._last_edit_at = 0.0

    def client_token(self, asset_sha256: str, operation: str) -> str:
        value = f"{self.document_id}:{asset_sha256}:{operation}"
        return str(uuid.uuid5(CLIENT_TOKEN_NAMESPACE, value))

    async def resolve_target_column(self) -> str:
        top_level = await self._list_children(self.document_id)
        grids = [
            block
            for block in self._section_blocks(top_level)
            if block.get("block_type") == GRID_BLOCK_TYPE
        ]
        if not grids:
            raise CatalogDocumentError(
                "locate catalog grid",
                status_code=None,
                code=None,
                message="the catalog section does not contain a grid",
            )

        grid_id = grids[-1]["block_id"]
        columns = [
            block
            for block in await self._list_children(grid_id)
            if block.get("block_type") == GRID_COLUMN_BLOCK_TYPE
        ]
        if len(columns) != 4:
            raise CatalogDocumentError(
                "locate catalog columns",
                status_code=None,
                code=None,
                message=f"expected 4 columns in grid {grid_id}, found {len(columns)}",
            )

        counts = await asyncio.gather(
            *(self._child_count(column["block_id"]) for column in columns)
        )
        target_index = min(range(len(columns)), key=lambda index: (counts[index], index))
        return str(columns[target_index]["block_id"])

    async def list_section_image_tokens(self) -> list[str]:
        """按文档顺序列出图鉴章节内所有图片的素材 token，包括人工放进去的图，用于追加前去重。"""
        blocks = await self._list_paginated(
            "list document blocks",
            "/open-apis/docx/v1/documents/:document_id/blocks",
            paths={"document_id": self.document_id},
        )
        blocks_by_id = {block.get("block_id"): block for block in blocks}
        root = blocks_by_id.get(self.document_id) or {}
        top_level = [
            blocks_by_id[block_id]
            for block_id in root.get("children") or []
            if block_id in blocks_by_id
        ]
        section_ids = {block["block_id"] for block in self._section_blocks(top_level)}

        tokens: list[str] = []
        for block in blocks:
            token = (block.get("image") or {}).get("token")
            if block.get("block_type") != IMAGE_BLOCK_TYPE or not token:
                continue
            # 图片一般挂在 grid_column → grid → 顶层块下面，
            # 顺着 parent 往上找到顶层块，再判断它是否在图鉴章节内。
            ancestor_id = block.get("parent_id")
            while ancestor_id and ancestor_id not in section_ids:
                ancestor_id = (blocks_by_id.get(ancestor_id) or {}).get("parent_id")
            if ancestor_id:
                tokens.append(str(token))
        return tokens

    async def download_media(self, file_token: str) -> bytes:
        response = await self.client.arequest(
            self._request(
                HttpMethod.GET,
                "/open-apis/drive/v1/medias/:file_token/download",
                paths={"file_token": file_token},
            )
        )
        raw = response.raw
        content_type = next(
            (
                value
                for key, value in (raw.headers if raw else {}).items()
                if key.lower() == "content-type"
            ),
            "",
        )
        # 成功时返回的是图片二进制；失败时是 JSON 错误体，交给 _response_data 抛出具体错误码。
        if (
            raw
            and raw.status_code is not None
            and 200 <= raw.status_code < 300
            and not content_type.startswith("application/json")
            and raw.content
        ):
            return raw.content
        self._response_data("download catalog image", response)
        raise CatalogDocumentError(
            "download catalog image",
            status_code=raw.status_code if raw else None,
            code=None,
            message="response did not contain image bytes",
        )

    def _section_blocks(self, top_level: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """返回图鉴章节标题之后、下一个一级标题之前的顶层块。"""
        section_index = next(
            (
                index
                for index, block in enumerate(top_level)
                if block.get("block_id") == self.section_block_id
            ),
            None,
        )
        if section_index is None:
            raise CatalogDocumentError(
                "locate catalog section",
                status_code=None,
                code=None,
                message=f"section block {self.section_block_id} was not found",
            )

        section_end = next(
            (
                index
                for index in range(section_index + 1, len(top_level))
                if top_level[index].get("block_type") == HEADING_1_BLOCK_TYPE
            ),
            len(top_level),
        )
        return top_level[section_index + 1 : section_end]

    async def create_image_block(
        self,
        *,
        parent_block_id: str,
        client_token: str,
    ) -> str:
        data = await self._json_request(
            "create image block",
            HttpMethod.POST,
            "/open-apis/docx/v1/documents/:document_id/blocks/:block_id/children",
            paths={
                "document_id": self.document_id,
                "block_id": parent_block_id,
            },
            queries={
                "document_revision_id": -1,
                "client_token": client_token,
            },
            body={
                "index": -1,
                "children": [
                    {
                        "block_type": IMAGE_BLOCK_TYPE,
                        "image": {},
                    }
                ],
            },
            document_edit=True,
        )
        children = data.get("children") or []
        if len(children) != 1 or not children[0].get("block_id"):
            raise CatalogDocumentError(
                "create image block",
                status_code=None,
                code=None,
                message="response did not contain the created image block",
            )
        return str(children[0]["block_id"])

    async def upload_image(self, image_path: Path, *, image_block_id: str) -> str:
        response = await asyncio.to_thread(
            self._upload_image_sync,
            image_path,
            image_block_id,
        )
        data = self._response_data("upload catalog image", response)
        file_token = data.get("file_token")
        if not file_token:
            raise CatalogDocumentError(
                "upload catalog image",
                status_code=None,
                code=None,
                message="response did not contain a file token",
            )
        return str(file_token)

    async def replace_image(
        self,
        *,
        image_block_id: str,
        file_token: str,
        client_token: str,
    ) -> None:
        await self._json_request(
            "replace catalog image",
            HttpMethod.PATCH,
            "/open-apis/docx/v1/documents/:document_id/blocks/:block_id",
            paths={
                "document_id": self.document_id,
                "block_id": image_block_id,
            },
            queries={
                "document_revision_id": -1,
                "client_token": client_token,
            },
            body={"replace_image": {"token": file_token}},
            document_edit=True,
        )

    async def _child_count(self, block_id: str) -> int:
        return len(await self._list_children(block_id))

    async def _list_children(self, block_id: str) -> list[dict[str, Any]]:
        return await self._list_paginated(
            "list document children",
            "/open-apis/docx/v1/documents/:document_id/blocks/:block_id/children",
            paths={
                "document_id": self.document_id,
                "block_id": block_id,
            },
        )

    async def _list_paginated(
        self,
        operation: str,
        uri: str,
        *,
        paths: dict[str, str],
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            queries: dict[str, Any] = {
                "document_revision_id": -1,
                "page_size": 500,
            }
            if page_token:
                queries["page_token"] = page_token
            data = await self._json_request(
                operation,
                HttpMethod.GET,
                uri,
                paths=paths,
                queries=queries,
            )
            items.extend(data.get("items") or [])
            if not data.get("has_more"):
                return items
            page_token = data.get("page_token")
            if not page_token:
                raise CatalogDocumentError(
                    operation,
                    status_code=None,
                    code=None,
                    message="paginated response did not contain a page token",
                )

    async def _json_request(
        self,
        operation: str,
        method: HttpMethod,
        uri: str,
        *,
        paths: dict[str, str] | None = None,
        queries: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        document_edit: bool = False,
    ) -> dict[str, Any]:
        request = self._request(method, uri, paths=paths, queries=queries, body=body)
        if document_edit:
            async with self._edit_lock:
                elapsed = time.monotonic() - self._last_edit_at
                if elapsed < DOCUMENT_EDIT_INTERVAL_SECONDS:
                    await asyncio.sleep(DOCUMENT_EDIT_INTERVAL_SECONDS - elapsed)
                response = await self.client.arequest(request)
                self._last_edit_at = time.monotonic()
        else:
            response = await self.client.arequest(request)
        return self._response_data(operation, response)

    def _upload_image_sync(self, image_path: Path, image_block_id: str):
        with image_path.open("rb") as image_file:
            encoder = MultipartEncoder(
                fields={
                    "file_name": image_path.name,
                    "parent_type": "docx_image",
                    "parent_node": image_block_id,
                    "size": str(image_path.stat().st_size),
                    "file": (
                        image_path.name,
                        image_file,
                        "application/octet-stream",
                    ),
                }
            )
            request = self._request(
                HttpMethod.POST,
                "/open-apis/drive/v1/medias/upload_all",
                body=encoder,
            )
            request.headers["Content-Type"] = encoder.content_type
            return self.client.request(request)

    @staticmethod
    def _request(
        method: HttpMethod,
        uri: str,
        *,
        paths: dict[str, str] | None = None,
        queries: dict[str, Any] | None = None,
        body: Any = None,
    ) -> BaseRequest:
        request = BaseRequest()
        request.http_method = method
        request.uri = uri
        request.token_types = {AccessTokenType.TENANT}
        request.paths = paths or {}
        for key, value in (queries or {}).items():
            request.add_query(key, value)
        request.body = body
        return request

    @staticmethod
    def _response_data(operation: str, response) -> dict[str, Any]:
        status_code = response.raw.status_code if response.raw else None
        try:
            payload = json.loads(response.raw.content.decode("utf-8"))
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CatalogDocumentError(
                operation,
                status_code=status_code,
                code=None,
                message="response was not valid JSON",
            ) from error

        code = payload.get("code")
        if status_code is None or not 200 <= status_code < 300 or code != 0:
            raise CatalogDocumentError(
                operation,
                status_code=status_code,
                code=code,
                message=str(payload.get("msg") or "unknown error"),
            )
        return payload.get("data") or {}
