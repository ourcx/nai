from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from nailong_classifier.catalog_document import CatalogDocumentClient, CatalogDocumentError


class FakeClient:
    def __init__(self, responses: list[dict | bytes]) -> None:
        self.responses = list(responses)
        self.requests = []

    async def arequest(self, request):
        self.requests.append(request)
        payload = self.responses.pop(0)
        if isinstance(payload, bytes):
            return SimpleNamespace(
                raw=SimpleNamespace(
                    status_code=200,
                    headers={"Content-Type": "image/png"},
                    content=payload,
                )
            )
        return SimpleNamespace(
            raw=SimpleNamespace(
                status_code=200 if payload.get("code") == 0 else 400,
                headers={"Content-Type": "application/json; charset=utf-8"},
                content=json.dumps(payload).encode("utf-8"),
            )
        )


def success(data: dict) -> dict:
    return {"code": 0, "msg": "success", "data": data}


def test_resolve_target_column_uses_last_grid_and_shortest_column() -> None:
    client = FakeClient(
        [
            success(
                {
                    "items": [
                        {"block_id": "before", "block_type": 2},
                        {"block_id": "section", "block_type": 3},
                        {"block_id": "old-grid", "block_type": 24},
                        {"block_id": "last-grid", "block_type": 24},
                        {"block_id": "next-section", "block_type": 3},
                    ],
                    "has_more": False,
                }
            ),
            success(
                {
                    "items": [
                        {"block_id": "column-1", "block_type": 25},
                        {"block_id": "column-2", "block_type": 25},
                        {"block_id": "column-3", "block_type": 25},
                        {"block_id": "column-4", "block_type": 25},
                    ],
                    "has_more": False,
                }
            ),
            success({"items": [{}, {}], "has_more": False}),
            success({"items": [{}], "has_more": False}),
            success({"items": [{}, {}], "has_more": False}),
            success({"items": [{}], "has_more": False}),
        ]
    )
    document = CatalogDocumentClient(
        client,
        document_id="document",
        section_block_id="section",
    )

    target = asyncio.run(document.resolve_target_column())

    assert target == "column-2"
    assert client.requests[1].paths["block_id"] == "last-grid"


def test_create_image_block_uses_stable_client_token() -> None:
    client = FakeClient(
        [
            success(
                {
                    "children": [
                        {
                            "block_id": "image-block",
                            "block_type": 27,
                        }
                    ]
                }
            )
        ]
    )
    document = CatalogDocumentClient(
        client,
        document_id="document",
        section_block_id="section",
    )
    client_token = document.client_token("asset-sha256", "create-image-block")

    block_id = asyncio.run(
        document.create_image_block(
            parent_block_id="column-1",
            client_token=client_token,
        )
    )

    assert block_id == "image-block"
    request = client.requests[0]
    assert request.paths == {
        "document_id": "document",
        "block_id": "column-1",
    }
    assert ("client_token", client_token) in request.queries
    assert request.body == {
        "index": -1,
        "children": [{"block_type": 27, "image": {}}],
    }
    assert client_token == document.client_token(
        "asset-sha256",
        "create-image-block",
    )
    assert client_token != document.client_token("other-sha256", "create-image-block")


def test_list_section_image_tokens_only_returns_images_inside_section() -> None:
    client = FakeClient(
        [
            success(
                {
                    "items": [
                        {
                            "block_id": "document",
                            "block_type": 1,
                            "children": ["before", "section", "grid", "next-section"],
                        },
                        {"block_id": "before", "block_type": 2, "parent_id": "document"},
                        {"block_id": "section", "block_type": 3, "parent_id": "document"},
                        {"block_id": "grid", "block_type": 24, "parent_id": "document"},
                    ],
                    "has_more": True,
                    "page_token": "page-2",
                }
            ),
            success(
                {
                    "items": [
                        {"block_id": "column", "block_type": 25, "parent_id": "grid"},
                        {
                            "block_id": "inside",
                            "block_type": 27,
                            "parent_id": "column",
                            "image": {"token": "token-inside"},
                        },
                        {
                            "block_id": "empty",
                            "block_type": 27,
                            "parent_id": "column",
                            "image": {},
                        },
                        {"block_id": "next-section", "block_type": 3, "parent_id": "document"},
                        {
                            "block_id": "outside",
                            "block_type": 27,
                            "parent_id": "document",
                            "image": {"token": "token-outside"},
                        },
                    ],
                    "has_more": False,
                }
            ),
        ]
    )
    document = CatalogDocumentClient(
        client,
        document_id="document",
        section_block_id="section",
    )

    tokens = asyncio.run(document.list_section_image_tokens())

    assert tokens == ["token-inside"]
    assert ("page_token", "page-2") in client.requests[1].queries


def test_download_media_returns_bytes_and_raises_on_json_error() -> None:
    client = FakeClient(
        [
            b"\x89PNG-bytes",
            {"code": 99991672, "msg": "Access denied", "data": {}},
        ]
    )
    document = CatalogDocumentClient(
        client,
        document_id="document",
        section_block_id="section",
    )

    assert asyncio.run(document.download_media("file-a")) == b"\x89PNG-bytes"
    assert client.requests[0].paths == {"file_token": "file-a"}
    try:
        asyncio.run(document.download_media("file-b"))
    except CatalogDocumentError as error:
        assert error.code == 99991672
    else:
        raise AssertionError("download_media should raise on JSON error")
