from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from nailong_classifier.catalog_document import CatalogDocumentClient


class FakeClient:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.requests = []

    async def arequest(self, request):
        self.requests.append(request)
        payload = self.responses.pop(0)
        return SimpleNamespace(
            raw=SimpleNamespace(
                status_code=200,
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
