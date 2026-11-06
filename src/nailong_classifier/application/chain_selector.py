from __future__ import annotations

import asyncio
import logging
import secrets

import numpy as np

from ..classification.chain_features import DinoEncoder
from ..imaging.image_fingerprint import (
    DUPLICATE_DISTANCE,
    hash_distance,
    try_file_content_hash,
)
from ..persistence.storage import AssetRecord, BotStore, ChainCandidate

LOGGER = logging.getLogger(__name__)


class ChainSelector:
    """程序决定分流，DINO 决定视觉相近程度；所有分支共用防重复过滤。"""

    def __init__(
        self, store: BotStore, *, weights: tuple[int, ...] = (20, 40, 30),
        top_k: int = 5, min_similarity: float = 0.55,
    ) -> None:
        if len(weights) != 3 or any(weight < 0 for weight in weights) or not any(weights):
            raise ValueError("chain weights require three nonnegative integers with a positive total")
        if top_k < 1 or not 0 <= min_similarity <= 1:
            raise ValueError("chain top-k must be positive; similarity must be between 0 and 1")
        self.store = store
        self.weights = weights
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.encoder = DinoEncoder()
        self.index_event = asyncio.Event()

    @staticmethod
    def _fingerprint(candidate: ChainCandidate) -> str | None:
        return candidate.fingerprint or try_file_content_hash(candidate.asset.path)

    def select(self, source: AssetRecord, chat_id: str) -> AssetRecord | None:
        candidates = self.store.chain_candidates(source.label, self.encoder.model_id)
        source_candidate = next((item for item in candidates if item.asset.id == source.id), None)
        source_hash = (
            self._fingerprint(source_candidate) if source_candidate
            else try_file_content_hash(source.path)
        )
        if not source_hash:
            return None
        recent = self.store.recent_chain_assets(chat_id)
        excluded_shas = {source.sha256, *(item.asset.sha256 for item in recent)}
        excluded_hashes = [source_hash]
        excluded_hashes.extend(value for item in recent if (value := self._fingerprint(item)))
        eligible = []
        for item in candidates:
            if item.asset.sha256 in excluded_shas or not item.asset.path.is_file():
                continue
            fingerprint = self._fingerprint(item)
            if not fingerprint or any(
                hash_distance(fingerprint, known) <= DUPLICATE_DISTANCE
                for known in excluded_hashes
            ):
                continue
            eligible.append(item)
        if not eligible:
            return None

        branch = secrets.SystemRandom().choices(
            ("similar", "weird", "random"), weights=self.weights, k=1,
        )[0]
        pool = []
        if branch == "weird":
            pool = [item for item in eligible if item.weird]
        elif branch == "similar":
            try:
                pool = self._similar_candidates(source_candidate, eligible)
            except Exception:
                # 模型下载或单图解析失败时不影响正常接龙。
                LOGGER.exception("Similar chain failed; using random source_asset=%d", source.id)
        if not pool:
            pool = eligible
            if branch != "random":
                LOGGER.info("Chain %s pool unavailable; using random", branch)
                branch = "random"
        chosen = secrets.choice(pool).asset
        LOGGER.info(
            "Chain route=%s source_asset=%d reply_asset=%d candidates=%d",
            branch, source.id, chosen.id, len(pool),
        )
        return chosen

    def _similar_candidates(
        self, source: ChainCandidate | None, eligible: list[ChainCandidate],
    ) -> list[ChainCandidate]:
        indexed = [item for item in eligible if item.feature]
        if source is None or not indexed:
            return []
        if source.feature:
            query = np.frombuffer(source.feature, dtype=np.float32)
        elif self.encoder.ready:
            query = self.encoder.encode(source.asset.path)
            self.store.save_chain_feature(source.asset.id, self.encoder.model_id, query.tobytes())
        else:
            return []
        compatible = [
            item for item in indexed
            if np.frombuffer(item.feature, dtype=np.float32).shape == query.shape
        ]
        if not compatible:
            return []
        vectors = np.stack([np.frombuffer(item.feature, dtype=np.float32) for item in compatible])
        scores = np.dot(vectors, query)
        # 排除几乎完全一致的特征，避免 dHash 漏过的裁剪/轻微改字版本。
        ranked = np.argsort(scores)[::-1]
        matches = [
            compatible[int(index)] for index in ranked
            if self.min_similarity <= scores[index] < 0.995
        ]
        return matches[:self.top_k]

    async def build_index(self) -> None:
        """首次后台下载权重并分批落库；中断可续算，新收录图片自动补进索引。"""
        failed: set[int] = set()
        LOGGER.info(
            "Chain selector weights(similar,weird,random)=%s top_k=%d min_similarity=%.3f model=%s",
            self.weights, self.top_k, self.min_similarity, self.encoder.model_id,
        )
        while True:
            self.index_event.clear()
            pending = await asyncio.to_thread(
                self.store.unindexed_chain_assets, self.encoder.model_id,
            )
            batch = [asset for asset in pending if asset.id not in failed][:32]
            for asset in batch:
                try:
                    feature = await asyncio.to_thread(self.encoder.encode, asset.path)
                    await asyncio.to_thread(
                        self.store.save_chain_feature,
                        asset.id, self.encoder.model_id, feature.tobytes(),
                    )
                except Exception:
                    LOGGER.exception("Chain index failed asset_id=%d", asset.id)
                    if not self.encoder.ready:
                        # 下载失败最多每十分钟重试一次；不逐图重试网络请求。
                        await asyncio.sleep(600)
                        break
                    failed.add(asset.id)
            else:
                if batch:
                    LOGGER.info("Chain index processed batch=%d", len(batch))
                    continue
                try:
                    await asyncio.wait_for(self.index_event.wait(), timeout=60)
                except TimeoutError:
                    pass
