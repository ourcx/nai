from __future__ import annotations

from dataclasses import dataclass

# 稀有度按“这张图在群里被发过多少次”定级：发得越多越烂大街，发得少才稀有。
# seen_count 由每次观测累加，天然反映一张表情在群里的热度，比感知指纹“撞脸”靠谱得多。
# 阈值按群聊表情的实际分布拍定：发一两次是冷门，十几次往上就是刷屏常客。
RARITY_TIERS = (
    # (seen_count 下限, 名称, 星级, 说明)
    (16, "烂大街", 1, "天天刷屏"),
    (8, "常见", 2, "群里常客"),
    (4, "稀有", 3, "偶尔刷到"),
    (2, "史诗", 4, "难得一见"),
    (1, "传说", 5, "深藏不露"),
)


@dataclass(frozen=True)
class Rarity:
    """一张图鉴图的稀有度评级。seen_count 为这张图在群里累计被发送的次数。"""

    name: str
    stars: int
    seen_count: int
    note: str

    @property
    def stars_text(self) -> str:
        return "⭐" * self.stars


def classify_rarity(seen_count: int) -> Rarity:
    """发送次数越少越稀有：只冒过一两次是传说，天天刷屏就是烂大街。"""
    if seen_count < 1:
        raise ValueError("seen_count must be at least 1")
    for threshold, name, stars, note in RARITY_TIERS:
        if seen_count >= threshold:
            return Rarity(name=name, stars=stars, seen_count=seen_count, note=note)
    # 阈值表最低一档是 1，正常不会落到这里；兜底按传说处理。
    return Rarity(name="传说", stars=5, seen_count=seen_count, note="深藏不露")
