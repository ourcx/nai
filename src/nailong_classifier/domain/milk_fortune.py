from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from decimal import Decimal
from math import prod

from .milk_fortune_memes import MEME_HEADLINES, MEME_SCENES, MEME_TITLES, THEME_SETS

NAILONG_SETS = THEME_SETS["奶龙"]
NAIWA_SETS = THEME_SETS["奶蛙"]
FORTUNE_SETS = {name: faces for sets in THEME_SETS.values() for name, faces in sets.items()}
FORTUNE_FACES = tuple(
    (set_name, face) for set_name, faces in FORTUNE_SETS.items() for face in faces
)
FORTUNE_CHARACTERS = tuple(THEME_SETS)
FORTUNE_TEXT_FIELDS = ("keyword", "good", "avoid", "message", "item", "event", "mood")
FORTUNE_POOLS = {
    character: {
        # 按整条文案去重，词库维护时不会因重复条目悄悄改变抽取权重。
        field: tuple(dict.fromkeys(getattr(scene, field) for scene in scenes))
        for field in FORTUNE_TEXT_FIELDS
    }
    for character, scenes in MEME_SCENES.items()
}
THEME_FACES = {
    character: tuple((name, face) for name, faces in sets.items() for face in faces)
    for character, sets in THEME_SETS.items()
}


def fortune_text_combination_count() -> int:
    """任意固定分数档的文案组合下界；不把分数、身份、日期或隐藏字段当成变化凑数。"""
    return sum(
        prod((
            len(THEME_FACES[character]),
            min(map(len, MEME_TITLES[character])),
            *(len(pool) for pool in FORTUNE_POOLS[character].values()),
        ))
        for character in FORTUNE_CHARACTERS
    )


# 保留旧词库常量供旧调用方引用；新的抽签统一使用上面的四主题完整梗库。
MILK_FORTUNE_GOOD = (
    "主动接住群友的梗", "发一张压箱底奶图", "夸群友一句真心话",
    "猜图时相信第一直觉", "忙里偷闲摸鱼五分钟", "给今天定个小目标",
    "先吃饭再想难题", "把拖了很久的小事做掉", "出门走一小圈",
    "给自己留一口喜欢的零食", "和老朋友打个招呼", "笑够了再开工",
    "早点收工回家", "把有趣的东西分享出来", "喝完这杯水", "坦然说一句不懂",
    "把鸡腿留给最饿的奶友", "给今天存一张黄乎乎的奶图",
    "和奶蛙交换一张表情包", "先吃饱再去拯救群聊",
    "笑到一半记得喘口气", "带着鸡腿和群友和解",
    "把压箱底的奶龙发出来", "晒晒太阳再回消息",
)
MILK_FORTUNE_AVOID = (
    "把奶蛙认成奶龙", "连续刷屏十张表情", "空腹参加长会议",
    "把群聊聊成工单群", "为了最后一句争输赢", "潜水到大家以为你掉线",
    "吃饭时还盯着工作消息", "替明天提前焦虑", "拿别人的进度催自己",
    "困了还硬撑", "对着未读消息发呆", "把玩笑当成考试",
    "一口气答应所有事情", "等完美时机才开始", "忘记自己也要休息", "抢最后一根鸡腿",
    "嘴里有鸡腿还急着接梗", "对着空碗等奶龙变出饭",
    "拿奶蛙的荷叶擦嘴", "吃完鸡腿忘记说好吃",
    "为了一张奶图熬到太晚", "把自己的饭点让给会议",
    "空着肚子当群聊气氛组", "盯着别人碗里的鸡腿",
)
MILK_FORTUNE_KEYWORDS = (
    "接梗", "松弛", "好奇", "开怀", "摸鱼", "行动", "分享", "好运",
    "开饭", "散步", "直觉", "留白", "重逢", "轻松", "勇气", "收工",
)
MILK_FORTUNE_MESSAGES = (
    "今天的快乐藏在第一句主动开口里。",
    "别急着证明自己，先把群里的气氛接住。",
    "适合做一点不费劲、但能让人开心的小事。",
    "你的好运来得很轻，可能就是一个刚好的表情包。",
    "先完成最小的一步，剩下的奶气会自己跟上。",
    "今天很适合把严肃的话说得稍微可爱一点。",
    "鸡腿凉了可以再热，今天没做完的事也可以明天再说。",
    "不用每句话都接住，笑一下也算参与。",
    "碰到小麻烦先吃口饭，奶龙吃饱了才有主意。",
    "把今天过得舒服一点，就已经很不错了。",
    "有想分享的笑话就发出来，说不定有人正等这一口快乐。",
    "别把一时的不顺当成全天的结局，下一张奶图还没来。",
    "奶龙说，天大的事也得等这根鸡腿吃完。",
    "今天不一定赢过所有人，但可以多吃一口喜欢的。",
    "群里没梗的时候，奶龙也会安静嚼鸡腿。",
    "快乐不必一口气吃完，留一点给下班后的自己。",
    "奶龙已经替你把今天的烦恼放到碗底了。",
    "这根鸡腿先给你，剩下的事慢慢说。",
    "累了就靠一会儿，奶龙的肚子暂时可以借你。",
    "别怕笑得太大声，奶龙比你笑得还响。",
    "今天适合把收藏夹里最好笑的那张奶图翻出来。",
    "出场也可以很简单，带根鸡腿来就挺好。",
    "奶龙正在等你开饭，别让自己饿着。",
    "奶图可以慢慢收，今天这口饭要趁热吃。",
)
LUCKY_ITEMS = (
    "一根热乎鸡腿", "黄色水杯", "刚洗好的睡衣", "口袋里的薄荷糖",
    "压箱底表情包", "一杯温牛奶", "绿色小挂件", "软乎乎的抱枕",
    "常用的那副耳机", "冰箱里最后一盒酸奶", "没写完的小纸条", "晒到太阳的椅子",
    "奶龙小挂件", "黄色饭盒", "刚拆开的鸡腿包装", "备用的小勺子",
    "兜里剩下的一颗奶糖", "柔软的黄色拖鞋", "热饭时用的小盘子", "一张笑场奶龙图",
)
DAILY_EVENTS = (
    "你发的表情包可能会被群友顺走。", "会有人接住你随口说出的梗。",
    "可能找到一张忘了存在哪里的奶图。", "午饭可能比预想的好吃。",
    "一件拖着的小事可能突然有了办法。", "有人可能会问你这张奶图是哪来的。",
    "可能赶上刚好不用排队的那一刻。", "摸鱼时可能撞见同样在摸鱼的朋友。",
    "一条旧消息可能让你笑出声。", "可能突然很想吃鸡腿。",
    "可能听到一首刚好合心情的歌。", "随手分享的小事可能引出一场闲聊。",
    "奶龙可能隔着屏幕问你吃饭了没。", "可能在群里遇见另一位鸡腿爱好者。",
    "你存的奶龙图可能刚好派上用场。", "可能听见一句让你想立刻发奶图的话。",
    "有人可能把最后一口好吃的留给你。", "今天的饭搭子可能比平时更好找。",
    "奶龙可能把你的笑点又往下拽了一点。", "可能因为一张黄乎乎的图笑出声。",
)
NAIWA_GOOD = (
    "蹲上荷叶歇一会儿", "发一张绿乎乎的奶蛙图", "给群友捧个呱场",
    "跳过一件没必要争的小事", "和奶龙交换今天的零食", "听两分钟窗外的雨",
    "遇到好笑的事就呱出来", "把今天的小进展说给朋友听", "带伞再出门",
    "在群里冒一个友善的泡", "吃饱了再练蛙跳", "给发呆留一小块荷叶",
    "夸一句群友的表情包", "忙完这件事就蹦去吃饭", "喝口水润润呱",
    "试着接住奶龙没讲完的梗", "从收藏夹捞一张旧奶蛙", "走路时看看路边的绿色",
    "把不好意思说的谢谢说出来", "困了就收呱下班", "把伞挪一点给朋友",
    "让自己慢半拍再回话", "给好消息配一张蹦跶奶蛙", "找到自己的那片荷叶",
)
NAIWA_AVOID = (
    "把荷叶当成加班工位", "没看落点就往前蹦", "一口气呱到群友没法插话",
    "把奶龙的鸡腿藏进池塘", "下雨不带伞还嘴硬", "把潜水当成失联",
    "为了一句梗跳进争论", "在空荷叶上等奶龙开饭", "把一句玩笑呱成辩论赛",
    "拿别人的蛙跳进度催自己", "抢群友刚找到的阴凉", "憋着笑把自己憋成绿脸",
    "对着未读消息连呱三遍", "让小水花变成大风浪", "蹲到腿麻还说不累",
    "把每一次沉默都当成冷场", "困了还撑着当呱声担当", "顺手忘了带回自己的伞",
    "给自己安排一整天连跳", "用奶蛙图刷屏代替一句谢谢", "把饭点让给群聊",
    "为了最后一声呱争输赢", "揣着烦恼一起跳进被窝", "总等别人先冒泡",
)
NAIWA_KEYWORDS = (
    "冒泡", "蹦跶", "荷叶", "捧场", "慢呱", "轻跳", "听雨", "放晴",
    "绿意", "合拍", "歇脚", "开呱", "小水花", "同行", "回塘", "笑场",
)
NAIWA_MESSAGES = (
    "奶蛙给你留了片荷叶，累了就过来蹲一会儿。",
    "今天不用跳得最远，稳稳落在自己的位置就好。",
    "想说的就呱一句，总有人愿意听。",
    "奶蛙也有不想蹦跶的时候，歇着不丢蛙。",
    "小水花也挺好看，不必每次都掀起大浪。",
    "池塘今天不考核蛙跳，允许你原地发呆。",
    "奶龙负责吃鸡腿，奶蛙负责给你捧场。",
    "有空就冒个泡，没空就安心潜水。",
    "一声呱不一定解决问题，但可能让朋友知道你在。",
    "今天的荷叶够大，放得下你和你的小烦恼。",
    "下雨就听雨，放晴就晒背，奶蛙今天没别的要求。",
    "奶蛙替你把伞撑好了，慢一点出发也没关系。",
    "别拿自己和飞过去的鸟比，奶蛙有奶蛙的路。",
    "有人冷场时轻轻接一句，奶蛙给你记个捧场奖。",
    "蹦了一整天，也该轮到荷叶接住你了。",
    "今天不用当全塘最响的蛙，舒服地呱就行。",
    "听见好消息就跳一下，没听见就先吃饭。",
    "这张奶蛙签的意思是：你可以歇口气。",
    "奶蛙已经从水里冒头了，就差你来接一句梗。",
    "小小一跳也算往前，别替自己扣分。",
    "荷叶晃了也别急，先稳住脚下这一小块。",
    "奶蛙不催你回消息，只催你记得喝水。",
    "今天群里哪怕只笑了一次，也值得收一张奶蛙。",
    "最后一声呱留给晚安，今天到这里就很好。",
)
NAIWA_ITEMS = (
    "一片宽宽的荷叶", "绿色小水杯", "奶蛙钥匙扣", "收好的小雨伞",
    "绿乎乎的表情包", "荷叶形的小杯垫", "一颗薄荷糖", "窗边那盆绿植",
    "一碗热乎乎的饭", "刚擦干的小板凳", "雨后的绿色拖鞋", "装零食的小绿盒",
    "一张冒泡奶蛙图", "落在桌边的小纸条", "软软的青蛙抱枕", "晒暖了的小毛巾",
    "一副听雨的耳机", "手边没喝完的温水", "奶龙分享的半根鸡腿", "画着荷叶的便签",
)
NAIWA_EVENTS = (
    "你发的奶蛙图可能被群友捞走。", "可能有人接住你突然冒出来的一声呱。",
    "可能在群里遇到同样蹲着发呆的奶友。", "奶龙可能来问你荷叶上还有没有空位。",
    "可能找到一张很适合今天心情的绿脸奶蛙。", "一个小好消息可能让你想蹦一下。",
    "可能刚好赶上雨停的那一刻。", "群友可能拿一张奶龙图和你换奶蛙。",
    "你随手捧的一次场可能让别人开心一会儿。", "可能遇见一位愿意陪你慢慢走的人。",
    "奶蛙可能蹲在群聊角落等你冒泡。", "可能因为一张奶蛙图笑到忘了刚才的烦恼。",
    "一段旧聊天可能让你又想呱一声。", "今天的饭可能让奶蛙都停止蹦跶。",
    "有人可能听懂你没讲完的半句梗。", "可能收获一句刚好的谢谢。",
    "奶蛙可能提醒你把水喝完。", "可能有朋友愿意分你一点伞下的位置。",
    "你没留意的小进展可能被人夸一句。", "可能碰见一片很像荷叶的绿色。",
)
NAILONG_MOODS = (
    "抱着鸡腿等开饭", "笑到肚子一颤一颤", "黄乎乎地晒太阳", "吃饱了想发呆",
    "准备发一张压箱底奶图", "先摸五分钟鱼", "只想准点收工", "随时准备接梗",
    "今天胃口和心情都不错", "端着饭碗看群聊", "蹭到一点小好运", "给自己留一颗奶糖",
)
NAIWA_MOODS = (
    "蹲荷叶上看热闹", "随时准备呱一句", "绿着脸慢慢回消息", "小蹦一下就去吃饭",
    "趴在荷叶上省电", "听雨听到忘了时间", "冒个泡表示我在", "给奶龙的笑话捧场",
    "只想稳稳落地", "想发张奶蛙但还在挑", "把烦恼留在水面上", "蹦到下班就回塘",
)
NAILONG_TITLES = (
    ("天选奶龙", "鸡腿锦鲤", "奶气王者"),
    ("群聊锦鲤", "鸡腿幸运星", "奶龙团宠"),
    ("元气奶龙", "开饭先锋", "接梗奶龙"),
    ("稳稳奶友", "暖胃奶龙", "松弛奶龙"),
    ("慢热奶龙", "饭前奶龙", "省电奶龙"),
    ("潜水奶苗", "被窝奶龙", "歇脚奶龙"),
)
NAIWA_TITLES = (
    ("天选奶蛙", "荷塘锦鲤", "荷叶王者"),
    ("奶蛙团宠", "蹦跶幸运星", "全塘好呱"),
    ("元气奶蛙", "冒泡先锋", "接梗奶蛙"),
    ("稳稳奶蛙", "荷叶闲蛙", "松弛奶蛙"),
    ("慢热奶蛙", "轻跳奶蛙", "省电奶蛙"),
    ("潜水奶蛙", "听雨奶蛙", "歇脚奶蛙"),
)


@dataclass(frozen=True)
class MilkFortune:
    score: int
    banter: int
    title: str
    keyword: str
    good: str
    avoid: str
    message: str
    chicken_luck: int = 0
    lucky_item: str = ""
    daily_event: str = ""
    headline: str = ""
    face: str = ""
    set_name: str = ""
    # 缺字段的旧快照按原来的奶龙文案展示；姓名在回复时读取，换奶名不重抽日签。
    character: str = "奶龙"
    pond_luck: int = 0
    mood: str = ""
    theme_luck: int = 0

    @property
    def special_luck_label(self) -> str:
        return {
            "奶龙": "鸡腿运", "奶蛙": "荷塘运",
            "大肥鱼DeepSeek": "深潜运", "豆包": "嘴甜运",
        }.get(self.character, "鸡腿运")

    @property
    def special_luck(self) -> int:
        if self.character == "奶蛙":
            return self.pond_luck
        if self.character in {"大肥鱼DeepSeek", "豆包"}:
            return self.theme_luck
        return self.chicken_luck


@dataclass(frozen=True)
class FortuneEntry:
    day: str
    fortune: MilkFortune


@dataclass(frozen=True)
class FortuneCollection:
    total: int
    faces: frozenset[str]
    streak: int
    longest_streak: int
    entries: tuple[FortuneEntry, ...]
    page: int
    pages: int
    completed_sets: tuple[str, ...]


def make_milk_fortune(*, chat_id: str, sender_id: str, day: str) -> MilkFortune:
    """四主题等概率出场，主题内签面、称号与七项文案分别等概率抽取。"""
    digest = hashlib.sha256(f"{day}\0{chat_id}\0{sender_id}".encode()).digest()
    score = int(Decimal(int.from_bytes(digest[0:4], "big")) % Decimal(101))
    banter = int(Decimal(int.from_bytes(digest[4:8], "big")) % Decimal(101))
    rng = random.Random(digest)
    character = rng.choice(FORTUNE_CHARACTERS)
    set_name, face = rng.choice(THEME_FACES[character])
    grade = next(index for index, threshold in enumerate((95, 80, 60, 40, 20, 0))
                 if score >= threshold)
    title = rng.choice(MEME_TITLES[character][grade])
    selected = {field: rng.choice(pool) for field, pool in FORTUNE_POOLS[character].items()}
    return MilkFortune(
        score=score, banter=banter, title=title,
        headline=MEME_HEADLINES[character][grade], character=character,
        keyword=selected["keyword"], good=selected["good"], avoid=selected["avoid"],
        message=selected["message"], lucky_item=selected["item"],
        daily_event=selected["event"], mood=selected["mood"],
        chicken_luck=rng.randrange(101), pond_luck=rng.randrange(101),
        theme_luck=rng.randrange(101), face=face, set_name=set_name,
    )
