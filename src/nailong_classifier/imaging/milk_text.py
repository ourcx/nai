"""奶系文字贴纸。机器人和网页共用本地渲染，随机样式与动效均不依赖在线生图。"""

from __future__ import annotations

import io
import re
import secrets
import threading
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

ASSET_DIR = Path(__file__).with_name("assets")
FONT_PATH = ASSET_DIR / "ZCOOLKuaiLe-Regular.ttf"
MAX_TEXT_LENGTH = 60
MAX_LINES = 4
PALETTES = {
    "gold": ("#FFE76D", "#FFB51B"),
    "cream": ("#FFF3B8", "#FFCC43"),
    "orange": ("#FFD23C", "#F28B18"),
    "white": ("#FFFFFF", "#FFFFFF"),
}
AVATAR_COUNT = 8
AVATARS = {
    label: tuple(
        ASSET_DIR / (f"{label}-avatar.png" if index == 1 else f"{label}-avatar-{index}.png")
        for index in range(1, AVATAR_COUNT + 1)
    )
    for label in ("nailong", "naiwa")
}
STATIC_ASSETS = {
    "nailong-avatar.png": ASSET_DIR / "nailong-avatar.png",
    "naiwa-avatar.png": ASSET_DIR / "naiwa-avatar.png",
    "milk-font.ttf": FONT_PATH,
    **{path.name: path for paths in AVATARS.values() for path in paths},
}
_RENDER_LOCK = threading.Lock()
# 每个维度只列真实选项一次；“随机”是抽取方式，不额外占一份概率。
STYLE_CHOICES = {
    "character": ("nailong", "naiwa", "both", "none"),
    "palette": tuple(PALETTES),
    "layout": ("jitter", "rise", "straight", "stack"),
    "background": ("transparent", "cream"),
    "motion": ("bounce", "sway", "wave", "static"),
}
MILK_TEXT_USAGE = (
    "奶峰生成器 鸡腿拿来！：不加参数时全部随机，各选项等概率，可连续重复。\n"
    "可加 --角色 随机/奶龙/奶蛙/双奶/纯文字 --头像 随机/1-8 "
    "--配色 随机/奶龙黄/奶油黄/鸡腿橙/黄边白字 "
    "--排版 随机/错位/跳字/平排/叠字 --动作 随机/跳动/摇摆/波浪/静态 "
    "--背景 随机/透明/奶油。\n"
    "微调：--字号 48-140 --描边 2-14 --倾斜 -12到12 "
    "--幅度 0-24 --速度 60-300（毫秒）。未填写时在范围内均匀随机；字号 0 自动适配。\n"
    "例如：奶峰生成器 鸡腿拿来！ --角色 奶蛙 --配色 黄边白字 --动作 波浪。\n"
    "最多 60 字、四行；飞书和网页可出 GIF，QQ 出静态 PNG。"
)
_PARAMETER_FIELDS = {
    "角色": "character", "character": "character",
    "头像": "avatar", "avatar": "avatar",
    "配色": "palette", "palette": "palette",
    "排版": "layout", "layout": "layout",
    "动作": "motion", "motion": "motion",
    "背景": "background", "background": "background",
    "字号": "font_size", "size": "font_size",
    "描边": "stroke", "stroke": "stroke",
    "倾斜": "tilt", "tilt": "tilt",
    "幅度": "amplitude", "amplitude": "amplitude",
    "速度": "duration_ms", "speed": "duration_ms",
}
_PARAMETER_VALUES = {
    "character": {"奶龙": "nailong", "奶蛙": "naiwa", "双奶": "both", "纯文字": "none"},
    "palette": {"奶龙黄": "gold", "奶油黄": "cream", "鸡腿橙": "orange",
                "黄边白字": "white", "黄边白色": "white", "白字": "white"},
    "layout": {"错位": "jitter", "跳字": "rise", "平排": "straight", "叠字": "stack"},
    "motion": {"跳动": "bounce", "摇摆": "sway", "波浪": "wave", "静态": "static"},
    "background": {"透明": "transparent", "奶油": "cream"},
}
_NUMERIC_LIMITS = {
    "font_size": (0, 140, "字号"), "stroke": (2, 14, "描边"),
    "tilt": (-12, 12, "倾斜"), "amplitude": (0, 24, "幅度"),
    "duration_ms": (60, 300, "速度"),
}


@dataclass(frozen=True)
class MilkTextOptions:
    text: str
    character: str = "random"
    palette: str = "random"
    layout: str = "random"
    background: str = "random"
    avatar: str = "random"
    motion: str = "random"
    # None 表示未指定，不能用 0 代替：0 是字号自动适配、零倾斜和零幅度的有效值。
    font_size: int | None = None
    stroke: int | None = None
    tilt: int | None = None
    amplitude: int | None = None
    duration_ms: int | None = None


def validate_options(options: MilkTextOptions) -> str:
    """限制输入和画布大小，并保留用户主动换行；失败时给出可以直接展示的中文提示。"""
    if not isinstance(options.text, str):
        raise ValueError("请填写文字。")
    text = options.text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise ValueError("先写一句，再做成奶字。")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"文字最多 {MAX_TEXT_LENGTH} 字。")
    if any(not line.strip() for line in text.split("\n")):
        raise ValueError("换行之间请写上文字。")
    if len(text.split("\n")) > MAX_LINES:
        raise ValueError(f"最多写 {MAX_LINES} 行。")
    if re.search(r"[\x00-\x08\x0b-\x1f\x7f]", text):
        raise ValueError("文字含有无法显示的控制字符。")
    if options.character not in {*STYLE_CHOICES["character"], "random"}:
        raise ValueError("请选择奶龙、奶蛙、双奶或纯文字。")
    if options.palette not in {*PALETTES, "random"}:
        raise ValueError("请选择现有奶系配色。")
    if options.layout not in {*STYLE_CHOICES["layout"], "random"}:
        raise ValueError("请选择现有文字排版。")
    if options.background not in {*STYLE_CHOICES["background"], "random"}:
        raise ValueError("请选择透明底或奶油底。")
    if options.avatar not in {"random", *(str(index) for index in range(1, AVATAR_COUNT + 1))}:
        raise ValueError(f"头像请选随机或 1-{AVATAR_COUNT}。")
    if options.motion not in {*STYLE_CHOICES["motion"], "random"}:
        raise ValueError("动作请选随机、跳动、摇摆、波浪或静态。")
    for field, (lower, upper, label) in _NUMERIC_LIMITS.items():
        value = getattr(options, field)
        if value is None:
            continue
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"{label}请填写 {lower} 到 {upper} 的整数。")
    if options.font_size and options.font_size < 48:
        raise ValueError("字号请填写 48 到 140 的整数，或用 0 自动适配。")
    return text


def parse_milk_text_command(text: str) -> MilkTextOptions | None:
    """只识别明确的制作命令，普通群聊里的“奶龙字”不被当作指令词截断。"""
    match = re.fullmatch(
        r"(奶峰生成器|奶龙字|奶蛙字|奶字)(?:[\s：:]+(.*))?", text.strip(), re.DOTALL
    )
    if not match:
        return None
    body = match.group(2) or ""
    # -- 后面的内容作为原样文字；参数必须显式带 --，不吞掉正文中的角色或颜色名称。
    command, separator, literal = body.partition(" -- ")
    values: dict[str, str | int] = {}
    # 角色别名相当于显式指定角色；通用“奶峰生成器”和“奶字”保留全随机。
    if match.group(1) in {"奶龙字", "奶蛙字"}:
        values["character"] = "naiwa" if match.group(1) == "奶蛙字" else "nailong"
    seen = set()

    def parameter(found: re.Match[str]) -> str:
        key = found.group(1)
        field = _PARAMETER_FIELDS.get(key)
        if field is None:
            raise ValueError(f"不认识参数 --{key}。发送“奶峰生成器”查看用法。")
        value = found.group(2) or found.group(3)
        if not value:
            raise ValueError(f"--{key} 后面需要填写参数值。")
        if field in seen:
            raise ValueError(f"--{key} 重复了，每个参数只写一次。")
        seen.add(field)
        if field in _NUMERIC_LIMITS:
            if not re.fullmatch(r"-?\d{1,3}", value):
                raise ValueError(f"--{key} 请填写整数。")
            values[field] = int(value)
        else:
            values[field] = (
                "random" if value == "随机" else _PARAMETER_VALUES.get(field, {}).get(value, value)
            )
        return ""

    remaining = re.sub(
        r"(?<!\S)--([^\s=＝]+)(?:[=＝]((?!--)\S+)|[ \t]+((?!--)\S+))?",
        parameter, command,
    ).strip()
    if separator:
        remaining = f"{remaining} {literal}".strip()
    return MilkTextOptions(text=remaining, **values)


def _resolve_random_options(options: MilkTextOptions) -> MilkTextOptions:
    """每次从完整候选集独立抽取，保留显式值；整张动图只抽一次，帧间样式不闪变。"""
    values = {
        field: secrets.choice(choices)
        for field, choices in STYLE_CHOICES.items()
        if getattr(options, field) == "random"
    }
    for field, (lower, upper, _) in _NUMERIC_LIMITS.items():
        if getattr(options, field) is None:
            # 字号随机仅抽实际尺寸；自动适配须主动指定 0，避免引入额外的尺寸权重。
            lower = 48 if field == "font_size" else lower
            values[field] = secrets.choice(range(lower, int(np.add(upper, 1))))
    return replace(options, **values)


@lru_cache(maxsize=12)
def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size)


def _rgb(hex_color: str) -> np.ndarray:
    return np.array(tuple(bytes.fromhex(hex_color.removeprefix("#"))), dtype=np.float32)


def _text_layer(
    text: str, *, options: MilkTextOptions, size: int, seed: int, phase: float = 0,
) -> Image.Image:
    font = _font(size)
    # The bundled font has CJK coverage; unsupported symbols are rejected rather than sent as tofu.
    if any(
        not char.isspace() and bytes(font.getmask(char)) == bytes(font.getmask("\U0010ffff"))
        for char in text
    ):
        raise ValueError("有些符号暂不支持，请换成中文、字母或常用标点。")
    rng = np.random.default_rng(seed)
    glyphs = []
    for index, char in enumerate(text):
        advance = max(1, round(font.getlength(char)))
        glyph = Image.new("L", (advance + 48, size + 64))
        ImageDraw.Draw(glyph).text((24, 16), char, font=font, fill=255, anchor="lt")
        random_angle, random_offset, scale, delay = rng.uniform(
            (-abs(options.tilt), -options.amplitude, 0.9, 0),
            (abs(options.tilt), options.amplitude, 1.08, np.multiply(np.pi, 2)),
        )
        animated_phase = np.add(phase, np.multiply(index, 0.8))
        if options.motion in {"bounce", "sway"}:
            animated_phase = np.add(animated_phase, delay)
        movement = np.sin(animated_phase)
        offset = 0
        angle = 0.0
        if options.layout == "jitter":
            angle = random_angle
            offset = round(random_offset)
        elif options.layout == "rise":
            offset = round(np.multiply(options.amplitude, np.sin(np.multiply(index, 0.8))))
        else:
            scale = 1.0
        if options.motion == "bounce":
            offset = round(np.add(offset, np.multiply(options.amplitude, abs(movement))))
        elif options.motion == "wave":
            offset = round(np.add(offset, np.multiply(options.amplitude, movement)))
        elif options.motion == "sway":
            angle = np.add(angle, np.multiply(options.tilt, movement))
        glyph = glyph.resize(
            tuple(np.maximum(1, np.rint(np.multiply(glyph.size, scale))).astype(int)),
            Image.Resampling.BICUBIC,
        ).rotate(float(angle), resample=Image.Resampling.BICUBIC, expand=True)
        advance = round(np.add(np.multiply(advance, scale), 8))
        glyphs.append((glyph, advance, offset))
    # 留足所有动作的空间，动图每帧共用画布，不随笔画边界伸缩或裁掉上下跳动的字。
    mask = Image.new("L", (sum(advance for _, advance, _ in glyphs) + 100, size * 2 + 100))
    left = 28
    for glyph, advance, offset in glyphs:
        layer = Image.new("L", mask.size)
        layer.paste(glyph, (left, int(np.subtract(48, offset))))
        mask = ImageChops.lighter(mask, layer)
        left += advance
    if options.layout == "rise":
        mask = mask.rotate(options.tilt, resample=Image.Resampling.BICUBIC, expand=True)
    if options.motion == "static":
        mask = mask.crop(mask.getbbox())
    padding = 24
    padded = Image.new("L", (mask.width + padding * 2, mask.height + padding * 2))
    padded.paste(mask, (padding, padding))
    outline = padded.filter(ImageFilter.MaxFilter(int(np.add(np.multiply(options.stroke, 2), 1))))
    shadow = outline.filter(ImageFilter.GaussianBlur(2))
    image = Image.new("RGBA", padded.size)
    image.paste("#B56D10", (0, 5), shadow)
    image.paste("#FFC32F" if options.palette == "white" else "#FFFFFF", (0, 0), outline)
    top, bottom = (_rgb(color) for color in PALETTES[options.palette])
    mix = np.linspace(0, 1, padded.height, dtype=np.float32).reshape(-1, 1, 1)
    rgb = np.broadcast_to(top * (1 - mix) + bottom * mix, (*padded.size[::-1], 3))
    gradient = Image.fromarray(rgb.astype(np.uint8)).convert("RGBA")
    gradient.putalpha(padded)
    image.alpha_composite(gradient)
    return image


def render_milk_text(options: MilkTextOptions) -> bytes:
    """Render a bounded PNG or looping GIF shared by chat adapters and the web editor.

    Rendering is serialized to limit Pillow memory use under concurrent bot/web requests.
    Input text is never written to the database or used as a filename.
    """
    text = validate_options(options)
    with _RENDER_LOCK:
        options = _resolve_random_options(options)
        lines = [line.strip() for line in text.split("\n")]
        if len(lines) == 1 and (options.layout == "stack" or len(lines[0]) > 14):
            width = max(8 if options.layout == "stack" else 14, int(np.ceil(len(text) / MAX_LINES)))
            lines = [text[index:index + width] for index in range(0, len(text), width)]
        if len(lines) > MAX_LINES or any(len(line) > 24 for line in lines):
            raise ValueError("这句话有点长，请分成最多四行，每行不超过 24 字。")
        longest = max(map(len, lines))
        size = options.font_size or min(126, max(48, int(1100 / max(1, longest))))
        avatar_names = {
            "nailong": ("nailong",),
            "naiwa": ("naiwa",),
            "both": ("nailong", "naiwa"),
            "none": (),
        }[options.character]
        avatars = []
        for name in avatar_names:
            avatar = (
                secrets.choice(tuple(str(path) for path in AVATARS[name]))
                if options.avatar == "random" else str(AVATARS[name][int(options.avatar) - 1])
            )
            with Image.open(avatar) as raw:
                avatars.append(raw.convert("RGBA"))
        avatar_size = max(100, min(190, size + 46))
        avatar_width = (avatar_size - 28) * len(avatar_names) + 18 if avatar_names else 0
        for avatar in avatars:
            avatar.thumbnail((avatar_size, avatar_size), Image.Resampling.LANCZOS)
        seed = secrets.randbits(32)
        frames = []
        frame_count = 1 if options.motion == "static" else 10
        avatar_center = None
        for frame_index in range(frame_count):
            phase = float(np.multiply(np.divide(frame_index, frame_count), np.multiply(np.pi, 2)))
            layers = [
                _text_layer(
                    line, options=options, size=size, seed=int(np.add(seed, index)), phase=phase,
                )
                for index, line in enumerate(lines)
            ]
            width = max(layer.width for layer in layers) + avatar_width + 40
            height = max(sum(layer.height for layer in layers), avatar_size) + 40
            image = Image.new(
                "RGBA", (width, height),
                "#FFF5D8" if options.background == "cream" else (0, 0, 0, 0),
            )
            top = (height - sum(layer.height for layer in layers)) // 2
            if avatar_center is None:
                first_bounds = layers[0].getbbox()
                last_bounds = layers[-1].getbbox()
                start = np.add(top, first_bounds[1])
                end = np.add(
                    np.add(top, sum(layer.height for layer in layers[:-1])), last_bounds[3],
                )
                avatar_center = int(np.floor_divide(np.add(start, end), 2))
            for index, avatar in enumerate(avatars):
                image.alpha_composite(
                    avatar,
                    (20 + index * (avatar_size - 28),
                     max(0, int(np.subtract(avatar_center, np.floor_divide(avatar.height, 2))))),
                )
            for layer in layers:
                image.alpha_composite(layer, (20 + avatar_width, top))
                top += layer.height
            # 动图压缩到较小画布，限定帧数和像素规模，避免群消息渲染占用过多内存。
            image.thumbnail(
                (1500, 1500) if frame_count == 1 else (960, 960), Image.Resampling.LANCZOS,
            )
            frames.append(image)
        output = io.BytesIO()
        if frame_count == 1:
            frames[0].save(output, "PNG", optimize=True)
        else:
            if options.background == "transparent":
                bounds = np.asarray([frame.getbbox() for frame in frames])
                left, top = np.maximum(0, np.subtract(bounds[:, :2].min(axis=0), 8))
                right, bottom = np.minimum(
                    frames[0].size, np.add(bounds[:, 2:].max(axis=0), 8),
                )
                # 按全动画的共同边界裁剪，保留跳动空间，同时消掉固定画布留下的大块空白。
                frames = [frame.crop((left, top, right, bottom)) for frame in frames]
            # 共用调色板避免闪色，255 号颜色只用于透明像素；disposal=2 防止运动拖影。
            palette = frames[0].convert("RGB").quantize(colors=255)
            encoded = []
            for image in frames:
                frame = image.convert("RGB").quantize(palette=palette, dither=Image.Dither.NONE)
                pixels = np.asarray(frame).copy()
                pixels[np.asarray(image.getchannel("A")) < 128] = 255
                frame = Image.fromarray(pixels, mode="P")
                frame.putpalette(palette.getpalette())
                encoded.append(frame)
            encoded[0].save(
                output, "GIF", save_all=True, append_images=encoded[1:], loop=0,
                duration=options.duration_ms, transparency=255, disposal=2, optimize=False,
            )
        return output.getvalue()
