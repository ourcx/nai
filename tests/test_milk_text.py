from __future__ import annotations

import asyncio
import base64
import io
from dataclasses import replace
from types import SimpleNamespace

import pytest
from PIL import Image
from starlette.testclient import TestClient
from test_hdr_bot import run_mock_bot
from test_qq_adapter import event, mock_adapter

import nailong_classifier.application.bot as bot
from nailong_classifier.application.services import BackendServices, SerializedPredictor
from nailong_classifier.imaging.milk_text import (
    AVATARS,
    STYLE_CHOICES,
    MilkTextOptions,
    parse_milk_text_command,
    render_milk_text,
)
from nailong_classifier.persistence.storage import BotStore
from nailong_classifier.web.app import create_app


def fixed_options(text, **changes):
    """格式和像素断言固定全部参数，避免测试结果依赖真实随机抽取。"""
    return replace(MilkTextOptions(
        text, character="nailong", palette="gold", layout="rise", background="transparent",
        avatar="1", motion="static", font_size=0, stroke=9, tilt=4, amplitude=12,
        duration_ms=100,
    ), **changes)


def test_renderer_preserves_transparency_and_bounds_long_multiline_text():
    png = render_milk_text(fixed_options("鸡腿拿来！\n今天也是奶龙", character="both"))
    with Image.open(io.BytesIO(png)) as image:
        assert image.format == "PNG"
        assert image.mode == "RGBA"
        assert image.getpixel((0, 0))[3] == 0
        assert image.getbbox()
        assert max(image.size) < 1600
    cream = render_milk_text(fixed_options("笑得肚子疼", character="naiwa", background="cream"))
    with Image.open(io.BytesIO(cream)) as image:
        assert image.getextrema()[3] == (255, 255)


@pytest.mark.parametrize("text", ["", "a" * 61, "一\n二\n三\n四\n五", "一\n\n二", "😈"])
def test_renderer_rejects_invalid_text(text):
    with pytest.raises(ValueError):
        render_milk_text(MilkTextOptions(text))


@pytest.mark.parametrize("layout", ["jitter", "rise", "straight", "stack"])
def test_renderer_accepts_maximum_length_without_exceeding_canvas_bounds(layout):
    png = render_milk_text(fixed_options("奶" * 60, layout=layout, font_size=140, stroke=14))
    with Image.open(io.BytesIO(png)) as image:
        assert image.getbbox()
        assert max(image.size) < 1600


def test_commands_are_explicit():
    assert parse_milk_text_command("奶峰生成器 鸡腿拿来！").text == "鸡腿拿来！"
    assert parse_milk_text_command("奶峰生成器").text == ""
    assert parse_milk_text_command("奶蛙字：笑得肚子疼").character == "naiwa"
    assert parse_milk_text_command("奶字 鸡腿拿来！").text == "鸡腿拿来！"
    assert parse_milk_text_command("奶龙字很好看") is None
    assert parse_milk_text_command("奶峰生成器很好看") is None
    options = parse_milk_text_command(
        "奶峰生成器 --角色 奶蛙 鸡腿\n拿来！ --配色=黄边白字 --头像 3 "
        "--动作 波浪 --字号 100 --描边 12 --倾斜 -6 --幅度 20 --速度 80"
    )
    assert options == MilkTextOptions(
        "鸡腿\n拿来！", character="naiwa", palette="white", layout="random",
        avatar="3", motion="wave", font_size=100, stroke=12, tilt=-6, amplitude=20,
        duration_ms=80,
    )
    literal = parse_milk_text_command("奶峰生成器 --动作 静态 -- --配色 不是参数")
    assert literal.text == "--配色 不是参数"


@pytest.mark.parametrize("command", [
    "奶峰生成器 鸡腿 --未知 参数",
    "奶峰生成器 鸡腿 --配色",
    "奶峰生成器 鸡腿 --配色 黄边白字 --palette gold",
    "奶峰生成器 鸡腿 --字号 很大",
])
def test_invalid_command_parameters_are_actionable(command):
    with pytest.raises(ValueError, match="参数|重复|整数"):
        parse_milk_text_command(command)


@pytest.mark.parametrize("changes", [
    {"avatar": "../secret"}, {"motion": "飞走"}, {"font_size": 300},
    {"stroke": True}, {"tilt": -13}, {"amplitude": 25}, {"duration_ms": 10},
])
def test_renderer_rejects_unbounded_style_parameters(changes):
    with pytest.raises(ValueError):
        render_milk_text(MilkTextOptions("鸡腿", **changes))


def test_each_character_has_eight_distinct_transparent_real_head_assets():
    for paths in AVATARS.values():
        assert len(paths) == len({path.read_bytes() for path in paths}) == 8
        for path in paths:
            with Image.open(path) as image:
                assert image.mode == "RGBA"
                assert image.size == (256, 256)
                assert image.getextrema()[3] == (0, 255)


def test_defaults_draw_independently_from_full_equal_weight_candidates(monkeypatch):
    import nailong_classifier.imaging.milk_text as milk_text

    draws = []

    def choose_first(values):
        draws.append(tuple(values))
        return values[0]

    monkeypatch.setattr(milk_text.secrets, "choice", choose_first)
    options = parse_milk_text_command("奶峰生成器 鸡腿")
    assert options == MilkTextOptions("鸡腿")
    assert parse_milk_text_command("奶字 鸡腿") == options
    assert parse_milk_text_command("奶龙字 鸡腿").character == "nailong"
    expected = [
        ("nailong", "naiwa", "both", "none"),
        ("gold", "cream", "orange", "white"),
        ("jitter", "rise", "straight", "stack"),
        ("transparent", "cream"),
        ("bounce", "sway", "wave", "static"),
        tuple(range(48, 141)), tuple(range(2, 15)), tuple(range(-12, 13)),
        tuple(range(0, 25)), tuple(range(60, 301)),
    ]
    first = milk_text._resolve_random_options(options)
    assert draws == expected
    draws.clear()
    assert milk_text._resolve_random_options(options) == first
    # 每次仍传入完整无重复列表，而不是给上一次结果降权或剔除。
    assert draws == expected


def test_explicit_parameters_bypass_random_draws_and_preserve_zero(monkeypatch):
    import nailong_classifier.imaging.milk_text as milk_text

    def unexpected_draw(_values):
        pytest.fail("显式参数不应再次随机")

    monkeypatch.setattr(milk_text.secrets, "choice", unexpected_draw)
    options = parse_milk_text_command(
        "奶峰生成器 鸡腿 --角色 纯文字 --头像 2 --配色 黄边白字 --排版 平排 "
        "--背景 透明 --动作 静态 --字号 0 --描边 2 --倾斜 0 --幅度 0 --速度 60"
    )
    assert milk_text._resolve_random_options(options) == options
    assert options.font_size == options.tilt == options.amplitude == 0


def test_partial_parameters_keep_all_other_dimensions_random(monkeypatch):
    import nailong_classifier.imaging.milk_text as milk_text

    draws = []

    def choose_last(values):
        draws.append(tuple(values))
        return values[-1]

    monkeypatch.setattr(milk_text.secrets, "choice", choose_last)
    options = parse_milk_text_command("奶峰生成器 鸡腿 --配色 黄边白字 --幅度 0")
    resolved = milk_text._resolve_random_options(options)
    assert resolved.palette == "white" and resolved.amplitude == 0
    assert tuple(STYLE_CHOICES["palette"]) not in draws
    assert tuple(range(0, 25)) not in draws
    assert resolved.character == "none"
    assert resolved.layout == "stack" and resolved.background == "cream"
    assert resolved.font_size == 140 and resolved.duration_ms == 300
    assert resolved.stroke == 14 and resolved.tilt == 12 and resolved.motion == "static"


def test_random_heads_use_all_eight_candidates_and_allow_repeats(monkeypatch):
    import nailong_classifier.imaging.milk_text as milk_text

    draws = []

    def choose_first(values):
        draws.append(tuple(values))
        return values[0]

    monkeypatch.setattr(milk_text.secrets, "choice", choose_first)
    options = fixed_options("奶", character="both", avatar="random")
    render_milk_text(options)
    expected = [tuple(str(path) for path in paths) for paths in AVATARS.values()]
    assert draws == expected
    draws.clear()
    render_milk_text(options)
    assert draws == expected


@pytest.mark.parametrize("motion", ["bounce", "sway", "wave"])
def test_animated_letters_have_distinct_frames_and_preserve_transparency(motion):
    png = render_milk_text(fixed_options(
        "鸡腿拿来！", palette="white", avatar="1", motion=motion, layout="jitter",
    ))
    with Image.open(io.BytesIO(png)) as image:
        assert image.format == "GIF"
        assert image.n_frames == 10
        assert image.info["loop"] == 0
        assert image.info["duration"] == 100
        frames = []
        for index in range(image.n_frames):
            image.seek(index)
            frame = image.convert("RGBA")
            assert frame.getpixel((0, 0))[3] == 0
            frames.append(frame.tobytes())
        assert len(set(frames)) > 1
        assert max(image.size) <= 960


def test_white_fill_yellow_outline_is_visible_and_fixed_styles_are_reproducible():
    import numpy as np

    options = fixed_options(
        "奶龙", character="none", palette="white", layout="straight", avatar="1",
    )
    first = render_milk_text(options)
    assert first == render_milk_text(options)
    pixels = np.asarray(Image.open(io.BytesIO(first)))
    assert np.any(np.all(pixels == (255, 255, 255, 255), axis=2))
    assert np.any(np.all(pixels == (255, 195, 47, 255), axis=2))


def test_web_editor_and_image_endpoint_work_without_gallery_database(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        page = client.get("/text/")
        assert page.status_code == 200
        assert "奶峰生成器" in page.text
        assert "blob:" in page.headers["content-security-policy"]
        assert client.get("/static/milk-font.ttf").status_code == 200
        response = client.post("/api/milk-text", json={
            "text": "奶龙批准", "character": "nailong", "motion": "static",
        })
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.headers["cache-control"] == "no-store"
        with Image.open(io.BytesIO(response.content)) as image:
            assert image.getbbox()
        assert client.post("/api/milk-text", json={"text": "x" * 61}).status_code == 400
        assert client.post("/api/milk-text", json={"text": []}).status_code == 400
        assert client.post("/api/milk-text", content=b"{}" * 3000).status_code == 413
        animated = client.post("/api/milk-text", json={
            "text": "鸡腿拿来！", "palette": "white", "motion": "wave", "avatar": "8",
            "amplitude": 12, "font_size": 100, "tilt": 4,
        })
        assert animated.headers["content-type"] == "image/gif"
        assert animated.headers["cache-control"] == "no-store"
        with Image.open(io.BytesIO(animated.content)) as image:
            assert image.n_frames == 10
        assert client.post("/api/milk-text", json={"text": "奶", "stroke": []}).status_code == 400
        assert client.get("/static/naiwa-avatar-8.png").status_code == 200
        assert client.get("/static/secret.env").status_code == 404
        assert not (tmp_path / "bot.sqlite3").exists()


def test_web_api_uses_same_random_defaults_and_preserves_explicit_overrides(tmp_path, monkeypatch):
    import nailong_classifier.web.app as web_app

    captured = []
    png = render_milk_text(fixed_options("奶"))

    def capture(options):
        captured.append(options)
        return png

    monkeypatch.setattr(web_app, "render_milk_text", capture)
    with TestClient(create_app(tmp_path)) as client:
        assert client.post("/api/milk-text", json={"text": "鸡腿"}).status_code == 200
        assert captured.pop() == parse_milk_text_command("奶峰生成器 鸡腿")
        response = client.post("/api/milk-text", json={
            "text": "鸡腿", "background": "random", "layout": "jitter",
            "palette": "white", "font_size": 0, "amplitude": 0,
        })
        assert response.status_code == 200
        assert captured.pop() == parse_milk_text_command(
            "奶峰生成器 鸡腿 --背景 随机 --排版 错位 --配色 黄边白字 --字号 0 --幅度 0"
        )


@pytest.mark.parametrize("parameters", ["", " --动作 波浪 --幅度 12 --字号 100"])
def test_bot_requires_mention_and_does_not_classify_or_collect_stickers(
    tmp_path, monkeypatch, parameters,
):
    sent, store = run_mock_bot(
        tmp_path, monkeypatch, text=f"奶峰生成器 鸡腿拿来！{parameters}",
        mentioned_bot=True, random_push_probability=0,
    )
    assert len(sent) == 1
    content, options = sent[0]
    assert isinstance(content["image"]["source"], bytes)
    with Image.open(io.BytesIO(content["image"]["source"])) as image:
        assert image.format in {"PNG", "GIF"}
        if parameters:
            assert image.format == "GIF"
            assert image.n_frames == 10
    assert options == {"reply_to": "source-id"}
    assert store.catalog_stats().total == 0
    sent, _ = run_mock_bot(
        tmp_path, monkeypatch, text="奶峰生成器 鸡腿拿来！",
        mentioned_bot=False, random_push_probability=0,
    )
    assert sent == []


def test_bot_parameter_errors_only_reply_when_addressed(tmp_path, monkeypatch):
    sent, _ = run_mock_bot(
        tmp_path, monkeypatch, text="奶峰生成器 鸡腿 --未知 123",
        mentioned_bot=True, random_push_probability=0,
    )
    assert len(sent) == 1 and "不认识参数" in sent[0][0]["text"]
    sent, _ = run_mock_bot(
        tmp_path, monkeypatch, text="奶峰生成器 鸡腿 --未知 123",
        mentioned_bot=False, random_push_probability=0,
    )
    assert sent == []


def test_qq_command_uploads_generated_png_and_replies_to_original_message(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["nailong-bot"])
    args = bot.parse_args()
    args.state_dir = tmp_path
    args.catalog_document_id = args.catalog_section_block_id = None
    args.random_push_probability = args.douyin_video_push_probability = 0
    store = BotStore(tmp_path)

    def unexpected_prediction(_data):
        pytest.fail("文字生成不应触发识图")

    services = BackendServices(SerializedPredictor(SimpleNamespace(
        device="cpu", threshold=0.65, predict_bytes=unexpected_prediction,
    )), store)

    async def exercise():
        adapter, requests = await mock_adapter()

        async def connect_events():
            await adapter.handle_event("GROUP_AT_MESSAGE_CREATE", event("奶蛙字 鸡腿拿来！"))

        monkeypatch.setattr(adapter, "connect", connect_events)
        try:
            await bot.run_adapter(args, adapter, services)
            uploads = [data for path, data in requests if path.endswith("/files")]
            replies = [data for path, data in requests if path.endswith("/messages")]
            assert len(uploads) == len(replies) == 1
            with Image.open(io.BytesIO(base64.b64decode(uploads[0]["file_data"]))) as image:
                assert image.format == "PNG"
                assert image.getpixel((0, 0)) == (255, 245, 216)
            assert replies[0]["msg_type"] == 7
            assert replies[0]["msg_id"] == "source"
            assert store.catalog_stats().total == 0
        finally:
            await adapter.close()

    asyncio.run(exercise())
