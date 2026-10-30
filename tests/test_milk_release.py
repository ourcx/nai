from __future__ import annotations

import hashlib
import io
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image

from nailong_classifier.doubao import DoubaoPrediction
from nailong_classifier.milk_runtime import MilkPriorityPredictor, SharedMilkGuard, verified_weight
from nailong_classifier.model import Prediction


def test_release_rejects_modified_weights_and_paths_outside_bundle(tmp_path):
    weight = tmp_path / "model.pt"
    weight.write_bytes(b"verified checkpoint")
    info = {"file": weight.name, "sha256": hashlib.sha256(weight.read_bytes()).hexdigest()}
    assert verified_weight(tmp_path, info) == weight
    weight.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="hash mismatch"):
        verified_weight(tmp_path, info)
    with pytest.raises(ValueError, match="inside"):
        verified_weight(tmp_path, {**info, "file": "../outside.pt"})


def test_guard_and_classifier_share_one_inference_for_concurrent_calls():
    # 不加载真实权重，测试机器人先过滤再分类时的缓存与线程边界。
    predictor = object.__new__(MilkPriorityPredictor)
    predictor.device = "cpu"
    predictor.guard_threshold = .3
    predictor._lock = threading.Lock()
    predictor._cache = OrderedDict()
    calls = []
    result = (
        Prediction("naiwa", .8, True, {"naiwa": .8}),
        DoubaoPrediction(.01, False, {"doubao": .01, "clean": .99}),
    )

    def predict(image):
        calls.append(image.size)
        return result

    predictor._predict_image = predict
    guard = SharedMilkGuard(predictor)
    content = io.BytesIO()
    Image.new("RGB", (10, 10)).save(content, format="PNG")
    data = content.getvalue()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(guard.predict_bytes, data),
            executor.submit(predictor.predict_bytes, data),
            executor.submit(predictor.predict_bytes, data),
            executor.submit(guard.predict_bytes, data),
        ]
        assert [future.result() for future in futures] == [
            result[1], result[0], result[0], result[1],
        ]
    assert calls == [(10, 10)]


def test_release_cache_is_bounded():
    predictor = object.__new__(MilkPriorityPredictor)
    predictor._lock = threading.Lock()
    predictor._cache = OrderedDict()
    predictor._predict_image = lambda image: (image.size, None)
    for color in range(9):
        content = io.BytesIO()
        Image.new("RGB", (10, 10), color=(color, 0, 0)).save(content, format="PNG")
        predictor._predict_bytes(content.getvalue())
    assert len(predictor._cache) == 8
