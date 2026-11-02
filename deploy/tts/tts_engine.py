"""加载已试听的固定权重，提供中文 WAV 合成。"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from time import perf_counter


class GptSoVitsEngine:
    def __init__(self, root: Path, state_dir: Path) -> None:
        import numpy as np
        import soundfile as sf
        import torch

        self.np, self.sf = np, sf
        metadata = json.loads((root / "release.json").read_text())
        self.release_id = metadata["release_id"]
        for item in metadata["weights"]:
            path = root / item["file"]
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != item["sha256"]:
                raise ValueError(f"Model integrity check failed: {item['file']}")
        threads = int(os.environ.get("TTS_NUM_THREADS", "4"))
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        upstream = root / "upstream"
        os.chdir(upstream)
        sys.path.insert(0, str(upstream))
        sys.path.insert(0, str(upstream / "GPT_SoVITS"))
        os.environ.setdefault("language", "zh_CN")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from TTS_infer_pack.TTS import TTS, TTS_Config

        config = TTS_Config(
            {
                "custom": {
                    "device": "cpu",
                    "is_half": False,
                    "version": "v2ProPlus",
                    "t2s_weights_path": str(root / "weights/NaiLong-e16.ckpt"),
                    "vits_weights_path": str(root / "weights/NaiLong_e14_s504.pth"),
                    "bert_base_path": str(
                        upstream / "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
                    ),
                    "cnhuhbert_base_path": str(
                        upstream / "GPT_SoVITS/pretrained_models/chinese-hubert-base"
                    ),
                }
            }
        )
        state_dir.mkdir(parents=True, exist_ok=True)
        config.configs_path = str(state_dir / "tts_infer.yaml")
        self.pipeline = TTS(config)
        self.reference = str(root / "reference.wav")
        self.reference_text = metadata["reference_text"]
        # 中文前端和声纹模型按需加载；先真实合成一次，再允许健康检查返回成功。
        self.synthesize("我是奶龙。", 20261103)

    def synthesize(self, text: str, seed: int) -> tuple[bytes, float, float]:
        started = perf_counter()
        # 上游会打印目标台词；服务运行时不把用户输入写入日志。
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            pieces = list(
                self.pipeline.run(
                    {
                        "text": text,
                        "text_lang": "all_zh",
                        "ref_audio_path": self.reference,
                        "prompt_lang": "all_zh",
                        "prompt_text": self.reference_text,
                        "top_k": 5,
                        "top_p": 1.0,
                        "temperature": 1.0,
                        "text_split_method": "cut0",
                        "batch_size": 1,
                        "split_bucket": False,
                        "parallel_infer": False,
                        "seed": seed,
                        "fragment_interval": 0.3,
                        "speed_factor": 1.0,
                    }
                )
            )
        if not pieces or len({rate for rate, _ in pieces}) != 1:
            raise RuntimeError("Model did not produce consistent audio")
        rate = pieces[0][0]
        samples = self.np.concatenate([self.np.asarray(audio).reshape(-1) for _, audio in pieces])
        if not samples.size or not self.np.isfinite(samples).all() or not self.np.any(samples):
            raise RuntimeError("Model produced invalid audio")
        if samples.dtype == self.np.int16:
            samples = samples.astype(self.np.float32) / 32768
        output = io.BytesIO()
        self.sf.write(output, samples, rate, format="WAV", subtype="PCM_16")
        return output.getvalue(), len(samples) / rate, perf_counter() - started
