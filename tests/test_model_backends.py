"""Configurable chat endpoint and ASR backends."""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from openai import NotFoundError
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from agent.llm_client import LLMClient  # noqa: E402
from config.settings import Settings  # noqa: E402
from voice.asr import OllamaASR, OpenAIASR, WhisperASR, create_asr  # noqa: E402


def _settings(**env: str) -> Settings:
    return Settings(_env_file=None, **env)


def test_legacy_remote_backend_means_ollama() -> None:
    assert _settings(ASR_BACKEND="Remote").asr_backend == "ollama"


def test_extra_body_must_be_a_json_object() -> None:
    with pytest.raises(ValidationError):
        _settings(LLM_EXTRA_BODY="[1, 2]")
    with pytest.raises(ValidationError):
        _settings(ASR_EXTRA_BODY="{not json")
    cfg = _settings(LLM_EXTRA_BODY='{"enable_thinking": false}')
    assert cfg.llm_extra_body_dict() == {"enable_thinking": False}
    assert cfg.asr_extra_body_dict() == {}


def test_asr_endpoint_falls_back_to_chat_endpoint() -> None:
    cfg = _settings(OPENAI_BASE_URL="http://chat/v1/", OPENAI_API_KEY="k1")
    assert cfg.resolve_asr_base_url() == "http://chat/v1"
    assert cfg.resolve_asr_api_key() == "k1"
    cfg = _settings(ASR_BASE_URL="https://asr/v1", ASR_API_KEY="k2")
    assert cfg.resolve_asr_base_url() == "https://asr/v1"
    assert cfg.resolve_asr_api_key() == "k2"


def test_create_asr_picks_backend() -> None:
    cfg = _settings()
    assert isinstance(create_asr(cfg, backend="local"), WhisperASR)
    assert isinstance(create_asr(cfg, backend="ollama"), OllamaASR)
    assert isinstance(create_asr(cfg, backend="openai"), OpenAIASR)


class _Recorder:
    def __init__(self, result) -> None:
        self.calls: list[dict] = []
        self._result = result

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._result


def test_transcriptions_mode_uploads_wav() -> None:
    transcriptions = _Recorder(SimpleNamespace(text=" 点亮 LED "))
    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=transcriptions))
    asr = OpenAIASR("", "", "whisper-1", language="zh", client=client)
    assert asr.transcribe_pcm(np.zeros(1600, dtype=np.float32)) == "点亮 LED"
    call = transcriptions.calls[0]
    name, wav, mime = call["file"]
    assert (name, mime, wav[:4]) == ("speech.wav", "audio/wav", b"RIFF")
    assert call["model"] == "whisper-1"
    assert call["language"] == "zh"
    assert call["extra_body"] is None


def test_chat_mode_sends_audio_data_url() -> None:
    message = SimpleNamespace(content="language Chinese<asr_text>下一步")
    completions = _Recorder(SimpleNamespace(choices=[SimpleNamespace(message=message)]))
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    extra = {"asr_options": {"language": "zh"}}
    asr = OpenAIASR("", "", "qwen3-asr-flash", mode="chat", extra_body=extra, client=client)
    assert asr.transcribe_pcm(np.zeros(1600, dtype=np.float32)) == "下一步"
    call = completions.calls[0]
    part = call["messages"][0]["content"][0]
    prefix = "data:audio/wav;base64,"
    assert part["type"] == "input_audio"
    assert part["input_audio"]["data"].startswith(prefix)
    assert base64.b64decode(part["input_audio"]["data"][len(prefix):])[:4] == b"RIFF"
    assert call["extra_body"] == extra
    assert call["stream"] is False


def test_whisper_gets_float_audio_and_auto_language() -> None:
    seen: dict = {}

    class _Model:
        def transcribe(self, audio, **kwargs):
            seen["audio"], seen["kwargs"] = audio, kwargs
            return [SimpleNamespace(text="你好")], None

    asr = WhisperASR(language="")
    asr._model = _Model()
    assert asr.transcribe_pcm(np.zeros(160, dtype=np.int16)) == "你好"
    assert seen["audio"].dtype == np.float32
    assert seen["kwargs"] == {"language": None}


def test_ollama_missing_model_error_names_the_fix() -> None:
    asr = OllamaASR("http://ollama:11434", "frozenlab/qwen3-asr:0.6b")
    missing = httpx.Response(404, json={"error": "model 'frozenlab/qwen3-asr:0.6b' not found"})
    assert "ollama pull frozenlab/qwen3-asr:0.6b" in asr._describe_error(missing)
    broken = httpx.Response(500, text="boom")
    assert asr._describe_error(broken) == "Ollama ASR 请求失败 HTTP 500: boom"


def _llm_with_models(list_models) -> LLMClient:
    client = LLMClient.__new__(LLMClient)
    models = SimpleNamespace(list=list_models)
    client._client = SimpleNamespace(with_options=lambda **_: SimpleNamespace(models=models))
    client._model = "qwen-plus"
    client._base_url = "https://cloud/v1"
    client._vision_mode = "false"
    client._extra_body = {}
    client.supports_vision = False
    return client


def test_health_check_tolerates_missing_models_endpoint() -> None:
    def list_models(**_):
        request = httpx.Request("GET", "https://cloud/v1/models")
        raise NotFoundError("no", response=httpx.Response(404, request=request), body=None)

    ok, msg = _llm_with_models(list_models).health_check()
    assert ok
    assert "未校验模型名" in msg


def test_forced_vision_skips_ollama_probe() -> None:
    client = _llm_with_models(lambda **_: [SimpleNamespace(id="qwen-plus")])
    client._probe_vision = lambda: pytest.fail("probe should not run")
    ok, msg = client.health_check()
    assert ok and client.supports_vision is False
