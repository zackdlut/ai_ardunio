"""Settings edited in the app are written back to .env."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from dotenv import dotenv_values
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from config.env_file import update_env_file  # noqa: E402
from config.settings import save_settings  # noqa: E402


def test_update_keeps_comments_and_appends_new_keys(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("# chat\nOPENAI_MODEL=old\nOPENAI_MODEL=dup\n\nTTS_RATE=180\n", encoding="utf-8")
    update_env_file(env, {"OPENAI_MODEL": "qwen-plus", "ASR_MODEL": "whisper-1"})
    assert env.read_text(encoding="utf-8").splitlines() == [
        "# chat",
        "OPENAI_MODEL=qwen-plus",
        "",
        "TTS_RATE=180",
        "",
        "ASR_MODEL=whisper-1",
    ]


def test_json_and_quotes_round_trip_through_dotenv(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    values = {
        "ASR_EXTRA_BODY": '{"asr_options": {"language": "zh"}}',
        "OPENAI_API_KEY": "a'b\\c #x",
        "OPENAI_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "ASR_LANGUAGE": "",
    }
    update_env_file(env, values)
    assert dotenv_values(env) == values
    assert "OPENAI_BASE_URL=https://dashscope" in env.read_text(encoding="utf-8")


def test_invalid_values_are_not_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text("LLM_EXTRA_BODY=\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        save_settings({"LLM_EXTRA_BODY": "[1]"}, env_path=env)
    assert env.read_text(encoding="utf-8") == "LLM_EXTRA_BODY=\n"

    monkeypatch.delenv("ASR_MODEL", raising=False)
    cfg = save_settings({"ASR_MODEL": "FunAudioLLM/SenseVoiceSmall"}, env_path=env)
    assert cfg.asr_model == "FunAudioLLM/SenseVoiceSmall"
    assert os.environ["ASR_MODEL"] == "FunAudioLLM/SenseVoiceSmall"
