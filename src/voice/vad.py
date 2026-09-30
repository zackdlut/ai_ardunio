from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VadConfig:
    sample_rate: int = 16000
    noise_multiplier: float = 2.2
    min_rms: float = 0.004
    onset_ms: int = 250
    barge_ms: int = 200
    silence_ms: int = 800
    min_speech_ms: int = 300
    max_utterance_s: float = 20.0
    preroll_ms: int = 300
    onset_hang_ms: int = 160


def _rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))


class EndpointDetector:
    """Energy endpointing with a moving noise floor and a short pre-roll."""

    def __init__(self, config: VadConfig | None = None) -> None:
        self._cfg = config or VadConfig()
        self._noise_rms = 0.005
        self._preroll: deque[np.ndarray] = deque()
        self._preroll_n = 0
        self._speech: list[np.ndarray] = []
        self._in_speech = False
        self._above_n = 0
        self._gap_n = 0
        self._silence_n = 0
        self._speech_n = 0
        self._level = 0.0
        self.reset()

    def reset(self) -> None:
        multiplier = max(self._cfg.noise_multiplier, 0.1)
        self._noise_rms = max(self._cfg.min_rms / multiplier, 0.002)
        self._level = 0.0
        self.reset_utterance()

    def reset_utterance(self) -> None:
        self._preroll.clear()
        self._preroll_n = 0
        self._speech = []
        self._in_speech = False
        self._above_n = 0
        self._gap_n = 0
        self._silence_n = 0
        self._speech_n = 0

    def observe_noise(self, samples: np.ndarray) -> None:
        if samples.size == 0 or self._in_speech:
            return
        rms = _rms(samples)
        if rms < self._threshold():
            self._noise_rms = 0.95 * self._noise_rms + 0.05 * rms

    def process(
        self,
        samples: np.ndarray,
        *,
        onset_ms: int,
        min_rms: float | None = None,
        noise_multiplier: float | None = None,
    ) -> tuple[str, np.ndarray | None]:
        if samples.size == 0:
            return "none", None
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        rms = _rms(samples)
        level = self._track_level(rms)
        threshold = self._threshold(min_rms=min_rms, noise_multiplier=noise_multiplier)
        count = int(samples.size)
        hang = self._ms_to_samples(self._cfg.onset_hang_ms)

        if not self._in_speech:
            self._push_preroll(samples)
            if level >= threshold:
                self._above_n += count
                self._gap_n = 0
            else:
                self._gap_n += count
                if self._above_n == 0 or self._gap_n >= hang:
                    self._above_n = 0
                    self._gap_n = 0
                    self._noise_rms = 0.95 * self._noise_rms + 0.05 * rms
            if self._above_n < self._ms_to_samples(onset_ms):
                return "none", None
            self._in_speech = True
            self._speech = list(self._preroll)
            self._speech_n = sum(int(chunk.size) for chunk in self._speech)
            self._silence_n = 0
            self._gap_n = 0
            self._preroll.clear()
            self._preroll_n = 0
            return "onset", None

        self._speech.append(samples)
        self._speech_n += count
        if level < threshold:
            self._silence_n += count
        else:
            self._silence_n = 0

        max_samples = self._ms_to_samples(int(self._cfg.max_utterance_s * 1000))
        silence_needed = self._ms_to_samples(self._cfg.silence_ms)
        if self._speech_n < max_samples and self._silence_n < silence_needed:
            return "none", None

        audio = np.concatenate(self._speech) if self._speech else np.zeros(0, dtype=np.float32)
        voiced = self._speech_n - self._silence_n
        self.reset_utterance()
        if voiced >= self._ms_to_samples(self._cfg.min_speech_ms) and audio.size:
            return "endpoint", audio
        return "reject", None

    def _track_level(self, rms: float) -> float:
        if self._level <= 0:
            self._level = rms
            return rms
        if rms >= self._level:
            self._level = 0.45 * self._level + 0.55 * rms
        else:
            self._level = 0.72 * self._level + 0.28 * rms
        return self._level

    def _threshold(
        self,
        min_rms: float | None = None,
        noise_multiplier: float | None = None,
    ) -> float:
        floor = self._cfg.min_rms if min_rms is None else min_rms
        multiplier = self._cfg.noise_multiplier if noise_multiplier is None else noise_multiplier
        return max(floor, self._noise_rms * multiplier)

    def _ms_to_samples(self, duration_ms: int) -> int:
        return max(int(self._cfg.sample_rate * duration_ms / 1000), 1)

    def _push_preroll(self, samples: np.ndarray) -> None:
        self._preroll.append(samples)
        self._preroll_n += int(samples.size)
        limit = self._ms_to_samples(self._cfg.preroll_ms)
        while self._preroll_n > limit and self._preroll:
            dropped = self._preroll.popleft()
            self._preroll_n -= int(dropped.size)
