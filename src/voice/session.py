from __future__ import annotations

import threading
import time

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from config.settings import Settings, get_settings
from voice.vad import EndpointDetector, VadConfig


def _rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


class VoiceSession(QObject):
    """Open-mic tutoring session: energy endpointing on the GUI thread."""

    utterance_ready = Signal(object)
    barge_in = Signal()
    speech_started = Signal()
    capture_rejected = Signal()
    error = Signal(str)

    def __init__(self, settings: Settings | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        cfg = settings or get_settings()
        self._sample_rate = 16000
        self._vad = EndpointDetector(
            VadConfig(
                sample_rate=self._sample_rate,
                noise_multiplier=cfg.vad_noise_multiplier,
                min_rms=cfg.vad_min_rms,
                onset_ms=cfg.vad_onset_ms,
                barge_ms=cfg.vad_barge_ms,
                silence_ms=cfg.vad_silence_ms,
                min_speech_ms=cfg.vad_min_speech_ms,
                max_utterance_s=cfg.vad_max_utterance_s,
                preroll_ms=cfg.vad_preroll_ms,
            )
        )
        self._barge_ms = cfg.vad_barge_ms
        self._onset_ms = cfg.vad_onset_ms
        self._barge_min_rms = cfg.vad_barge_min_rms
        self._barge_multiplier = cfg.vad_barge_noise_multiplier
        self._barge_grace_s = max(cfg.vad_barge_grace_ms, 0) / 1000.0

        self._mode = "paused"
        self._running = False
        self._barge_latched = False
        self._monitor_since = 0.0
        self._echo_peak = 0.0
        self._barge_enabled = True
        self._stream = None
        self._lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self._audio_error = ""

        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._poll)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def start(self) -> None:
        if self._running:
            return
        import sounddevice as sd

        self._vad.reset()
        self._barge_latched = False
        self._mode = "listen"
        with self._lock:
            self._chunks.clear()
        stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            callback=self._on_audio,
        )
        try:
            stream.start()
        except Exception:
            stream.close()
            raise
        self._stream = stream
        self._running = True
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        self._running = False
        self._mode = "paused"
        self._barge_latched = False
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        with self._lock:
            self._chunks.clear()
            self._audio_error = ""
        self._vad.reset_utterance()

    def set_mode(self, mode: str) -> None:
        if mode not in {"listen", "paused", "monitor"}:
            raise ValueError(f"unknown voice mode: {mode}")
        self._mode = mode
        self._barge_latched = False
        self._echo_peak = 0.0
        self._monitor_since = time.monotonic() if mode == "monitor" else 0.0
        self._vad.reset_utterance()
        with self._lock:
            self._chunks.clear()

    def set_barge_enabled(self, enabled: bool) -> None:
        """Speaker playback must not be treated as the student interrupting."""
        self._barge_enabled = enabled
        if not enabled:
            self._barge_latched = False
            self._vad.reset_utterance()

    def refresh_barge_grace(self) -> None:
        """Ignore speaker bleed at the moment playback starts."""
        if self._mode != "monitor" or self._barge_latched:
            return
        self._monitor_since = time.monotonic()
        self._echo_peak = 0.0
        self._vad.reset_utterance()
        with self._lock:
            self._chunks.clear()

    def _on_audio(self, indata, frames, time_info, status) -> None:  # noqa: ARG002
        try:
            data = np.asarray(indata, dtype=np.float32)
            if data.size == 0:
                return
            if data.ndim > 1:
                data = data[:, 0]
            chunk = np.ascontiguousarray(data, dtype=np.float32).copy()
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._audio_error = str(exc)
            return
        with self._lock:
            self._chunks.append(chunk)

    def _take(self) -> np.ndarray:
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            data = np.concatenate(self._chunks)
            self._chunks.clear()
            return data

    def _poll(self) -> None:
        with self._lock:
            audio_error = self._audio_error
            self._audio_error = ""
        if audio_error:
            self.error.emit(f"麦克风采集失败: {audio_error}")
            self.stop()
            return
        stream = self._stream
        if self._running and stream is not None and not stream.active:
            self.error.emit("麦克风中断了，请再点一次语音辅导")
            self.stop()
            return
        samples = self._take()
        if samples.size == 0 or not self._running:
            return
        if self._mode == "paused":
            self._vad.observe_noise(samples)
            return
        if self._mode == "monitor":
            if not self._barge_enabled:
                return
            if self._calibrating_echo(samples):
                return
            floor = max(self._barge_min_rms, self._echo_peak * 2.5)
            if _rms(samples) < floor:
                return
            self._feed(
                samples,
                onset_ms=self._barge_ms,
                min_rms=floor,
                noise_multiplier=self._barge_multiplier,
            )
            return
        self._feed(samples, onset_ms=self._onset_ms)

    def _calibrating_echo(self, samples: np.ndarray) -> bool:
        """Learn speaker bleed before treating loud audio as the student interrupting."""
        elapsed = time.monotonic() - self._monitor_since
        calibrating = elapsed < self._barge_grace_s or (
            self._echo_peak < 0.02 and elapsed < 2.0
        )
        if not calibrating:
            return False
        self._echo_peak = max(self._echo_peak, _rms(samples))
        return True

    def _feed(self, samples: np.ndarray, **kwargs) -> None:
        frame = max(int(self._sample_rate * 0.02), 1)
        offset = 0
        while offset < int(samples.size):
            piece = samples[offset : offset + frame]
            offset += frame
            event, audio = self._vad.process(piece, **kwargs)
            if event == "onset":
                if self._mode == "monitor" and not self._barge_latched:
                    self._barge_latched = True
                    self.barge_in.emit()
                elif self._mode == "listen":
                    self.speech_started.emit()
                continue
            if event == "reject":
                latched = self._barge_latched
                self._barge_latched = False
                if latched:
                    self.capture_rejected.emit()
                    return
                continue
            if event == "endpoint" and audio is not None:
                self._barge_latched = False
                self.utterance_ready.emit(audio)
                return
