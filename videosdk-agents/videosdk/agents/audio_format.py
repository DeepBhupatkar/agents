from __future__ import annotations
import logging
import threading
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 48000
DEFAULT_CHANNELS = 2

_lock = threading.Lock()
_sample_rate: Optional[int] = None
_channels: Optional[int] = None


def publish(sample_rate: int, channels: int) -> None:
    """Report the format of inbound audio. First call wins."""
    global _sample_rate, _channels

    try:
        rate = int(sample_rate)
        ch = max(1, int(channels))
    except (TypeError, ValueError):
        return
    if rate <= 0:
        return

    with _lock:
        if _sample_rate is not None:
            if (_sample_rate, _channels) != (rate, ch):
                logger.warning(
                    f"[audio_format] already reported as {_sample_rate}Hz "
                    f"x{_channels}ch; ignoring {rate}Hz x{ch}ch"
                )
            return
        _sample_rate = rate
        _channels = ch

    logger.info(f"[audio_format] input audio is {rate}Hz x{ch}ch")


def publish_from_frame(frame) -> None:
    """Report the format of an audio frame. Never raises."""
    try:
        layout = getattr(frame, "layout", None)
        channels = getattr(layout, "nb_channels", None)
        if channels is None and layout is not None:
            channels = len(getattr(layout, "channels", []) or []) or None
        publish(
            getattr(frame, "sample_rate", DEFAULT_SAMPLE_RATE) or DEFAULT_SAMPLE_RATE,
            channels or 1,
        )
    except Exception as e:
        logger.debug(f"[audio_format] could not read frame format: {e}")


def get() -> Tuple[Optional[int], Optional[int]]:
    """Return the reported ``(sample_rate, channels)``, or ``(None, None)``."""
    with _lock:
        return _sample_rate, _channels


def is_known() -> bool:
    return _sample_rate is not None


def reset() -> None:
    """Clear the reported format."""
    global _sample_rate, _channels
    with _lock:
        _sample_rate = None
        _channels = None
