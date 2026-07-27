from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections import deque
from typing import Any, Optional, Dict

import aiohttp
import numpy as np

from videosdk.agents import audio_format
from videosdk.agents.denoise import Denoise as BaseDenoise
from videosdk.agents.utils import resolve_videosdk_auth_token

logger = logging.getLogger(__name__)

VIDEOSDK_INFERENCE_URL = "wss://inference-gateway.videosdk.live"

LATENCY_WINDOW = 50

# Sends can outpace responses indefinitely, so cap the in-flight set.
PENDING_CHUNK_LIMIT = 200

FRAMEWORK_SAMPLE_RATE = 48000
FRAMEWORK_CHANNELS = 2


class _WireSpec:
    """Audio format a provider requires.

    ``channels``: what the provider needs; None to send unchanged.
    ``rate_means``: "client" to report the rate being sent, "model" to report
    the model's target rate instead.
    """

    __slots__ = ("channels", "rate_means", "default_model_rate")

    def __init__(self, channels, rate_means, default_model_rate=None):
        self.channels = channels
        self.rate_means = rate_means
        self.default_model_rate = default_model_rate


_PROVIDER_WIRE = {
    "aicoustics": _WireSpec(channels=1, rate_means="client"),
    "sanas": _WireSpec(channels=2, rate_means="model", default_model_rate=16000),
    "krisp": _WireSpec(channels=None, rate_means="client"),
}

_DEFAULT_WIRE = _WireSpec(channels=None, rate_means="client")


class Denoise(BaseDenoise):
    """
    VideoSDK Inference Gateway Denoise Plugin.

    A lightweight noise cancellation client that connects to VideoSDK's Inference Gateway.
    Supports SANAS and AI-Coustics noise cancellation through a unified interface.

    Audio is returned in the same format it was given, so the denoiser can be
    added to or removed from a pipeline without changing anything around it.
    Sample rate and channel count are handled automatically.

    Example:
        denoise = Denoise.aicoustics()                    # voice-AI default
        denoise = Denoise.sanas()

        pipeline = CascadingPipeline(
            stt=DeepgramSTT(),
            llm=GoogleLLM(),
            tts=ElevenLabsTTS(),
            vad=SileroVAD(),
            turn_detector=TurnDetector(),
            denoise=denoise,
        )
    """

    def __init__(
        self,
        *,
        provider: str,
        model_id: str,
        sample_rate: int | None = None,
        channels: int | None = None,
        chunk_ms: int = 10,
        config: Dict[str, Any] | None = None,
        base_url: str | None = None,
        max_connection_attempts: int = 5,
        auth_token: str | None = None,
        input_sample_rate: int | None = None,
        input_channels: int | None = None,
    ) -> None:
        """
        Initialize the VideoSDK Inference Denoise plugin.

        Args:
            provider: Denoise provider name (e.g., "aicoustics")
            model_id: Model identifier for the provider
            sample_rate: Optional. Handled automatically; only set this if a
                provider requires a specific rate.
            channels: Deprecated and ignored.
            input_sample_rate: Optional. Sample rate of the incoming audio.
                Detected automatically; set only for a custom transport that
                cannot report its own format.
            input_channels: Optional. Channel count of the incoming audio.
                Detected automatically, as above.
            config: Provider-specific configuration dictionary
            base_url: Custom inference gateway URL (default: production gateway)
            max_connection_attempts: After this many consecutive connection
                attempts produce no denoised audio, the client stops trying
                and passes raw audio through to the next pipeline stage for
                the remainder of the session. The counter resets when a
                denoised chunk is successfully received. Default: 5.
            auth_token: VideoSDK auth token. Falls back to the resolved token
                (RoomOptions/WorkerOptions, VIDEOSDK_AUTH_TOKEN, or
                VIDEOSDK_API_KEY + VIDEOSDK_SECRET_KEY) when not provided.
        """
        super().__init__()

        self._videosdk_token = resolve_videosdk_auth_token(auth_token)
        if not self._videosdk_token:
            raise ValueError("VIDEOSDK_AUTH_TOKEN environment variable must be set")

        self.provider = provider
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.channels = channels if channels is not None else 1
        self.chunk_ms = chunk_ms
        self._wire = _PROVIDER_WIRE.get(provider, _DEFAULT_WIRE)
        self.config = config or {}
        self.base_url = base_url or VIDEOSDK_INFERENCE_URL
        self.max_connection_attempts: int = max(1, int(max_connection_attempts))

        self._input_rate_pinned = input_sample_rate is not None
        self._input_channels_pinned = input_channels is not None
        self._input_format_resolved = (
            self._input_rate_pinned and self._input_channels_pinned
        )

        self.input_sample_rate = int(
            input_sample_rate
            if input_sample_rate is not None
            else FRAMEWORK_SAMPLE_RATE
        )
        self.input_channels = max(
            1,
            int(
                input_channels
                if input_channels is not None
                else (self._wire.channels or FRAMEWORK_CHANNELS)
            ),
        )

        # Denoised audio in input format, awaiting return to the pipeline.
        self._out_buffer: bytearray = bytearray()

        self._apply_wire_spec()

        # WebSocket state
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._config_sent: bool = False
        self.connected: bool = False
        self._shutting_down: bool = False

        # Circuit breaker: consecutive connection attempts that produced no
        # denoised audio. Reset on the first denoised chunk after a connect.
        # When it reaches max_connection_attempts, denoise is disabled for
        # the rest of this instance's lifetime and audio is passed through raw.
        self._connection_failures: int = 0
        self._got_denoised_since_connect: bool = False
        self._denoise_disabled: bool = False

        # Audio
        self._send_buffer: bytearray = bytearray()
        self._audio_buffer: asyncio.Queue = asyncio.Queue(maxsize=20)
        self._connect_lock: asyncio.Lock | None = None

        # Latency tracking
        # Maps send sequence number → send timestamp (monotonic)
        self._pending_chunks: deque[tuple[int, float]] = deque(
            maxlen=PENDING_CHUNK_LIMIT
        )
        self._send_seq: int = 0
        self._recv_seq: int = 0

        # Rolling window of round-trip latencies (ms)
        self._latency_window: deque[float] = deque(maxlen=LATENCY_WINDOW)

        # Stats
        self._stats = {
            "chunks_sent": 0,
            "bytes_sent": 0,
            "chunks_received": 0,
            "bytes_received": 0,
            "errors": 0,
            "reconnections": 0,
            "buffer_drops": 0,
            "connection_failures": 0,
            "denoise_disabled": False,
            # Latency stats (ms)
            "latency_last_ms": 0.0,
            "latency_avg_ms": 0.0,
            "latency_min_ms": float("inf"),
            "latency_max_ms": 0.0,
            "latency_p95_ms": 0.0,
        }

        logger.info(
            f"[InferenceDenoise] Initialized: provider={provider}, "
            f"model={model_id}, sample_rate={sample_rate}Hz, channels={channels}"
        )

    # ==================== Factory Methods ====================

    @staticmethod
    def aicoustics(
        *,
        model_id: str = "quail-l-16khz",
        sample_rate: int | None = None,
        channels: int | None = None,
        base_url: str | None = None,
        max_connection_attempts: int = 5,
    ) -> "Denoise":
        """
        Create a Denoise instance configured for AI-Coustics.

        Args:
            model_id: AI-Coustics model (default: "quail-l-16khz").

                Quail — general-purpose speech enhancement for voice AI:
                - "quail-l-16khz": general purpose (recommended default)
                - "quail-s-16khz": faster, smaller
                - "quail-vf-2.2-l-16khz": voice focus, primary-speaker isolation
                - "quail-vf-2.2-s-16khz": voice focus, faster

                Rook — enhancement for human intelligibility:
                - "rook-l-48khz" / "rook-s-48khz" (also 16khz and 8khz variants)

                Legacy "sparrow-*" ids map to the default model.
            sample_rate: Optional. Handled automatically.
            channels: Deprecated and ignored.
            base_url: Custom inference gateway URL

        Returns:
            Configured Denoise instance for AI-Coustics

        Example:
            >>> # Recommended default for voice AI
            >>> denoise = Denoise.aicoustics()
            >>>
            >>> # Primary-speaker isolation
            >>> denoise = Denoise.aicoustics(model_id="quail-vf-2.2-l-16khz")
        """

        return Denoise(
            provider="aicoustics",
            model_id=model_id,
            sample_rate=sample_rate,
            channels=channels,
            chunk_ms=10,
            config={},
            base_url=base_url or VIDEOSDK_INFERENCE_URL,
            max_connection_attempts=max_connection_attempts,
        )

    @staticmethod
    def sanas(
        *,
        model_id: str = "VI_G_NC3.0",
        sample_rate: int | None = None,
        channels: int | None = None,
        base_url: str | None = None,
        max_connection_attempts: int = 5,
    ) -> "Denoise":
        """
        Create a Denoise instance configured for Sanas.

        Args:
            model_id: Sanas model (default: "VI_G_NC3.0")

            sample_rate: Optional. Handled automatically.
            channels: Deprecated and ignored.
            base_url: Custom inference gateway URL

        Returns:
            Configured Denoise instance for Sanas

        Example:
            >>> denoise = Denoise.sanas()
            >>>
            >>> denoise = Denoise.sanas(model_id="VI_G_NC3.0")
        """

        return Denoise(
            provider="sanas",
            model_id=model_id,
            sample_rate=sample_rate,
            channels=channels,
            chunk_ms=20,
            config={},
            base_url=base_url or VIDEOSDK_INFERENCE_URL,
            max_connection_attempts=max_connection_attempts,
        )

    # ==================== Format Adaptation ====================

    def _apply_wire_spec(self) -> None:
        """Resolve the outgoing format from the provider's spec."""
        wire_channels = self._wire.channels
        self._wire_channels = (
            self.input_channels if wire_channels is None else max(1, wire_channels)
        )

        if self._wire.rate_means == "model":
            self._declared_sample_rate = (
                self.sample_rate
                if self.sample_rate is not None
                else self._wire.default_model_rate
            )
            if self.input_sample_rate != FRAMEWORK_SAMPLE_RATE:
                logger.warning(
                    f"[InferenceDenoise] provider={self.provider} expects "
                    f"{FRAMEWORK_SAMPLE_RATE}Hz input but the pipeline supplies "
                    f"{self.input_sample_rate}Hz — audio may be distorted"
                )
        else:
            # Must match the rate actually sent, or the provider resamples from
            # a rate that was never used.
            self._declared_sample_rate = self.input_sample_rate
            if (
                self.sample_rate is not None
                and self.sample_rate != self.input_sample_rate
            ):
                logger.warning(
                    f"[InferenceDenoise] Ignoring sample_rate="
                    f"{self.sample_rate}; audio is sent at "
                    f"{self.input_sample_rate}Hz and the rate is handled "
                    f"automatically"
                )

        self.channels = self._wire_channels

        if self._wire_channels != self.input_channels:
            logger.info(
                f"[InferenceDenoise] Channel adaptation: "
                f"{self.input_channels}ch <-> {self._wire_channels}ch "
                f"at {self._declared_sample_rate}Hz"
            )

    def _resolve_input_format(self) -> None:
        """Adopt the detected input format once, on first audio.

        Values supplied by the caller are never overridden.
        """
        if self._input_format_resolved:
            return
        if not audio_format.is_known():
            return

        rate, channels = audio_format.get()
        new_rate = self.input_sample_rate if self._input_rate_pinned else rate
        new_channels = (
            self.input_channels if self._input_channels_pinned else channels
        )
        self._input_format_resolved = True

        if (new_rate, new_channels) == (self.input_sample_rate, self.input_channels):
            return

        logger.info(
            f"[InferenceDenoise] Adopting transport audio format: "
            f"{self.input_sample_rate}Hz x{self.input_channels}ch -> "
            f"{new_rate}Hz x{new_channels}ch"
        )
        self.input_sample_rate = int(new_rate)
        self.input_channels = max(1, int(new_channels))
        self._apply_wire_spec()
        self._reset_format_state()

    @property
    def _channels_match(self) -> bool:
        return self._wire_channels == self.input_channels

    @property
    def _input_frame_bytes(self) -> int:
        return 2 * self.input_channels

    @property
    def _wire_frame_bytes(self) -> int:
        return 2 * self._wire_channels

    @property
    def _wire_chunk_bytes(self) -> int:
        return (self.chunk_ms * self.input_sample_rate // 1000) * self._wire_frame_bytes

    @staticmethod
    def _remix(pcm: bytes, src_channels: int, dst_channels: int) -> bytes:
        """Convert interleaved int16 between channel counts."""
        if src_channels == dst_channels:
            return pcm
        samples = np.frombuffer(pcm, dtype=np.int16)
        if samples.size == 0:
            return b""

        if src_channels > 1:
            usable = samples.size - (samples.size % src_channels)
            if usable <= 0:
                return b""
            # int32 accumulator: averaging int16 in place would overflow.
            frames = samples[:usable].reshape(-1, src_channels)
            mono = frames.astype(np.int32).mean(axis=1)
        else:
            mono = samples.astype(np.int32)

        if dst_channels > 1:
            mono = np.repeat(mono, dst_channels)

        return np.clip(mono, -32768, 32767).astype(np.int16).tobytes()

    def _to_wire_format(self, pcm: bytes) -> bytes:
        return self._remix(pcm, self.input_channels, self._wire_channels)

    def _to_input_format(self, pcm: bytes) -> bytes:
        return self._remix(pcm, self._wire_channels, self.input_channels)

    def _reset_format_state(self) -> None:
        """Drop pending output on reconnect."""
        self._out_buffer.clear()

    # ==================== Latency Helpers ====================

    def _record_latency(self, latency_ms: float) -> None:
        """Update all latency stats with a new measurement."""
        self._latency_window.append(latency_ms)

        self._stats["latency_last_ms"] = round(latency_ms, 2)
        self._stats["latency_min_ms"] = round(
            min(self._stats["latency_min_ms"], latency_ms), 2
        )
        self._stats["latency_max_ms"] = round(
            max(self._stats["latency_max_ms"], latency_ms), 2
        )
        self._stats["latency_avg_ms"] = round(
            sum(self._latency_window) / len(self._latency_window), 2
        )

        # p95 over rolling window
        if len(self._latency_window) >= 2:
            sorted_w = sorted(self._latency_window)
            p95_idx = int(len(sorted_w) * 0.95)
            self._stats["latency_p95_ms"] = round(sorted_w[p95_idx], 2)

        logger.debug(
            f"[InferenceDenoise] Latency: last={latency_ms:.1f}ms  "
            f"avg={self._stats['latency_avg_ms']}ms  "
            f"min={self._stats['latency_min_ms']}ms  "
            f"max={self._stats['latency_max_ms']}ms  "
            f"p95={self._stats['latency_p95_ms']}ms"
        )

    def _reset_latency_state(self) -> None:
        """Clear pending chunk map on reconnect so stale timestamps don't pollute stats."""
        self._pending_chunks.clear()
        self._send_seq = 0
        self._recv_seq = 0

    # ==================== Circuit Breaker ====================

    def _record_connect_failure(self) -> None:
        """
        Mark one connection cycle as a failure (handshake raised, or the
        connection died before producing any denoised audio). When the
        consecutive count hits ``max_connection_attempts``, disable denoise
        for the rest of this instance's lifetime so audio bypasses the WS
        and flows straight to the next pipeline stage.
        """
        if self._denoise_disabled or self._shutting_down:
            return

        self._connection_failures += 1
        self._stats["connection_failures"] = self._connection_failures
        logger.warning(
            f"[InferenceDenoise] Connection failure "
            f"{self._connection_failures}/{self.max_connection_attempts} "
            f"(provider={self.provider})"
        )

        if self._connection_failures >= self.max_connection_attempts:
            self._denoise_disabled = True
            self._stats["denoise_disabled"] = True
            logger.error(
                f"[InferenceDenoise] Disabled for session after "
                f"{self._connection_failures} failed connection attempts — "
                f"passing audio through without denoising "
                f"(provider={self.provider}, model={self.model_id})"
            )
            # Drop any frames buffered for the dead connection
            self._send_buffer.clear()
            while not self._audio_buffer.empty():
                try:
                    self._audio_buffer.get_nowait()
                except asyncio.QueueEmpty:
                    break

    def _record_connect_success(self) -> None:
        """
        Mark the current connection cycle as successful (first denoised
        audio chunk arrived). Resets the consecutive-failure counter. No-op
        on subsequent chunks of the same cycle.
        """
        if self._got_denoised_since_connect:
            return
        self._got_denoised_since_connect = True
        if self._connection_failures > 0:
            logger.info(
                f"[InferenceDenoise] First denoised chunk received — "
                f"resetting failure counter (was {self._connection_failures})"
            )
            self._connection_failures = 0
            self._stats["connection_failures"] = 0

    # ==================== Core Denoise ====================

    async def denoise(self, audio_frames: bytes, **kwargs: Any) -> bytes:
        """Denoise a chunk, returning it in the same format it was given."""
        out = await self._denoise(audio_frames, **kwargs)
        return self._conform_to_input(out, len(audio_frames), audio_frames)

    def _conform_to_input(
        self, out: bytes, expected_len: int, original: bytes
    ) -> bytes:
        """Enforce the input format on the result.

        A partial frame would desync channel interleaving for the rest of the
        session, so fall back to the original rather than emit one.
        """
        if len(out) == expected_len:
            return out

        frame = self._input_frame_bytes
        if len(out) > expected_len:
            out = out[:expected_len]
        if len(out) % frame:
            out = out[: len(out) - (len(out) % frame)]
        if len(out) < expected_len:
            logger.debug(
                f"[InferenceDenoise] Short output ({len(out)}/{expected_len} "
                f"bytes) — passing input through to preserve frame alignment"
            )
            return original
        return out

    async def _denoise(self, audio_frames: bytes, **kwargs: Any) -> bytes:
        # logger.info(f"Using Sanas secret: {self._secret}")
        # print("enter in denoise")
        if self._denoise_disabled:
            return audio_frames
        try:
            self._resolve_input_format()

            if self._connect_lock is None:
                self._connect_lock = asyncio.Lock()

            frame_size = len(audio_frames)

            if self._shutting_down:
                return audio_frames

            if not self._ws or self._ws.closed:
                if self._connect_lock.locked():
                    return audio_frames

                async with self._connect_lock:
                    if not self._ws or self._ws.closed:
                        try:
                            await self._connect_ws()
                            self.connected = True
                            self._stats["errors"] = 0
                            await self._send_config()

                            chunk_size = self._wire_chunk_bytes
                            self._send_buffer.extend(
                                self._to_wire_format(audio_frames)
                            )
                            if len(self._send_buffer) >= chunk_size:
                                first_chunk = bytes(self._send_buffer[:chunk_size])
                                del self._send_buffer[:chunk_size]
                                await self._send_audio(first_chunk)

                            if not self._ws_task or self._ws_task.done():
                                self._ws_task = asyncio.create_task(
                                    self._listen_for_responses()
                                )
                            logger.info(
                                f"[InferenceDenoise] Ready (provider={self.provider})"
                            )
                        except Exception as e:
                            logger.error(f"[InferenceDenoise] Setup failed: {e}")
                            self._ws = None
                            self._config_sent = False
                            self._send_buffer.clear()
                            self._record_connect_failure()
                            return audio_frames

            if not self._config_sent:
                return audio_frames

            chunk_size = self._wire_chunk_bytes
            self._send_buffer.extend(self._to_wire_format(audio_frames))

            while len(self._send_buffer) >= chunk_size:
                chunk = bytes(self._send_buffer[:chunk_size])
                del self._send_buffer[:chunk_size]
                try:
                    await self._send_audio(chunk)
                except Exception as e:
                    logger.error(f"[InferenceDenoise] Send failed: {e} — resetting")
                    await asyncio.sleep(0.5)
                    self._ws = None
                    self._config_sent = False
                    self._send_buffer.clear()
                    self._reset_latency_state()
                    self._reset_format_state()
                    return audio_frames

            denoised_chunks = []
            while not self._audio_buffer.empty():
                try:
                    denoised_chunks.append(self._audio_buffer.get_nowait())
                except asyncio.QueueEmpty:
                    break

            if denoised_chunks:
                all_denoised = b"".join(denoised_chunks)
                self._stats["chunks_received"] += len(denoised_chunks)
                self._stats["bytes_received"] += len(all_denoised)
                self._out_buffer.extend(self._to_input_format(all_denoised))

                # Drop stale audio rather than accumulate latency.
                max_backlog = frame_size * 20
                if len(self._out_buffer) > max_backlog:
                    dropped = len(self._out_buffer) - max_backlog
                    del self._out_buffer[:dropped]
                    self._stats["buffer_drops"] += 1
                    logger.warning(
                        f"[InferenceDenoise] Output backlog exceeded "
                        f"{max_backlog} bytes — dropped {dropped} bytes"
                    )

            if len(self._out_buffer) >= frame_size:
                out = bytes(self._out_buffer[:frame_size])
                del self._out_buffer[:frame_size]
                return out

            return audio_frames

        except Exception as e:
            logger.error(f"[InferenceDenoise] Error in denoise: {e}", exc_info=True)
            self._stats["errors"] += 1
            return audio_frames

    # ==================== WebSocket ====================

    async def _connect_ws(self) -> None:
        try:
            if self._shutting_down:
                return

            # New connection cycle — until we receive a denoised chunk on this
            # WS, the cycle is provisionally a failure for breaker purposes.
            self._got_denoised_since_connect = False

            if not self._session or self._session.closed:
                self._session = aiohttp.ClientSession()

            ws_url = (
                f"{self.base_url}/v1/denoise"
                f"?provider={self.provider}"
                f"&secret={self._videosdk_token}"
                f"&modelId={self.model_id}"
            )

            logger.info(
                f"[InferenceDenoise] Connecting to {self.base_url} "
                f"(provider={self.provider}, model={self.model_id})"
            )

            self._ws = await self._session.ws_connect(
                ws_url, timeout=aiohttp.ClientTimeout(total=10)
            )
            if self._shutting_down:
                await self._ws.close()
                return
            self._config_sent = False
            self._send_buffer.clear()
            self._reset_latency_state()
            self._reset_format_state()
            logger.info("[InferenceDenoise] Connected successfully")

        except Exception as e:
            logger.error(f"[InferenceDenoise] Connection failed: {e}", exc_info=True)
            raise

    async def _send_config(self) -> None:
        """Send configuration message to the inference server."""
        if not self._ws or self._ws.closed:
            raise ConnectionError("WebSocket not connected")

        config_message = {
            "type": "config",
            "data": {
                "model": self.model_id,
                "sample_rate": self._declared_sample_rate,
                "channels": self._wire_channels,
                **self.config,
            },
        }
        await self._ws.send_str(json.dumps(config_message))
        self._config_sent = True
        logger.info(
            f"[InferenceDenoise] Config sent: "
            f"model={self.model_id}, "
            f"sample_rate={self._declared_sample_rate}Hz, "
            f"channels={self._wire_channels}"
        )

    async def _send_audio(self, audio_bytes: bytes) -> None:
        """Send one audio chunk, stamping it with a sequence number for latency tracking."""
        if not self._ws or self._ws.closed:
            raise ConnectionError("WebSocket not connected")

        seq = self._send_seq
        self._send_seq += 1

        # Record send timestamp BEFORE the await so network time is included
        self._pending_chunks.append((seq, time.monotonic()))

        await self._ws.send_str(
            json.dumps(
                {
                    "type": "audio",
                    "data": base64.b64encode(audio_bytes).decode("utf-8"),
                    "seq": seq,  # server will echo this back if it supports it
                }
            )
        )
        self._stats["chunks_sent"] += 1
        self._stats["bytes_sent"] += len(audio_bytes)

    def _resolve_latency(self, recv_seq: int | None = None) -> None:
        """
        Match a received chunk to a sent chunk and record the round-trip latency.

        If the server echoes 'seq' we match it exactly; otherwise the oldest
        pending timestamp is consumed (FIFO approximation).

        Sequence numbers only increase, so a bounded deque keeps this O(1) and
        caps the in-flight set when sends outpace responses.
        """
        now = time.monotonic()
        pending = self._pending_chunks
        if not pending:
            return

        if recv_seq is None:
            sent_at = pending.popleft()[1]
        else:
            sent_at = None
            while pending:
                seq, ts = pending[0]
                if seq > recv_seq:
                    return
                pending.popleft()
                if seq == recv_seq:
                    sent_at = ts
                    break
            if sent_at is None:
                return

        self._record_latency((now - sent_at) * 1000)

    async def _listen_for_responses(self) -> None:
        """Background task to listen for WebSocket responses from the server."""
        if not self._ws:
            return

        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(msg.data)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    # Binary frames carry raw denoised PCM — measure latency (FIFO)
                    self._resolve_latency()
                    if self._audio_buffer.full():
                        try:
                            self._audio_buffer.get_nowait()
                            self._stats["buffer_drops"] += 1
                        except asyncio.QueueEmpty:
                            pass
                    try:
                        self._audio_buffer.put_nowait(msg.data)
                    except asyncio.QueueFull:
                        pass
                    self._record_connect_success()
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(
                        f"[InferenceDenoise] WebSocket error: {self._ws.exception()}"
                    )
                    break
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                    logger.info("[InferenceDenoise] WebSocket closed by server")
                    break

        except asyncio.CancelledError:
            logger.debug("[InferenceDenoise] Listener cancelled")
        except Exception as e:
            logger.error(f"[InferenceDenoise] Listener error: {e}", exc_info=True)
        finally:
            self._ws = None
            self._config_sent = False
            logger.info("[InferenceDenoise] Listener exited — connection marked dead")
            # If the connection ended without ever producing a denoised chunk,
            # count it as a failed cycle for the circuit breaker.
            if not self._got_denoised_since_connect:
                self._record_connect_failure()

    async def _handle_message(self, raw_message: str) -> None:
        """
        Handle incoming messages from the inference server.

        Args:
            raw_message: Raw JSON message string from server
        """
        # logger.info(f"[STT DEBUG] raw server msg: {raw_message}")
        try:
            data = json.loads(raw_message)
            msg_type = data.get("type")

            if msg_type == "event":
                event_data = data.get("data", {})
                event_type = event_data.get("eventType")

                if event_type == "DENOISE_AUDIO":
                    audio_data = event_data.get("audio", "")
                    # Echo'd seq from server (optional — works without it too)
                    recv_seq = event_data.get("seq", None)
                    if audio_data:
                        self._resolve_latency(recv_seq)
                        denoised = base64.b64decode(audio_data)
                        if self._audio_buffer.full():
                            try:
                                self._audio_buffer.get_nowait()
                                self._stats["buffer_drops"] += 1
                            except asyncio.QueueEmpty:
                                pass
                        try:
                            self._audio_buffer.put_nowait(denoised)
                        except asyncio.QueueFull:
                            pass
                        self._record_connect_success()
                # START_SPEECH, END_SPEECH, TRANSCRIPT silently ignored

            elif msg_type == "audio":
                audio_data = data.get("data", "")
                recv_seq = data.get("seq", None)
                if audio_data:
                    self._resolve_latency(recv_seq)
                    denoised = base64.b64decode(audio_data)
                    if self._audio_buffer.full():
                        try:
                            self._audio_buffer.get_nowait()
                            self._stats["buffer_drops"] += 1
                        except asyncio.QueueEmpty:
                            pass
                    try:
                        self._audio_buffer.put_nowait(denoised)
                    except asyncio.QueueFull:
                        pass
                    self._record_connect_success()

            elif msg_type == "error":
                # logger.error(f"[InferenceDenoise] FULL ERROR MESSAGE: {raw_message}")
                error_data = data.get("data", {})

                # Safely extract error message
                error_msg = (
                    error_data.get("error")
                    or error_data.get("message")
                    or json.dumps(error_data)
                    or "Unknown error"
                )

                self._stats["errors"] += 1

                logger.error(
                    f"[InferenceDenoise] Server error: {error_msg} "
                    f"(total: {self._stats['errors']})"
                )

                # Force reset connection on first error
                if self._stats["errors"] == 1:
                    self._send_buffer.clear()
                    self._config_sent = False

                    if self._ws and not self._ws.closed:
                        try:
                            await self._ws.close()
                        except Exception:
                            pass

                    self._ws = None

        except json.JSONDecodeError as e:
            logger.error(f"[InferenceDenoise] Failed to parse message: {e}")
        except Exception as e:
            logger.error(
                f"[InferenceDenoise] Message handling error: {e}", exc_info=True
            )

    async def _cleanup_connection(self) -> None:
        if self._ws and not self._ws.closed:
            try:
                await asyncio.wait_for(
                    self._ws.send_str(json.dumps({"type": "stop"})), timeout=1.0
                )
                await asyncio.sleep(0.1)
            except Exception:
                pass
            try:
                await self._ws.close()
            except Exception:
                pass

        self._ws = None
        self._config_sent = False
        self._send_buffer.clear()

    # ==================== Utilities ====================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Dictionary containing processing statistics
        """
        return {
            **self._stats,
            "buffer_size": self._audio_buffer.qsize(),
            "pending_chunks": len(self._pending_chunks),
            "provider": self.provider,
            "model": self.model_id,
            "sample_rate": self._declared_sample_rate,
            "channels": self._wire_channels,
            "input_sample_rate": self.input_sample_rate,
            "input_channels": self.input_channels,
            "connected": self._ws is not None and not self._ws.closed,
        }

    def get_latency_stats(self) -> Dict[str, Any]:
        """Return only the latency-related stats — handy for logging/monitoring."""
        return {
            "last_ms": self._stats["latency_last_ms"],
            "avg_ms": self._stats["latency_avg_ms"],
            "min_ms": self._stats["latency_min_ms"],
            "max_ms": self._stats["latency_max_ms"],
            "p95_ms": self._stats["latency_p95_ms"],
            "samples": len(self._latency_window),
        }

    async def aclose(self) -> None:
        logger.info(
            f"[InferenceDenoise] Closing (provider={self.provider}). "
            f"Final stats: {self.get_stats()}"
        )
        self._shutting_down = True
        # Log final latency summary on close
        lat = self.get_latency_stats()
        if lat["samples"] > 0:
            logger.info(
                f"[InferenceDenoise] Latency summary — "
                f"avg={lat['avg_ms']}ms  p95={lat['p95_ms']}ms  "
                f"min={lat['min_ms']}ms  max={lat['max_ms']}ms  "
                f"over {lat['samples']} samples"
            )

        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await asyncio.wait_for(self._ws_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._ws_task = None

        await self._cleanup_connection()

        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

        while not self._audio_buffer.empty():
            try:
                self._audio_buffer.get_nowait()
            except asyncio.QueueEmpty:
                break

        await super().aclose()
        logger.info("[InferenceDenoise] Closed successfully")

    @property
    def label(self) -> str:
        return f"videosdk.inference.Denoise.{self.provider}.{self.model_id}"
