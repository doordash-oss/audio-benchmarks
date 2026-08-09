"""M4: the Pipecat model leg.

Wraps an already-connected OrchestratorConnection (the M3 audio leg) in a custom
Pipecat transport, and runs OpenAIRealtimeLLMService so the adapter actually
listens to the customer and speaks back. This is what replaces the M3 echo loop.

Audio rates: the orchestrator wire is 48 kHz; the OpenAI Realtime service is locked
to 24 kHz and does NOT resample its input. So the input transport resamples the
customer's 48 kHz audio down to 24 kHz before handing it to Pipecat; on the way out,
Pipecat's BaseOutputTransport resamples the model's 24 kHz audio up to 48 kHz for us
(audio_out_sample_rate=48000) before write_audio_frame, and we push that to the
outgoing WebRTC track (whose drift control keeps B.wav from drifting late).
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone

import aiohttp
from aiortc.mediastreams import MediaStreamError
from av.audio.resampler import AudioResampler

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.services.openai.realtime.events import (
    AudioConfiguration,
    AudioInput,
    AudioOutput,
    InputAudioTranscription,
    SessionProperties,
    TurnDetection,
)
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

OPENAI_SR = 24000   # OpenAI Realtime fixed input rate
GEMINI_SR = 16000   # Gemini Live fixed input rate (output 24k -> resampled to WIRE_SR)
CASCADED_SR = 16000  # cascaded STT/Silero VAD input rate (TTS output resampled to WIRE_SR)
WIRE_SR = 48000     # orchestrator wire rate
GROK_SR = WIRE_SR   # Grok PCM supports 48 kHz, so avoid an input resample.
ULTRAVOX_SR = WIRE_SR  # Ultravox Realtime serverWebSocket uses 48 kHz PCM.


class _OrchInput(BaseInputTransport):
    """Reads the customer audio off the WebRTC incoming track, resamples 48k->24k,
    and pushes it into the pipeline as InputAudioRawFrame."""

    def __init__(self, conn, params: TransportParams, **kwargs):
        super().__init__(params, **kwargs)
        self._conn = conn
        self._target_sr = params.audio_in_sample_rate or OPENAI_SR
        self._task = None
        self._resampler = None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        if self._task is None:
            self._resampler = AudioResampler(format="s16", layout="mono", rate=self._target_sr)
            await self.set_transport_ready(frame)
            self._task = self.create_task(self._recv_loop())

    async def _recv_loop(self):
        track = self._conn.incoming_track
        while True:
            try:
                frame = await track.recv()
            except MediaStreamError:
                break
            for f in self._resampler.resample(frame):
                await self.push_audio_frame(
                    InputAudioRawFrame(audio=f.to_ndarray().tobytes(),
                                       sample_rate=self._target_sr, num_channels=1)
                )

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._cancel_recv()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._cancel_recv()

    async def _cancel_recv(self):
        if self._task:
            await self.cancel_task(self._task)
            self._task = None


class _OrchOutput(BaseOutputTransport):
    """Receives the model's audio (already resampled to 48k by BaseOutputTransport)
    and pushes it to the outgoing WebRTC track -> B.wav / the customer."""

    def __init__(self, conn, params: TransportParams, **kwargs):
        super().__init__(params, **kwargs)
        self._conn = conn
        # Cumulative byte/sample counters for the in-flight bot turn.
        # `_StdoutEvents` reads + resets these on BotStoppedSpeakingFrame
        # to compute the true playback duration (BotStoppedSpeakingFrame
        # fires when TTS synthesis is done, NOT when audio actually
        # finishes playing — streaming TTS races ahead of realtime, so
        # the marker's wall-clock timestamp underestimates the span end
        # by several seconds on long responses). Counting samples here
        # — at the exact point where bytes hit the outgoing track — is
        # the wire-accurate equivalent of OpenAI Realtime's
        # `output_audio_buffer.stopped` event.
        self.audio_frames: int = 0
        self.audio_sample_rate: int | None = None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        # Required: creates the default audio sender (_media_senders[None]); without
        # this, BaseOutputTransport drops every TTSAudioRawFrame as "destination not
        # registered" and B.wav stays silent.
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        self._conn.outgoing.push_bytes(bytes(frame.audio))
        try:
            nf = int(getattr(frame, "num_frames", 0) or 0)
            self.audio_frames += nf
            if self.audio_sample_rate is None:
                sr = getattr(frame, "sample_rate", None)
                if sr:
                    self.audio_sample_rate = int(sr)
        except Exception:
            pass
        return True

    def take_audio_stats(self) -> tuple[int, int | None]:
        """Read and reset the accumulated frame count + sample rate for
        the current bot turn. Called by `_StdoutEvents` on
        BotStoppedSpeakingFrame to stamp the actual playback duration on
        the agent_stop marker."""
        frames = self.audio_frames
        sr = self.audio_sample_rate
        self.audio_frames = 0
        self.audio_sample_rate = None
        return frames, sr

    async def stop(self, frame: EndFrame):
        # Emit the final sub-frame remainder so the tail of the last utterance isn't dropped.
        flush = getattr(self._conn.outgoing, "flush", None)
        if callable(flush):
            flush()
        await super().stop(frame)


class OrchestratorTransport(BaseTransport):
    def __init__(self, conn, params: TransportParams, **kwargs):
        super().__init__(**kwargs)
        self._in = _OrchInput(conn, params)
        self._out = _OrchOutput(conn, params)

    def input(self) -> _OrchInput:
        return self._in

    def output(self) -> _OrchOutput:
        return self._out


class _RoleBRealtimeLLM(OpenAIRealtimeLLMService):
    """OpenAI Realtime service that can run NON-interruptible, to match the native
    adapter's `interrupt_response: false` — the agent keeps speaking through
    detected customer speech / background noise instead of being cut off.

    Pipecat 1.3.0 has no supported flag for this on the Realtime path: the base service
    unconditionally truncates the in-progress response + broadcasts an interruption on
    every speech_started (llm.py:1028-1031). So we override that one method. When
    interruptible we defer to base.

    NOTE: reaches into a Pipecat-internal method; re-verify on Pipecat upgrades."""

    def __init__(self, *args, interruptible: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._interruptible = interruptible

    async def _handle_evt_speech_started(self, evt):
        if self._interruptible:
            return await super()._handle_evt_speech_started(evt)
        # Non-interruptible: surface the event for metrics/logging but do NOT truncate
        # the in-progress response or broadcast an interruption — let the agent finish.
        await self.broadcast_frame(UserStartedSpeakingFrame)


def _make_role_b_gemini_cls():
    """Lazily-built Gemini Live subclass. Import deferred so this module is
    cheap to import for callers that don't use the Gemini provider."""
    from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService

    class _RoleBGeminiLLM(GeminiLiveLLMService):
        """Gemini Live service that can run NON-interruptible, mirroring the
        OpenAI Realtime `interrupt_response: false` behavior.

        Pipecat 1.3.0's GeminiLiveLLMService calls `_handle_interruption()` from
        `process_frame` whenever an `InterruptionFrame` arrives (llm.py:928-930),
        and that method flips `_bot_is_responding=False` + pushes `TTSStoppedFrame`
        which cuts the agent's audio mid-utterance. We override the hook to be a
        no-op when interruptible=False — the agent keeps speaking through customer
        audio, matching `_RoleBRealtimeLLM`'s OpenAI-side behavior.

        Gemini has NO native equivalent of OpenAI's `interrupt_response` flag, so
        this client-side suppression is the only way to get behavioral parity."""

        def __init__(self, *args, interruptible: bool = True, **kwargs):
            super().__init__(*args, **kwargs)
            self._interruptible = interruptible

        async def _handle_interruption(self):
            if self._interruptible:
                return await super()._handle_interruption()
            # Non-interruptible: skip _set_bot_is_responding(False) + TTSStoppedFrame.
            # The agent's audio stays in-flight; downstream just records the event.
            return None

    return _RoleBGeminiLLM


def _make_role_b_grok_cls():
    """Lazily-built Grok Realtime subclass. Import deferred so OpenAI/Gemini
    users don't need the xAI realtime module loaded unless they select Grok."""
    from pipecat.services.xai.realtime.llm import GrokRealtimeLLMService

    class _RoleBGrokLLM(GrokRealtimeLLMService):
        """Grok Realtime service that honors the benchmark's non-interruptible
        VAD profiles.

        Pipecat 1.3.0's Grok service truncates the current assistant response
        and broadcasts an interruption on server VAD speech_started. The native
        OpenAI adapter's fast profile sets interrupt_response:false, so we
        suppress the truncation/interruption for *-no-interrupt profiles while
        still surfacing user_started for metrics."""

        def __init__(self, *args, interruptible: bool = True, **kwargs):
            super().__init__(*args, **kwargs)
            self._interruptible = interruptible

        async def _handle_interruption(self):
            if self._interruptible:
                return await super()._handle_interruption()
            return None

        async def _handle_evt_speech_started(self, evt):
            if self._interruptible:
                return await super()._handle_evt_speech_started(evt)
            await self.broadcast_frame(UserStartedSpeakingFrame)

    return _RoleBGrokLLM


class _StdoutEvents(FrameProcessor):
    """Translate Pipecat frames into the Role B stdout contract (30_role_b_protocol.md):
      - BotStoppedSpeakingFrame      -> {"type":"event","event":"agent_stop"}
      - TranscriptionFrame (user)    -> {"type":"transcript","source":"input","text":...}
      - InterimTranscriptionFrame    -> {"type":"transcript_delta","source":"input","delta":...}
      - LLMTextFrame (bot tokens)    -> {"type":"transcript_delta","source":"response","delta":...}
      - LLMFullResponseEndFrame      -> {"type":"transcript","source":"response","text":...}
    Placed between the LLM and the output transport; sees the LLM's frames going
    downstream and the output transport's BotStoppedSpeakingFrame going upstream."""

    def __init__(self, emit, enable_transcription: bool, output_transport=None):
        super().__init__()
        self._emit = emit
        self._tx = enable_transcription
        self._resp: list[str] = []
        # The output transport (e.g. _OrchOutput) counts audio samples
        # AS THEY HIT THE WIRE — that's the wire-accurate equivalent of
        # OpenAI Realtime's `output_audio_buffer.stopped` event.
        # BotStoppedSpeakingFrame on the cascaded path fires when TTS
        # synthesis finishes, NOT when audio actually plays out (streaming
        # TTS races ahead of realtime, leaving several seconds of audio
        # queued past the marker). We read transport-side stats on
        # BotStoppedSpeakingFrame and stamp `audio_duration_s` on the
        # marker; the dashboard uses agent_start + audio_duration_s as
        # the span end. Doing the count in the transport (vs. here)
        # matters because _StdoutEvents sits BEFORE TTS in the cascaded
        # pipeline — OutputAudioRawFrame never flows through this
        # processor for that provider.
        self._output_transport = output_transport

    @staticmethod
    def _vlog(kind: str, **fields) -> None:
        """Timestamped (ISO-UTC) marker on stderr for the visualizer's
        contract-event parser (build_viz_data.parse_b_log_events). Distinct
        from emit() (the stdout wire protocol — untouched) and from the
        adapter's human log(): the format here is exactly what build_viz_data's
        ISO_RE matches (`[YYYY-MM-DDTHH:MM:SS.mmmZ] [roleB-viz] <kind> {json}`),
        in UTC so it shares the recording's wall-clock anchor with the native
        adapter's logs. Agent speech is otherwise invisible in the dashboard
        for non-OpenAI providers (Gemini etc.), since their logs don't carry
        the native OpenAI realtime events the legacy parser keys on."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        rec = (" " + json.dumps(fields)) if fields else ""
        sys.stderr.write(f"[{ts}] [roleB-viz] {kind}{rec}\n")
        sys.stderr.flush()

    def _flush_agent_transcript(self) -> None:
        """Emit any accumulated LLM text as one `agent_transcript` viz
        marker and clear the buffer. Idempotent: returns immediately
        if nothing has been collected. Called from the natural turn
        boundary (LLMFullResponseEndFrame) AND as a fallback on
        BotStoppedSpeakingFrame — the cascaded chat-completions LLM
        sometimes emits LLMFullResponseEndFrame upstream of the
        bot-stopped audio event, so the dashboard pairs the turn's
        span with the next turn's text and the current span renders
        as 'transcript missing'. Falling through to bot-stopped picks
        up the orphan."""
        if not self._resp:
            return
        text = "".join(self._resp)
        # `chars` + `words` are denormalized alongside the text so the
        # dashboard timeline can show response length without
        # re-tokenizing every render.
        self._vlog("agent_transcript", text=text,
                   chars=len(text), words=len(text.split()))
        if self._tx:
            self._emit({"type": "transcript", "source": "response", "text": text})
        self._resp = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            # Defensive reset on the transport's counters in case the
            # previous turn never closed (e.g. CancelFrame mid-utterance).
            if self._output_transport is not None:
                try:
                    self._output_transport.take_audio_stats()
                except Exception:
                    pass
            self._vlog("agent_start")
        elif isinstance(frame, BotStoppedSpeakingFrame):
            # Flush BEFORE emitting agent_stop so the dashboard sees
            # the transcript paired with this span, not the next.
            self._flush_agent_transcript()
            audio_frames, sr = 0, None
            if self._output_transport is not None:
                try:
                    audio_frames, sr = self._output_transport.take_audio_stats()
                except Exception:
                    pass
            audio_duration_s = round(audio_frames / sr, 3) if (sr and audio_frames > 0) else 0.0
            self._emit({"type": "event", "event": "agent_stop"})
            self._vlog(
                "agent_stop", audio_duration_s=audio_duration_s,
                audio_frames=audio_frames,
            )
        elif isinstance(frame, UserStartedSpeakingFrame):
            # B's VAD perception of the customer talking. Mirrors the OpenAI
            # native adapter's input_audio_buffer.speech_started — build_viz_data
            # turns these into customer_speech rows for the dashboard's VAD
            # bars. Without them the timeline shows agent intervals only and
            # there's no way to see when Gemini's server-side VAD actually
            # committed to "user is speaking" (the diagnostic signal for
            # premature-response / no-response cases).
            self._vlog("user_started")
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._vlog("user_stopped")
        elif isinstance(frame, LLMFullResponseStartFrame):
            # Always reset on a new LLM turn, regardless of _tx — the
            # accumulator is what powers the agent_transcript viz
            # marker, and we want that for every provider (cascaded
            # included) so the dashboard shows agent speech.
            self._resp = []
        elif isinstance(frame, LLMTextFrame):
            self._resp.append(frame.text)
            if self._tx:
                self._emit({"type": "transcript_delta", "source": "response", "delta": frame.text})
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._flush_agent_transcript()
        elif self._tx and isinstance(frame, TranscriptionFrame):
            self._emit({"type": "transcript", "source": "input", "text": frame.text})
        elif self._tx and isinstance(frame, InterimTranscriptionFrame):
            self._emit({"type": "transcript_delta", "source": "input", "delta": frame.text})
        await self.push_frame(frame, direction)


# ── VAD mode profiles ─────────────────────────────────────────────────────────
# Four profiles, two orthogonal axes:
#   speed:        fast | slow         — what counts as a turn-end (timing).
#   interruptible: -interrupt | -no-interrupt
#                                     — whether customer speech truncates the
#                                       agent's in-progress response.
#
# Profile names: "fast-interrupt", "fast-no-interrupt", "slow-interrupt", "slow-no-interrupt".
# Back-compat aliases: "fast" → "fast-no-interrupt", "slow" → "slow-interrupt"
# (matches the prior 2-mode mapping where fast meant interrupt_response:false and
# slow meant interrupt_response:true on the native adapter).
_VAD_MODES = {
    "fast-interrupt", "fast-no-interrupt",
    "slow-interrupt", "slow-no-interrupt",
}
_VAD_ALIASES = {"fast": "fast-no-interrupt", "slow": "slow-interrupt"}


def _parse_vad_mode(vad_mode: str | None) -> tuple[str, bool]:
    """Parse a VAD profile name into (speed, interruptible).

    speed is "fast" or "slow"; interruptible is True/False. Unknown profile
    names default to ("slow", True) — the safest "patient + responsive" preset.
    """
    name = (vad_mode or "").strip().lower()
    name = _VAD_ALIASES.get(name, name)
    if name not in _VAD_MODES:
        name = "slow-interrupt"
    speed, _, suffix = name.partition("-")
    return speed, (suffix == "interrupt")


def _turn_detection(vad_mode: str) -> TurnDetection:
    """OpenAI Realtime server_vad config keyed off the SPEED axis only.
    `interrupt_response` is enforced separately at the service level via
    _RoleBRealtimeLLM, since Pipecat 1.3.0 has no field for it.

    Native presets (gptRealtime_adapter.js):
      fast: threshold 0.6, prefix_padding 1200ms, silence 450ms
      slow: threshold 0.9 (OpenAI defaults for padding/silence)
    """
    speed, _ = _parse_vad_mode(vad_mode)
    if speed == "fast":
        return TurnDetection(threshold=0.6, prefix_padding_ms=1200, silence_duration_ms=450)
    return TurnDetection(threshold=0.9)


def _vad_params_gemini(vad_mode: str):
    """Gemini Live VAD config keyed off the SPEED axis only.

    Maps the OpenAI fast/slow shape onto the discrete Gemini knobs:
      - threshold (continuous) → start_sensitivity (HIGH = fires easily, ≈ fast 0.6;
                                                   LOW  = requires confidence, ≈ slow 0.9)
      - silence_duration_ms passes through; OpenAI's slow leaves it at OpenAI's default
        (~500ms) so we hold near that with Gemini's default (None = service default)
        rather than hard-code a value.
      - end_sensitivity is a Gemini-only knob — set LOW in fast (commit to end of turn
        quickly) and HIGH in slow (tolerate gaps).
    """
    from pipecat.services.google.gemini_live.llm import GeminiVADParams
    from google.genai.types import StartSensitivity, EndSensitivity
    speed, _ = _parse_vad_mode(vad_mode)
    if speed == "fast":
        return GeminiVADParams(
            start_sensitivity=StartSensitivity.START_SENSITIVITY_HIGH,
            end_sensitivity=EndSensitivity.END_SENSITIVITY_LOW,
            prefix_padding_ms=1200,
            silence_duration_ms=450,
        )
    return GeminiVADParams(
        start_sensitivity=StartSensitivity.START_SENSITIVITY_LOW,
        end_sensitivity=EndSensitivity.END_SENSITIVITY_HIGH,
    )


def _build_tools_schema(tools) -> ToolsSchema | None:
    """Convert the runtime's configure_tools list ({name,description,parameters}) into
    a Pipecat ToolsSchema. Pipecat then renders it into the OpenAI Realtime tool format."""
    if not tools:
        return None
    schemas = []
    for t in tools:
        params = t.get("parameters") or {}
        schemas.append(FunctionSchema(
            name=t["name"],
            description=t.get("description", "") or "",
            properties=params.get("properties", {}) or {},
            required=params.get("required", []) or [],
        ))
    return ToolsSchema(standard_tools=schemas)


def _make_tool_handler(name: str, tool_server_url: str, log):
    """Forward a model tool call to the Order Up tool server (Role B contract §4):
    POST {serverUrl}/api/tool {name, arguments} -> {result}; feed result back to the model.
    The adapter never executes tools itself."""
    async def handler(p: FunctionCallParams):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{tool_server_url}/api/tool",
                    json={"name": p.function_name, "arguments": p.arguments},
                ) as resp:
                    data = await resp.json()
            result = data.get("result")
        except Exception as e:  # noqa: BLE001 — surface tool errors to the model, don't crash
            log(f"tool call {name}({p.arguments}) failed: {e!r}")
            result = {"error": str(e)}
        await p.result_callback(result)
    return handler


def _build_openai_llm(args, tools_schema, log):
    """OpenAI Realtime service (24 kHz). interruptible=ON by default — see PLAN.md M7."""
    session_props = SessionProperties(
        instructions=(args.systemPrompt or None),
        output_modalities=["audio"],
        tools=tools_schema,
        audio=AudioConfiguration(
            input=AudioInput(
                turn_detection=_turn_detection(args.vadMode),
                # Match the native adapter's input transcription model (gpt-4o-transcribe);
                # Pipecat's default is gpt-realtime-whisper, a parity gap on "B heard".
                transcription=(InputAudioTranscription(model="gpt-4o-transcribe")
                               if args.enableTranscription else None),
            ),
            output=AudioOutput(voice=(args.voice or "shimmer")),
        ),
    )
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    settings_kwargs = {"session_properties": session_props}
    if args.model:
        settings_kwargs["model"] = args.model
    # Interruptibility derives from the VAD profile suffix
    # (`fast-no-interrupt` / `slow-no-interrupt` → False; the *-interrupt variants → True),
    # but ROLE_B_INTERRUPTIBLE still overrides for ad-hoc runs that want to test
    # the non-default pairing without changing the VAD profile.
    _, interruptible = _parse_vad_mode(getattr(args, "vadMode", None))
    _ov = os.environ.get("ROLE_B_INTERRUPTIBLE")
    if _ov is not None:
        interruptible = _ov.strip().lower() in ("1", "true", "yes", "on")
    log(f"pipeline: provider=openai model={args.model} voice={args.voice} vad={args.vadMode} "
        f"interruptible={interruptible} transcription={args.enableTranscription} (OpenAI Realtime/WS)")
    return _RoleBRealtimeLLM(
        api_key=api_key,
        interruptible=interruptible,
        settings=OpenAIRealtimeLLMService.Settings(**settings_kwargs),
    )


def _build_gemini_llm(args, tools_schema, log):
    """Gemini Live service (16 kHz in / 24 kHz out). Uses a Live model — NOT the shared
    GEMINI_MODEL (that's the text judge model). Override with GEMINI_LIVE_MODEL."""
    from pipecat.services.google.gemini_live.llm import InputParams
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        # Loud log before raising: the adapter's exception path historically
        # swallowed this RuntimeError and the process just hung instead of
        # exiting — the unconditional log line is the only signal that
        # survives to the file when that happens. If you see this, the
        # subprocess env is missing the key (check scripts/run_complex_reeval_jobs.sh
        # set -a guard, or the wrapper invoking this adapter).
        log("ERROR: GEMINI_API_KEY not set in subprocess environment — adapter cannot continue")
        raise RuntimeError("GEMINI_API_KEY not set in environment")
    # TOOL-CALLING NOTE: the native-audio models (gemini-2.5-flash-native-audio-*) have the
    # best speech quality but are UNRELIABLE at function calling — verified one received
    # well-formed add_to_cart declarations yet made ZERO tool calls (acknowledges verbally
    # → empty cart → score 0). For a cart benchmark, tool-calling is essential, so we default
    # to the half-cascade Live model gemini-3.1-flash-live-preview, which calls tools (verified
    # 23 calls vs 0) at the cost of more TTS-like audio. (NOTE: it currently OVER-calls —
    # re-adding items each turn → duplicate carts; that's prompt/behavior tuning, separate
    # from this plumbing.) Override via GEMINI_LIVE_MODEL. Available Live models on a key:
    # `client.models.list()` filtered for supported_action "bidiGenerateContent".
    model = os.environ.get("GEMINI_LIVE_MODEL") or "gemini-3.1-flash-live-preview"
    # Do NOT use args.voice — the runner passes OpenAI voice names (e.g. "shimmer")
    # which Gemini rejects (error 1007). Gemini voices: Charon/Puck/Kore/Fenrir/Aoede.
    # Default to the service voice; override with GEMINI_VOICE.
    voice = os.environ.get("GEMINI_VOICE")
    vad_params = _vad_params_gemini(getattr(args, "vadMode", None))
    _, interruptible = _parse_vad_mode(getattr(args, "vadMode", None))
    _ov = os.environ.get("ROLE_B_INTERRUPTIBLE")
    if _ov is not None:
        interruptible = _ov.strip().lower() in ("1", "true", "yes", "on")
    kwargs = dict(
        api_key=api_key,
        model=model,
        system_instruction=(args.systemPrompt or None),
        tools=tools_schema,
        params=InputParams(vad=vad_params),
    )
    if voice:
        kwargs["voice_id"] = voice
    log(f"pipeline: provider=gemini_live model={model} voice={voice or '(default)'} "
        f"vad={args.vadMode} interruptible={interruptible} "
        f"vad_sensitivities=(start={vad_params.start_sensitivity}, end={vad_params.end_sensitivity}) "
        f"transcription={args.enableTranscription} (Gemini Live/WS)")
    cls = _make_role_b_gemini_cls()
    return cls(interruptible=interruptible, **kwargs)


_GROK_VOICES = {"ara": "Ara", "rex": "Rex", "sal": "Sal", "eve": "Eve", "leo": "Leo"}


def _grok_voice_from_env(log) -> str | None:
    # The runner passes an OpenAI voice name via --voice; Grok voice selection
    # must be explicit to avoid handing xAI an invalid voice like "shimmer".
    raw = os.environ.get("GROK_VOICE") or os.environ.get("XAI_VOICE")
    if not raw:
        return None
    raw = raw.strip()
    voice = _GROK_VOICES.get(raw.lower())
    if voice:
        return voice
    log(f"WARNING: {raw!r} is not one of Grok's documented voices "
        f"{sorted(_GROK_VOICES.values())}; passing it through unchanged")
    return raw


def _build_grok_llm(args, tools_schema, log):
    """Grok Voice Agent service via xAI Realtime (48 kHz PCM in/out)."""
    from pipecat.services.xai.realtime import events as grok_events

    api_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
    if not api_key:
        log("ERROR: XAI_API_KEY/GROK_API_KEY not set in subprocess environment — "
            "Grok adapter cannot continue")
        raise RuntimeError("XAI_API_KEY or GROK_API_KEY not set in environment")

    model = (args.model
             or os.environ.get("GROK_REALTIME_MODEL")
             or os.environ.get("XAI_REALTIME_MODEL")
             or "grok-voice-think-fast-1.0")
    voice = _grok_voice_from_env(log)
    turn_detection = (
        None if getattr(args, "turnMode", "server_vad") == "none"
        else grok_events.TurnDetection(type="server_vad")
    )
    session_props = grok_events.SessionProperties(
        instructions=(args.systemPrompt or None),
        voice=voice or "Ara",
        turn_detection=turn_detection,
        audio=grok_events.AudioConfiguration(
            input=grok_events.AudioInput(format=grok_events.PCMAudioFormat(rate=GROK_SR)),
            output=grok_events.AudioOutput(format=grok_events.PCMAudioFormat(rate=WIRE_SR)),
        ),
        tools=tools_schema,
    )
    _, interruptible = _parse_vad_mode(getattr(args, "vadMode", None))
    _ov = os.environ.get("ROLE_B_INTERRUPTIBLE")
    if _ov is not None:
        interruptible = _ov.strip().lower() in ("1", "true", "yes", "on")
    log(f"pipeline: provider=grok_realtime model={model} voice={voice or 'Ara'} "
        f"vad={args.vadMode} turn={args.turnMode} interruptible={interruptible} "
        f"transcription={args.enableTranscription} (xAI Grok Realtime/WS)")
    cls = _make_role_b_grok_cls()
    return cls(
        api_key=api_key,
        interruptible=interruptible,
        settings=cls.Settings(
            model=model,
            system_instruction=(args.systemPrompt or None),
            session_properties=session_props,
        ),
    )


def _ultravox_system_prompt(base_prompt: str | None) -> str | None:
    if not base_prompt:
        return None
    suffix = """

# Ultravox tool-use reminder

If a menu option group says "REQUIRED ... (default: X)", the item is complete
when the customer names the item. Do not ask for that option. Call add_to_cart
immediately and omit that option group unless the customer explicitly requested
a non-default choice.
"""
    return base_prompt.rstrip() + suffix


def _build_ultravox_llm(args, tools_schema, log):
    """Ultravox Realtime service (48 kHz PCM in/out) via a one-shot call."""
    import datetime
    import uuid

    from pipecat.services.ultravox.llm import (
        OneShotInputParams,
        UltravoxRealtimeLLMService,
    )

    api_key = os.environ.get("ULTRAVOX_API_KEY")
    if not api_key:
        log("ERROR: ULTRAVOX_API_KEY not set in subprocess environment — "
            "Ultravox adapter cannot continue")
        raise RuntimeError("ULTRAVOX_API_KEY not set in environment")

    model = (
        args.model
        or os.environ.get("ULTRAVOX_MODEL")
        or "ultravox-v0.7"
    )
    voice_id = os.environ.get("ULTRAVOX_VOICE_ID")
    voice = None
    if voice_id:
        try:
            voice = uuid.UUID(voice_id)
        except ValueError as e:
            raise RuntimeError("ULTRAVOX_VOICE_ID must be a UUID") from e

    try:
        temperature = float(os.environ.get("ULTRAVOX_TEMPERATURE", "0"))
    except ValueError as e:
        raise RuntimeError("ULTRAVOX_TEMPERATURE must be a float") from e

    try:
        max_duration_seconds = int(os.environ.get("ULTRAVOX_MAX_DURATION_SECONDS", "3600"))
    except ValueError as e:
        raise RuntimeError("ULTRAVOX_MAX_DURATION_SECONDS must be an integer") from e

    log(f"pipeline: provider=ultravox model={model} voice={voice_id or '(default)'} "
        f"vad={args.vadMode} turn={args.turnMode} temperature={temperature} "
        f"transcription={args.enableTranscription} (Ultravox Realtime/WS)")
    return UltravoxRealtimeLLMService(
        params=OneShotInputParams(
            api_key=api_key,
            system_prompt=_ultravox_system_prompt(args.systemPrompt),
            temperature=temperature,
            model=model,
            voice=voice,
            output_medium="voice",
            max_duration=datetime.timedelta(seconds=max_duration_seconds),
        ),
        one_shot_selected_tools=tools_schema,
    )


def _vad_params_cascaded(vad_mode: str | None):
    """Silero VAD params for the cascaded pipeline, keyed off the SPEED axis only — same
    convention as _turn_detection (OpenAI) and _vad_params_gemini. `stop_secs` is the
    trailing silence an utterance needs before the STT segment closes: the cascaded analogue
    of OpenAI's silence_duration_ms / Gemini's end_sensitivity.
      fast: 0.45s  (commit to end-of-turn quickly, ~ native fast's 450ms)
      slow: 0.80s  (tolerate mid-utterance pauses)
    FDB-v3's cascaded agent used min_silence_duration ~0.55s; we split that across the axis."""
    from pipecat.audio.vad.vad_analyzer import VADParams
    speed, _ = _parse_vad_mode(vad_mode)
    return VADParams(stop_secs=0.45) if speed == "fast" else VADParams(stop_secs=0.8)


def _build_cascaded_vad(args):
    """VADProcessor (Silero) for the cascaded pipeline. Pipecat 1.3.0 has NO transport-level
    vad_analyzer on a custom BaseTransport (only the bundled Daily/WebRTC transports run VAD
    internally), so VAD must be an explicit pipeline stage: a VADProcessor emits the
    VADUserStarted/StoppedSpeakingFrame that the segmented Whisper STT consumes to mark
    utterance boundaries. Import deferred so the module stays cheap for realtime-only callers."""
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.processors.audio.vad_processor import VADProcessor
    analyzer = SileroVADAnalyzer(sample_rate=CASCADED_SR,
                                 params=_vad_params_cascaded(getattr(args, "vadMode", None)))
    return VADProcessor(vad_analyzer=analyzer)


def _build_cascaded(args, tools_schema, log):
    """Cascaded baseline (FDB-v3 parity): Silero VAD + OpenAI Whisper STT + OpenAI LLM +
    OpenAI TTS — the non-realtime reference point for the native S2S providers.

    Unlike the realtime builders (which return a single S2S service), this returns the four
    cascaded components (vad, stt, llm, tts). build_pipeline_task wires them as
    input -> VAD -> STT -> user_aggregator -> LLM -> TTS, so the segmented Whisper STT receives
    the VADUserStarted/StoppedSpeakingFrame it needs and the user aggregator (placed AFTER the
    STT) captures the resulting TranscriptionFrame. The LLM is the tool-registration target; the
    system prompt + tools schema ride on the shared LLMContext, since OpenAILLMService
    (chat-completions) reads both from there rather than from constructor params like the
    realtime services."""
    from pipecat.services.openai.stt import OpenAISTTService
    from pipecat.services.openai.tts import OpenAITTSService
    from pipecat.services.openai.llm import OpenAILLMService
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    # Default to FDB-v3's stack (whisper-1 / gpt-4o / tts-1); override each stage via env.
    # NOTE: --model / ROLE_B_MODEL defaults to a *realtime* model id (gpt-realtime-2) that is
    # meaningless for this chat-completions LLM, so the cascaded LLM is controlled by
    # CASCADED_LLM_MODEL (not args.model) to avoid handing the chat API a realtime model.
    stt_model = os.environ.get("CASCADED_STT_MODEL") or "whisper-1"
    llm_model = os.environ.get("CASCADED_LLM_MODEL") or "gpt-4.1"
    tts_model = os.environ.get("CASCADED_TTS_MODEL") or "tts-1"
    # OpenAI TTS voices overlap the Realtime set (alloy/echo/fable/onyx/nova/shimmer), so the
    # runner's --voice ("shimmer") is valid here; v3 used "nova". Override via CASCADED_TTS_VOICE.
    tts_voice = os.environ.get("CASCADED_TTS_VOICE") or args.voice or "nova"
    vad = _build_cascaded_vad(args)
    stt = OpenAISTTService(api_key=api_key, model=stt_model)
    llm = OpenAILLMService(api_key=api_key, model=llm_model)
    tts = OpenAITTSService(api_key=api_key, model=tts_model, voice=tts_voice)
    _, interruptible = _parse_vad_mode(getattr(args, "vadMode", None))
    _ov = os.environ.get("ROLE_B_INTERRUPTIBLE")
    if _ov is not None:
        interruptible = _ov.strip().lower() in ("1", "true", "yes", "on")
    log(f"pipeline: provider=cascaded stt={stt_model} llm={llm_model} tts={tts_model} "
        f"voice={tts_voice} vad={args.vadMode} interruptible={interruptible} "
        f"transcription={args.enableTranscription} (cascaded STT+LLM+TTS)")
    return (vad, stt, llm, tts)


# Per-provider model-side INPUT sample rate (output is always resampled to WIRE_SR=48k).
_PROVIDER_INPUT_SR = {
    "openai": OPENAI_SR,
    "gemini_live": GEMINI_SR,
    "ultravox": ULTRAVOX_SR,
    "grok": GROK_SR,
    "grok_realtime": GROK_SR,
    "cascaded": CASCADED_SR,
}
_PROVIDER_BUILDERS = {
    "openai": _build_openai_llm,
    "gemini_live": _build_gemini_llm,
    "ultravox": _build_ultravox_llm,
    "grok": _build_grok_llm,
    "grok_realtime": _build_grok_llm,
    "cascaded": _build_cascaded,
}


def build_pipeline_task(conn, args, log, emit, tools=None, tool_server_url=None):
    # Returns (PipelineTask, LLMContext) — the context is shared with the aggregators and
    # must seed the initial LLMContextFrame so tool-result feedback reaches the model.
    """Construct the Pipecat pipeline (transport.input -> <provider S2S LLM> -> stdout
    events -> transport.output), dispatching on args.provider. `emit` writes a contract
    JSON line to stdout; `tools`/`tool_server_url` (from configure_tools) wire the cart tools.
    Adding a provider = one entry in _PROVIDER_BUILDERS/_PROVIDER_INPUT_SR (e.g. xai, ultravox)."""
    provider = (getattr(args, "provider", None) or "openai").lower()
    if provider not in _PROVIDER_BUILDERS:
        raise RuntimeError(f"unknown provider {provider!r}; supported: {sorted(_PROVIDER_BUILDERS)}")

    params = TransportParams(
        audio_in_enabled=True, audio_in_sample_rate=_PROVIDER_INPUT_SR[provider],
        audio_out_enabled=True, audio_out_sample_rate=WIRE_SR,
    )
    transport = OrchestratorTransport(conn, params)

    tools_schema = _build_tools_schema(tools)
    # Provider shape differs around the user aggregator:
    #   realtime  -> [user_agg, <S2S llm>]                (the service does STT+TTS internally)
    #   cascaded  -> [VAD, STT, user_agg, LLM, TTS]       (explicit stages; STT must precede the
    #                aggregator so its TranscriptionFrame is captured, and VAD must precede the
    #                segmented STT so it gets VADUserStarted/StoppedSpeakingFrame)
    # `pre_agg` runs before the user aggregator, `post_agg` after; `tool_target` is the LLM.
    built = _PROVIDER_BUILDERS[provider](args, tools_schema, log)
    if provider == "cascaded":
        vad, stt, llm, tts = built
        tool_target = llm
        # `events` must sit BETWEEN llm and tts so it sees LLMTextFrame /
        # LLMFullResponseEndFrame (the frames _StdoutEvents reads to build
        # agent_transcript). Placing events AFTER tts (the realtime
        # arrangement) lets TTS consume those frames and emit only audio
        # frames downstream — the agent_transcript marker never fires, the
        # dashboard shows "(audio only — transcript missing)". Realtime
        # services emit text + audio frames as a pair so post-tts works
        # there; cascaded is two stages so events has to land in between.
        pre_agg, pre_events_post, post_events_post = [vad, stt], [llm], [tts]
    else:
        tool_target = built
        pre_agg, pre_events_post, post_events_post = [], [built], []

    # Tool handlers are provider-agnostic — every realtime service supports register_function.
    # cancel_on_interruption is gated on the VAD mode's interruptible axis:
    # pipecat's base LLMService cancels in-flight function calls on every
    # InterruptionFrame (llm_service.py:540), independent of the per-provider
    # speech-truncation hook our subclasses already suppress. The clip player's
    # customer audio overlaps the agent's tool-call emission, so an interruption
    # frame fires within milliseconds of the call → the cart action is
    # CANCELLED before the tool server ever sees it (observed end-to-end on a
    # Gemini run: model emitted add_to_cart(onion-rings); pipecat cancelled it
    # 4ms later → empty cart). When the user has chosen a non-interruptible
    # profile, we want tool calls to survive that interruption just like the
    # in-flight speech does. Default (interruptible) preserves pipecat's
    # original cancel-on-interrupt behavior.
    # Same derivation the builders use (VAD profile suffix + ROLE_B_INTERRUPTIBLE override), so
    # cancel_on_interruption and cascaded's allow_interruptions agree with the service-level
    # interruptibility the builder applied.
    _, interruptible = _parse_vad_mode(getattr(args, "vadMode", None))
    _ov = os.environ.get("ROLE_B_INTERRUPTIBLE")
    if _ov is not None:
        interruptible = _ov.strip().lower() in ("1", "true", "yes", "on")
    if tools_schema and tool_server_url:
        for t in tools:
            tool_target.register_function(
                t["name"],
                _make_tool_handler(t["name"], tool_server_url, log),
                cancel_on_interruption=interruptible,
            )
        log(
            f"registered {len(tools)} tool(s) {[t['name'] for t in tools]} -> "
            f"{tool_server_url}/api/tool cancel_on_interruption={interruptible}"
        )
    else:
        log("no tools configured — model will have no cart tools")

    events = _StdoutEvents(
        emit, bool(args.enableTranscription),
        output_transport=transport.output(),
    )
    # Context aggregator pair — REQUIRED for tool-result feedback. The assistant() aggregator
    # records each function-call RESULT into `context`; the realtime services then read it
    # back (Gemini's _process_completed_function_calls iterates self._context.get_messages()
    # to send results to the model). Without it, Gemini never learns add_to_cart succeeded and
    # re-adds the whole cart every turn → duplicate carts. (OpenAI Realtime masked this by
    # tracking results server-side.) The LLM and the aggregator MUST share one context object,
    # so we return `context` for the initial LLMContextFrame below to seed the same instance.
    # Cascaded's LLM (OpenAILLMService, chat-completions) reads its system prompt and tools
    # from the LLMContext, not from constructor params like the realtime services — so seed both
    # here. The realtime path keeps an empty context (system/tools live on the service).
    if provider == "cascaded":
        context = LLMContext(tools=tools_schema) if tools_schema else LLMContext()
        if args.systemPrompt:
            context.add_message({"role": "system", "content": args.systemPrompt})
    else:
        context = LLMContext()
    aggregators = LLMContextAggregatorPair(context)
    pipeline = Pipeline([
        transport.input(),
        *pre_agg,
        aggregators.user(),
        *pre_events_post,
        events,
        *post_events_post,
        transport.output(),
        aggregators.assistant(),
    ])
    # Interruptibility note: the realtime providers gate barge-in via per-service speech-truncation
    # hooks (_RoleBRealtimeLLM / _RoleBGeminiLLM), keyed off the VAD profile. Pipecat 1.3.0 has no
    # PipelineParams.allow_interruptions flag, so the cascaded path currently runs with the
    # framework's default (interruptible) turn-taking. In-flight TOOL calls are still protected via
    # cancel_on_interruption above; full non-interruptible *speech* parity for cascaded (via the
    # user-aggregator turn strategies) is a follow-up. `interruptible` above still drives the tool
    # behavior, so the value isn't ignored.
    return PipelineTask(pipeline, params=PipelineParams()), context


async def run_model_pipeline(conn, args, stop_event: asyncio.Event, log, emit,
                             tools=None, tool_server_url=None) -> None:
    """Run the pipeline until it finishes on its own or stop_event fires."""
    task, context = build_pipeline_task(conn, args, log, emit, tools=tools, tool_server_url=tool_server_url)
    runner = PipelineRunner(handle_sigint=False)  # the adapter owns signal handling
    runner_task = asyncio.create_task(runner.run(task))
    # Initial LLMContextFrame, queued for provider setup / greet-first:
    #   - Gemini Live REQUIRES it to flip _ready_for_realtime_input; without it all input
    #     audio is dropped and the model stays silent. (Verified: probe.)
    #   - Grok Realtime REQUIRES conversation setup before audio is sent; Pipecat drops
    #     user audio until the initial context frame has completed setup.
    #   - OpenAI Realtime doesn't need it to listen, but on the OpenAI path the frame
    #     triggers _create_response() (llm.py:609) → an opening greeting. That's exactly
    #     what --speakFirst asks for, matching the native adapter (which greets unless
    #     speakFirst is off). The native adapter PARSED --speakFirst but the Pipecat path
    #     never acted on it — this wires it.
    provider = (getattr(args, "provider", None) or "openai").lower()
    speak_first = bool(getattr(args, "speakFirst", False))
    if provider in ("gemini_live", "grok", "grok_realtime") or speak_first:
        await asyncio.sleep(0.5)
        # Seed the SAME context the aggregators manage (not a throwaway LLMContext), so the
        # LLM's self._context is the one tool results get written into.
        #
        # Gemini-specific: seed a USER-role priming turn so the model actually greets.
        # Pipecat's _handle_context tries to seed system-instruction-as-message when the
        # context is empty + _inference_on_context_initialization=True, but the Google
        # adapter routes system content to system_instruction (not the messages list).
        # _create_initial_response then sees `messages=[]` and early-returns, so Gemini
        # gets `_ready_for_realtime_input=True` but NO inference is triggered → no
        # greeting, and the customer adapter's first clip plays into silence. A
        # 1-token user turn primes Gemini to respond on connect, matching OpenAI's
        # native --speakFirst behavior.
        # Gemini/Grok/cascaded need a user-role priming turn to actually greet: Gemini routes
        # system content to system_instruction (empty messages → no inference), Grok's setup
        # is conversation-first, and cascaded's chat LLM only generates on a user turn.
        # OpenAI Realtime greets off the bare context frame (_create_response), so it needs
        # no priming turn.
        if provider in ("gemini_live", "grok", "grok_realtime", "cascaded"):
            context.add_message({
                "role": "user",
                "content": "(Customer has just walked up to the counter.)",
            })
        await task.queue_frames([LLMContextFrame(context=context)])
        why = ("Gemini greet-first seed" if provider == "gemini_live"
               else "Grok conversation setup/greet-first seed" if provider in ("grok", "grok_realtime")
               else "cascaded greet-first seed" if provider == "cascaded"
               else "speakFirst greeting")
        log(f"queued initial LLMContextFrame ({why})")
    stop_waiter = asyncio.create_task(stop_event.wait())
    done, _ = await asyncio.wait({runner_task, stop_waiter}, return_when=asyncio.FIRST_COMPLETED)
    if runner_task in done:
        log("pipeline ended on its own")
        stop_event.set()
        exc = runner_task.exception()
        if exc:
            log(f"pipeline error: {exc!r}")
    else:
        log("stop requested — cancelling pipeline")
        await task.cancel(reason="adapter shutdown")
        await asyncio.gather(runner_task, return_exceptions=True)
    stop_waiter.cancel()
