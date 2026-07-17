# NVIDIA Nemotron 3.5 ASR Research

Date: 2026-06-19 (implementation status updated 2026-07-17)

Source model card:
https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b

## Short Answer

`nvidia/nemotron-3.5-asr-streaming-0.6b` is the right NVIDIA ASR model to track
for this project. It is a 600M-parameter multilingual streaming ASR model, not a
chat LLM. It fits Aegis because the current project already has this intended
shape:

```text
audio_chunks -> ASR transcriber -> TranscriptEvent -> Store -> Brain
```

Unlike Parakeet TDT v3, this model includes Mandarin `zh-CN` in its
out-of-the-box broad-coverage tier. That makes it a realistic candidate for
Chinese classroom listening experiments.

## Verified Model Snapshot

- Model id: `nvidia/nemotron-3.5-asr-streaming-0.6b`
- Version: `nemotron-3.5-asr-streaming-0.6b-v1`
- Release date on Hugging Face: 2026-06-04
- Parameters: 600M
- Task: multilingual automatic speech recognition
- Runtime: NVIDIA NeMo 26.06
- Architecture: FastConformer-CacheAware-RNNT with language-ID prompt
- License: OpenMDW-1.1
- OS: Linux, Linux for Tegra
- Hardware families listed: NVIDIA Ampere, Blackwell, Hopper, Jetson, Lovelace,
  Turing, Volta
- Input: mono WAV audio plus language id string
- Output: text in the input language, with punctuation and capitalization
- Commercial use: model card says it is ready for commercial use

## Streaming Behavior

The model is built for native cache-aware streaming, not buffered overlapping
window inference. It processes only new audio chunks while reusing cached encoder
context.

Supported chunk/latency settings are configured through `att_context_size`:

- `[56, 0]`: 80 ms
- `[56, 1]`: 160 ms
- `[56, 3]`: 320 ms
- `[56, 6]`: 560 ms
- `[56, 13]`: 1120 ms

The model can be run with a target language such as `zh-CN`, or with
`target_lang=auto` for automatic language detection and language tagging. If
language tags are kept, output text includes a tag such as `<en-US>` after the
terminal punctuation.

## Language Coverage

The model card lists 40 language-locales total:

- Transcription-ready: 19 locales, highest accuracy out of the box.
- Broad-coverage: 13 additional production ASR locales.
- Adaptation-ready: 8 locales recognized by the tokenizer, but requiring
  fine-tuning for full transcription quality.

Mandarin `zh-CN` is in the broad-coverage tier, not the adaptation-only tier.
That means it is usable out of the box, but should be validated against actual
classroom audio before treating it as production-quality.

For Mandarin on FLEURS, the model card reports CER rather than WER:

- `zh-CN` with LangID at 320 ms: 20.03 CER
- `zh-CN` with LangID at 1120 ms: 19.28 CER
- `zh-CN` auto-detect at 320 ms: 20.59 CER
- `zh-CN` auto-detect at 1120 ms: 19.87 CER

## Fit With This Repo

Current Aegis code already stores WAV chunks and has `TranscriptEvent` as the
central transcript event. The implementation should avoid hard-coding the worker
as Whisper:

1. Add an ASR backend abstraction.
2. Implement an offline chunk transcriber first:
   `AudioChunk.path -> text -> TranscriptEvent(source="nemotron-3.5-asr")`.
3. Add a streaming transcriber path after the offline contract works.
4. Persist language code and streaming partial/final status before adding
   speaker diarization.

This model makes a real streaming path worth designing, but the first step
should still be offline chunk transcription because it matches the current DB and
test surface.

## Implementation Notes

Model loading:

```python
import nemo.collections.asr as nemo_asr

asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/nemotron-3.5-asr-streaming-0.6b"
)
```

The model card points to NeMo's cache-aware streaming script:

```bash
python examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py \
  model_path=<model_path> \
  dataset_manifest=<dataset_manifest> \
  batch_size=<batch_size> \
  target_lang=zh-CN \
  att_context_size="[56,13]" \
  strip_lang_tags=true \
  output_path=<output_folder>
```

For early experiments, use `target_lang=zh-CN` rather than `auto`. Auto language
detection is useful later, but explicit LangID gives a cleaner baseline and
slightly better reported Mandarin CER.

## Practical Caveats

- NeMo install is heavier than Whisper and should be tested carefully on
  CachyOS/KDE.
- Local model file is present at
  `nemotron/nemotron-3.5-asr-streaming-0.6b.nemo` and is about 2.3GB.
- Model card install examples use `apt-get`; on CachyOS this maps to system
  packages for `libsndfile` and `ffmpeg`, plus Python packages in the project
  venv.
- Mandarin is broad-coverage, not top-tier transcription-ready, so local
  classroom accuracy must be measured with real lecture audio.
- The current project cuts fixed-size WAV chunks. Streaming ASR will need a
  separate microphone stream path, VAD/silence handling, partial transcript
  updates, and final transcript commit rules.
- Host-level verification on this machine showed `NVIDIA GeForce RTX 5060
  Laptop GPU`, 8151 MiB VRAM, driver `610.43.03`; NeMo 3.1.0 main at commit
  `ba2cd63ef` restored the local checkpoint and completed CUDA inference.
- VRAM is tight: if `llama-server` is using about 6.6GB on the 8GB GPU, stop it
  before attempting Nemotron model load or inference tests.

## Recommended Next Step

Project plan is now "ASR transcriber backend" instead of a hard-coded Whisper
worker:

- `nemotron-3.5-asr`: primary candidate for multilingual/Chinese streaming.
- `whisper`: fallback multilingual backend.
- `parakeet-v3`: NVIDIA fallback for supported European-language audio.

Milestone 2B implements the backend abstraction and a fake backend first:

- `ASRBackend.transcribe(AudioChunk, language=...) -> ASRResult`
- executable backend: `fake`
- reserved backend names: `nemotron-3.5-asr`, `whisper`
- metadata table: `asr_transcriptions`

Milestone 2C recorded local model readiness and added the guarded adapter.
Milestone 2D now executes real offline chunk inference:

- default local model path:
  `nemotron/nemotron-3.5-asr-streaming-0.6b.nemo`
- `NemotronASRBackend` implements the `ASRBackend` interface without importing
  NeMo at module import time
- the model is lazily restored once per backend instance on CUDA
- each WAV chunk is passed through a temporary language-aware NeMo manifest
- clean transcript text is stored as
  `TranscriptEvent(source="asr:nemotron-3.5-asr")`; runtime and detected-language
  metadata are stored in `asr_transcriptions`
- missing model, NeMo, PyTorch, or CUDA still raises a specific
  `ASRBackendUnavailableError`
- `--probe-asr-env nemotron-3.5-asr --model-path ...` reports local model
  presence/size, ffmpeg, torch, CUDA, NeMo importability, and one of:
  `ready`, `model present, missing nemo`, `missing model`, `missing cuda`, or
  `missing dependencies`

The verified real inference command keeps the planned CLI shape:

```text
aegis.py --transcribe-chunks SESSION_ID --asr-backend nemotron-3.5-asr --model-path nemotron/nemotron-3.5-asr-streaming-0.6b.nemo --lang zh-CN
```

The remaining ASR work is chunk deduplication, a Chinese classroom benchmark,
timestamps/speaker metadata, and then true cache-aware microphone streaming.
