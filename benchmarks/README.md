# Chinese Classroom ASR Benchmark

Each JSONL row describes one manually transcribed WAV sample:

```json
{"id":"lecture-001","audio_filepath":"audio/lecture-001.wav","reference":"今天我們要介紹資訊熵的基本概念。","language":"zh-CN","speaker":"講師"}
```

- `audio_filepath`: absolute path or a path relative to the manifest.
- `reference`: manually verified transcript, not another model's output.
- `language`: Nemotron locale such as `zh-CN` or `zh-TW`.
- `speaker`: optional label included in the report.

The report uses aggregate character edit distance divided by aggregate reference characters. Text is normalized with Unicode NFKC, case folding, whitespace removal, and punctuation removal before CER calculation.
