# Step 2 — local streaming feasibility

2026-10-25, Asia/Jakarta. **Decision: Pass within the tested conditions.** Continuous local inference and browser updates were verified with paced speech, including 125 seconds of audio. The student confirmed successful real microphone use on the intended demonstration computer; local event logs support that observation. This is a feasibility prototype, not evidence that caption accuracy is adequate or that either assignment is complete.

## Foundation inspection and repair

The Step 1 README, brief, requirement matrix, experiment plan, risks, environment, AI log and dependency files were read. No applicable AGENTS.md or official assignment forms were found in inspected workspace/ancestor locations. The local folder had no Git metadata; it was connected to `muadzarfan/DengerKelas` and compared with `origin/main`, preserving existing files. The previous pip-less venv was retained as `.venv-step1-py314/`, and `.venv` was repaired with Python 3.12.14 and deployment dependencies. `pip check` passed. Training packages were not installed.

## Actual architecture

```mermaid
flowchart LR
    M[Browser microphone or labeled paced fixture] --> A[AudioWorklet]
    A --> P[Incremental mono float32 PCM]
    P --> W[Loopback WebSocket receiver]
    W --> R[Stateful PyAV resampler: 16 kHz]
    R --> B[Bounded ordered audio buffer]
    B --> D[One background ASR worker]
    D --> T[Timestamp history and active hypothesis]
    T --> U[Gradio HTML display]
    U --> L[DOM rendering-opportunity acknowledgement]
```

Gradio 6.30.0 hosts custom HTML through its supported `js_on_load` interface; FastAPI supplies the PCM socket. Gradio Audio is not used for capture: the bridge explicitly sends incremental 250 ms PCM packets, avoiding cumulative-recording ambiguity and ordering the last packet before Stop. Audio capture continues in a browser AudioWorklet while one Python thread decodes. Socket receipt does only short validation/resampling/appending work.

The model loads once. One session owns it at a time. Reset waits for the old worker to exit before new ownership. The worker snapshots current audio instead of queuing inference jobs. If capture exhausts the bounded buffer, the session fails explicitly rather than silently discarding unprocessed speech.

## Model and configuration

- Original model: multilingual [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny), without project fine-tuning; its card labels Apache-2.0.
- Converted artifact: [`Systran/faster-whisper-tiny`](https://huggingface.co/Systran/faster-whisper-tiny), revision `d90ca5fe260221311c53c58e660288d3deb8d356`; its card labels MIT and identifies conversion from the source model. Both license references are retained.
- Local folder: `data/checkpoints/whisper-tiny-ct2/`. Four artifact files total 78,203,619 bytes; model.bin is 75,538,270 bytes. Revision and SHA-256 hashes are in `data/manifests/step2_model.json`; weights remain excluded from Git.
- Runtime: faster-whisper 1.2.1 / CTranslate2 4.8.3, CPU INT8, four CPU threads, one worker, beam size 1, best_of 1, temperature 0, `language=id`, `task=transcribe`, word timestamps enabled, previous-text conditioning disabled, neural VAD disabled, no-speech threshold 0.6.
- Hardware: Intel Core i5-13420H, 8 physical/12 logical cores, 16,857,817,088 bytes RAM. CTranslate2 reports zero available CUDA devices. The student identified this as the demonstration computer.
- Starting engineering settings: 2.5-second initial audio, updates after one second of new audio, 18-second decoding context, three-second overlap, 24-second raw-audio bound and 300-second session limit. These are project choices, not assignment latency thresholds.

Version-sensitive behavior was checked against installed APIs/source and official [faster-whisper documentation](https://github.com/SYSTRAN/faster-whisper), [Gradio HTML source](https://github.com/gradio-app/gradio/blob/main/gradio/components/html.py), and the [converted model card](https://huggingface.co/Systran/faster-whisper-tiny). [WhisperStreaming](https://github.com/ufal/whisper_streaming) is the credited streaming/stabilization research reference; its full agreement policy remains deferred.

## Audio, buffering and transcript handling

Browser packets are noncumulative mono float32 at the AudioContext's actual rate: 48 kHz in observed checks. Multichannel input is averaged. The Python helper additionally handles signed/unsigned integer scaling, validates shape/finite/range values, and resamples waveforms with stateful PyAV/FFmpeg filtering. Filter state and sample-count timestamps persist across chunks; Stop drains the resampler. Tests verified chunked 44.1/48/16 kHz processing against whole-signal resampling.

Microphone audio lives only in bounded buffer/snapshot arrays; raw recordings are not written. The Worklet holds one packet, browser socket backlog is capped at 256 kB plus one packet, and receiver message/transport queues are bounded. The 300-second session limit bounds transcript/log duration. The public replay buffer is a known prerecorded test input, unlike microphone capture, which has no growing recording array. Logs can include microphone text and remain local/ignored.

Entire low-energy windows are gated at RMS 0.003. This is a heuristic, not a trained VAD; quiet speech can be missed and old speech in the overlap can still be decoded during pauses. Five synthetic silent seconds produced no text/model calls; this does not establish room-noise robustness.

Each decode replaces the active hypothesis using absolute word timestamps. On reaching 18 seconds, text before the retained overlap moves to history and older audio is released. No pause is required for rollover, and no global repeated-string removal occurs. History is frozen by age/context position, not agreement or confidence. Timestamp drift can cause boundary omissions, duplicated words or partial/misrecognized words. Stop flushes the final packet/resampler and decodes the tail; Reset cancels abandoned audio instead of promising a flush.

## Measurement definitions

Server timing uses `time.perf_counter()`; audio offsets use cumulative 16 kHz samples.

- **Inference duration:** includes generator iteration through completed faster-whisper decoding. Gated windows are distinguished.
- **Update interval:** between worker `transcript_emitted` events. A separate publisher sends latest state every 250 ms.
- **First-text delay:** first nonempty transcript event minus session start, or minus arrival of the first RMS-positive chunk. The latter is an onset-arrival heuristic, not aligned acoustic onset.
- **Processing lag:** received audio beyond the decoded snapshot end, measured when decoding finishes; not word-display latency.
- **Streaming compute load:** summed decoding durations divided by newly received audio duration, including overlap redecoding.
- **Resources:** actual process RSS sampled after decoding, with hardware/runtime inventory.
- **Browser acknowledgement:** DOM assignment followed by two requestAnimationFrame callbacks. Emission-to-ack includes publishing, transport, browser scheduling and return trip. It measures an upper-bound round-trip to a rendering opportunity, not visible-screen latency.

Counts and audio/wall durations accompany results. P95 appears only with at least 20 observations; smaller runs retain individual measurements. There is no assignment-specified numeric latency target.

## Fixtures and checks

One [CC BY 4.0 FLEURS](https://huggingface.co/datasets/google/fleurs) Indonesian training WAV was retrieved: row ID 1185, archive member `train/10001176200525981995.wav`, dataset revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`, 9.42 seconds/150,720 mono 16 kHz samples. Only 501,760 compressed archive-prefix bytes were read; the full archive was not saved/downloaded. Provenance, reference and hash are in `data/manifests/step2_fixture.json`. Do not reuse this already-exposed training fixture as independent final evaluation evidence.

| Check | Actual outcome |
|---|---|
| Focused checks | 11 Python test cases passed: channel/sample handling, continuous resampling, validation, repeats, rollover/bounds, capture during decoding, explicit overflow, session isolation/cancel, tail flush, silence/decoder exception, missing-model UI, simulated microphone failure/reset and audio without Start. Some cases combine related assertions. Two additional Node checks of the actual browser controller passed for stale status during restart and delayed old AudioContext cleanup. |
| Offline inference | Python socket connect/connect_ex/create_connection denied within the probe process; local model load and genuine speech decoding succeeded. No OS networking setting changed. |
| Short real-rate replay | CLI and browser emitted text before the 9.42-second clip ended. Browser path included Worklet, socket, resampler, worker and DOM acknowledgements. |
| Sustained browser replay | 125 seconds of repeated public speech with one-second pauses: 123 updates, eight rollovers, no error, final lag zero. This is a repeated-clip stress fixture, not distinct lecture speech. |
| Mid-speech Stop | Stop at 13.352 seconds during repeated speech; final packet and final_flush logged; lag returned to zero. |
| Reset/restart | UI text cleared and subsequent sessions had fresh IDs/sample counts. Consecutive replay sessions without Reset were rechecked after a lifecycle fix. |
| Silence | Five seconds of synthetic zeros/80,000 samples in 20 paced chunks: zero text updates or model calls, no error. Plumbing/gating evidence only. |
| Human microphone | Student confirmed live text, Stop/tail and Reset on the demo computer. Two runs logged final_flush; another 60.499-second live run produced 53 text updates before Reset cancellation. |

## Measured results

Primary replay: `results/step2/2f87c46520844ed2bbd2be7d7be1fb4e.jsonl`, 125.000 audio seconds/2,000,000 samples and 129.430 server-session seconds. The microphone column uses local private session `c2984588c34041ed811ad83b730f689b`; its text is omitted from public evidence.

| Metric | Sustained replay (123 decodes) | Successful microphone run (21 decodes) |
|---|---:|---:|
| Audio duration | 125.000 s | 21.8613 s |
| Text updates | 123 | 21 |
| First-text event from session start | 3.836 s | 3.469 s |
| First-text event from RMS onset arrival | 2.399 s | 2.803 s |
| Inference median / P95 / max | 0.763 / 1.121 / 1.582 s | 0.639 / 0.764 / 0.764 s |
| Emission interval median / P95 | 1.041 / 1.156 s | 1.011 / 1.056 s |
| Lag median / P95 / max | 0.500 / 1.250 / 2.249 s | 0.500 / 0.750 / 1.499 s |
| Streaming compute load | 0.795 | 0.611 |
| Maximum stored audio | 20.249 s (<24 s) | 19.499 s (<24 s) |
| Buffer rollovers | 8 | 1 |
| Peak sampled RSS | 276,844,544 bytes | 277,561,344 bytes |
| Browser DOM acknowledgements | 123 | 21 |
| Emission-to-ack median / P95 | 0.142 / 0.264 s | 0.106 / 0.146 s |

The 60.499-second microphone run (`0bea7c0670b84605ad8df1d95deeefb7`) had 53 nonempty updates, median inference 0.646 s, maximum inference 2.640 s, maximum lag 2.500 s, three rollovers and maximum buffer 19.999 s. First/last 20 lag medians were both 0.5 s. It was cancelled through Reset, so it is not used as Stop/flush evidence. Another 22.4907-second microphone run also flushed successfully.

Some decodes exceeded the proposed one-second cadence. Capture continued; current audio was coalesced instead of queuing jobs. No indefinitely growing backlog was observed, and successful Stop returned lag to zero. The public reference contains “logam paduan ... sejumlah kecil elemen non-logam seperti karbon”; model output included substitutions such as “lokam penduan ... karpon.” Boundary/recognition errors remain visible. No formal WER/CER result or claim of complete omission/duplication prevention is made.

## Failures and corrective work

1. Step 1 ensurepip failure: preserved old environment and provisioned Python 3.12 plus environment-targeted pip.
2. Public dataset viewer HTTP 500/501 responses: used a capped stream of the official archive prefix to obtain one clip.
3. Initial focused tests could not write the sandbox's external temp directory: moved temporary logs under project results. Gradio mounting/localhost checks required the normal host process outside the restricted automation sandbox. Final 11 checks passed.
4. One real zero-audio restart logged `Cannot read properties of null (reading 'port')`: an old finished-session status could close the next AudioContext during preparation. Cleanup now detaches old resources synchronously before awaiting close, and ignores old session statuses until a new ID is acknowledged. Consecutive replay sessions and two deterministic controller regression checks passed after the fix. The failed private session `c99c38a9290d47b29394aa9d6ed3c46c` remains locally recorded.

## Limits and Step 3 readiness

Pass applies to this CPU configuration, one speaker/session, the observed microphone checks and bounded replay. Accuracy, noise robustness, full natural lectures and other browsers/hardware remain unestablished. Browser rendering opportunities were acknowledged; visible-display/per-word latency was not measured. A natural two-minute microphone demonstration is useful further validation, though the required sustained buffer check was exercised through paced browser audio.

The final agreement policy, full clean/noisy evaluation, training and five-user study remain deferred. Original assignment-form verification and lecturer administrative questions remain open. Microphone text stays local and is excluded from Git/evidence bundles; no raw recordings or model weights are committed.

**Ready for Step 3:** acquire and inspect the chosen speech/noise sources, verify licenses/metadata and official splits, and implement reproducible preprocessing. Step 2 did not start fine-tuning or the formal dataset workflow.
