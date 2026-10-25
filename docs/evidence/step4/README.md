# Actual Step 4 diagnostic evidence

Exact copies from local `results/step4/`. Start with `verification_summary.json`.
Every metric is diagnostic, not a full B/E1/E2 result or a final-test benchmark.
Original public FLEURS references and generated outputs are included for verification;
raw speech/noise, weights, optimizer tensors, caches and private microphone logs are not.

`latest_smoke.json`, `latest_overfit.json`, `latest_pilot.json` identify selected evidence.
Run folders preserve batch checks, parameter-update proofs, per-step loss/timing/RSS,
reference/raw/normalized prediction files, metrics and environment/configuration.
`uninterrupted_continuation.json` plus `reload_resume_verification.json` prove resume.
`data_integration_verification.json` covers all eligible train/validation audio and E2/fixed-noise contracts.
`small_validation_verification.json` is only two validation records per condition.
`all_python_checks.txt` contains the full regression result. `diagnostics.png` plots actual logs.

Two preliminary runs are retained: the first smoke misreported the EOS/cap flag because
Whisper's tensor-only return strips EOS; the first overfit lacked the continuation reference.
The selected runs use raw decoder sequences, with prefix/EOS retained, and a verified
complete-resume comparison. These development issues did not change labels or official
experiment hyperparameters. Transformers also emits an upstream internal suppression-token
deprecation warning; explicit language/task and one total generation-length bound are used.
The first broad regression invocation stalled while Gradio mounted in the sandbox; it was
stopped and rerun in the normal host environment. Keep these limits visible.

`snapshot_checksums.json` records exact report and source bytes at export. Colab/CUDA
and mixed precision have not been executed; the included notebook is an execution path.
