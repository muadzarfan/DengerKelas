import json
from pathlib import Path
import numpy as np


def distribution(values):
    if not values:
        return {"n": 0}
    result = {"n": len(values), "median": float(np.median(values)), "max": max(values)}
    if len(values) >= 20:
        result["p95"] = float(np.percentile(values, 95))
    else:
        result["individual"] = values
    return result


def summarize(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    ends = [r for r in rows if r["event"] == "inference_finish"]
    updates = [r for r in rows if r["event"] == "transcript_emitted"]
    text_updates = [r for r in updates if r["history"] or r["partial"]]
    first_voice = next((r["t"] for r in rows if r["event"] == "first_energy_onset"), None)
    finish = next((r for r in reversed(rows) if r["event"] == "session_finished"), {})
    durations = [r["duration"] for r in ends if not r["gated"]]
    total_audio = finish.get("audio_seconds", 0)
    emits = [r["t"] for r in text_updates]
    acknowledgements = [r for r in rows if r["event"] == "browser_render_ack"]
    lags = [r["lag"] for r in ends]
    chunks = [r for r in rows if r["event"] == "chunk_arrival"]
    return {"log": str(path), "audio_seconds": total_audio, "elapsed_seconds": finish.get("t"),
            "source": rows[0].get("source"), "first_text_from_session_start_s": emits[0] if emits else None,
            "first_text_from_energy_onset_s": emits[0] - first_voice if emits and first_voice is not None else None,
            "inference_seconds": distribution(durations),
            "server_emission_interval_s": distribution([b-a for a,b in zip(emits, emits[1:])]),
            "processing_lag_s": distribution(lags),
            "first_20_lag_median_s": float(np.median(lags[:20])) if lags else None,
            "last_20_lag_median_s": float(np.median(lags[-20:])) if lags else None,
            "chunk_count": len(chunks), "audio_sample_count": round(total_audio*16000),
            "chunk_arrival_span_s": chunks[-1]["t"]-chunks[0]["t"] if len(chunks)>1 else None,
            "streaming_compute_load": sum(durations)/total_audio if total_audio else None,
            "rss_bytes": distribution([r["rss"] for r in ends]),
            "rollovers": sum(r["event"] == "buffer_rollover" for r in rows),
            "max_buffer_samples": finish.get("max_buffer_samples"),
            "max_buffer_seconds": finish.get("max_buffer_samples", 0)/16000,
            "inference_job_queue": 0,
            "text_updates": len(text_updates), "browser_render_ack_count": len(acknowledgements),
            "browser_emission_to_ack_upper_bound_s": distribution([r["emission_to_ack_s"] for r in acknowledgements if "emission_to_ack_s" in r]),
            "error": finish.get("error"), "cancelled": finish.get("cancelled"),
            "final_text": ((text_updates[-1]["history"] + " " + text_updates[-1]["partial"]).strip() if text_updates else ""),
            "notes": "Server emission is not visible UI latency. Browser ack after requestAnimationFrame is only a rendering opportunity/round-trip upper bound. Replays are not real microphone verification."}
