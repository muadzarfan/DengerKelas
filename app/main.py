import asyncio
import json
import os
from pathlib import Path
import time
import logging
from urllib.parse import urlparse
import numpy as np

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("GRADIO_TEMP_DIR", str(Path(__file__).resolve().parents[1] / ".cache/gradio"))

import gradio as gr
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import uvicorn
from src.streaming.config import ROOT, load_settings
from src.streaming.runtime import LocalWhisper
from src.streaming.session import Session


def create_app():
    settings = load_settings()
    backend, model_error = None, None
    try:
        backend = LocalWhisper(settings)
    except Exception as exc:
        model_error = str(exc)
        logging.getLogger(__name__).error("Local model load failed: %s", model_error)
    api = FastAPI()
    single_session = asyncio.Lock()

    @api.get("/step2/health")
    def health():
        return {"model_ready": backend is not None, "error": model_error,
                "runtime": backend.metadata() if backend else None}

    @api.get("/step2/fixture")
    def fixture():
        path = ROOT / "data/fixtures/fleurs_id_sample.wav"
        if not path.is_file():
            from fastapi import HTTPException
            raise HTTPException(404, "Run the single-fixture download command first")
        return FileResponse(path, media_type="audio/wav")

    @api.websocket("/step2/audio")
    async def audio_socket(socket: WebSocket):
        # Loopback binding plus an origin check avoids an unrelated web page
        # opening a microphone-processing session on this local service.
        origin = socket.headers.get("origin")
        if origin and urlparse(origin).hostname not in ("localhost", "127.0.0.1"):
            await socket.close(code=1008)
            return
        await socket.accept()
        session = None
        owns_lock = False
        send_lock = asyncio.Lock()

        async def send(value):
            async with send_lock:
                await socket.send_json(value)

        async def publish():
            while True:
                if session:
                    await send(session.snapshot())
                await asyncio.sleep(.25)

        async def dispose(cancel):
            nonlocal session, owns_lock
            if session:
                await asyncio.to_thread(session.close, cancel=cancel)
                session = None
            if owns_lock:
                single_session.release()
                owns_lock = False

        publisher = asyncio.create_task(publish())
        try:
            while True:
                message = await socket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                try:
                    if message.get("bytes") is not None:
                        if session is None:
                            raise RuntimeError("Start a session before sending audio")
                        payload = message["bytes"]
                        if not 0 < len(payload) <= 4 * 192000 or len(payload) % 4:
                            raise ValueError("Invalid PCM packet size")
                        # Web Audio sends little-endian float32, incremental mono.
                        session.push(np.frombuffer(payload, dtype="<f4"))
                        continue
                    command = json.loads(message.get("text") or "{}")
                    kind = command.get("type")
                    if kind == "start":
                        if backend is None:
                            raise RuntimeError("Model lokal belum siap. Periksa langkah unduh model di README, lalu jalankan ulang aplikasi.")
                        if session:
                            await dispose(cancel=True)
                        if single_session.locked():
                            raise RuntimeError("Sesi lain masih aktif. Hentikan sesi tersebut terlebih dahulu.")
                        await single_session.acquire()
                        owns_lock = True
                        session = Session(backend, settings, int(command["sample_rate"]),
                                          source=command.get("source", "browser_microphone"))
                        await send({"type": "started", "session": session.id})
                    elif kind == "stop" and session:
                        session.stop()
                        await send(session.snapshot())
                    elif kind == "reset":
                        await dispose(cancel=True)
                        await send({"type": "reset", "status": "Siap untuk sesi baru", "history": "", "partial": ""})
                    elif kind == "capture_error":
                        if session:
                            session.fail("Browser microphone: " + str(command.get("message")))
                        await send({"error": "Akses mikrofon gagal. Periksa izin browser dan pilihan mikrofon.", "done": True})
                    elif kind == "render_ack" and session and command.get("session") == session.id:
                        update_id = command.get("update_id")
                        with session.log_lock:
                            emitted = next((e["t"] for e in reversed(session.events)
                                            if e["event"] == "transcript_emitted" and e["update_id"] == update_id), None)
                        session.log("browser_render_ack", update_id=update_id,
                                    emission_to_ack_s=time.perf_counter()-session.started-emitted if emitted is not None else None,
                                    browser_ms=command.get("browser_ms"))
                except Exception as exc:
                    if session:
                        session.fail(str(exc))
                    await send({"error": str(exc), "done": True})
        except WebSocketDisconnect:
            pass
        finally:
            publisher.cancel()
            try:
                await publisher
            except (asyncio.CancelledError, RuntimeError):
                pass
            try:
                await dispose(cancel=True)
            except RuntimeError:
                # Keep exclusive ownership until the decoder actually exits.
                if session:
                    await asyncio.to_thread(session.worker.join)
                    await dispose(cancel=True)

    html = (ROOT / "app/interface.html").read_text(encoding="utf-8")
    javascript = (ROOT / "app/client.js").read_text(encoding="utf-8")
    with gr.Blocks(title="DengerKelas", analytics_enabled=False) as ui:
        gr.HTML(html, js_on_load=javascript)
    return gr.mount_gradio_app(api, ui, path="/", footer_links=[])


if __name__ == "__main__":
    uvicorn.run(create_app(), host="127.0.0.1", port=7860, ws="websockets-sansio",
                ws_max_size=768000, ws_max_queue=4)
