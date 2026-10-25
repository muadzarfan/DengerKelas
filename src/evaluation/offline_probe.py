"""Actual model load/decode with Python network connections denied locally."""
import json
import os
import socket
import time
import soundfile as sf
from ..streaming.config import ROOT, load_settings


def denied(*args, **kwargs):
    raise RuntimeError("Network connections disabled for this offline inference probe")


if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"]="1"
    socket.socket.connect=denied
    socket.socket.connect_ex=denied
    socket.create_connection=denied
    from ..streaming.runtime import LocalWhisper
    start=time.perf_counter()
    backend=LocalWhisper(load_settings())
    loaded=time.perf_counter()
    x,rate=sf.read(ROOT/"data/fixtures/fleurs_id_sample.wav",dtype="float32")
    assert rate==16000
    words=backend.decode(x,0)
    result={"guard":"Python socket connect/connect_ex/create_connection denied; no system networking changes",
            "model_load_seconds":loaded-start,"inference_seconds":time.perf_counter()-loaded,
            "audio_seconds":len(x)/rate,"text":"".join(w.text for w in words),
            "runtime":backend.metadata()}
    path=ROOT/"results/step2/offline_probe.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
