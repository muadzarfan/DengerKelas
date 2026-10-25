import json
import platform
import sys
import importlib.metadata
import winreg
import psutil
import ctranslate2
from ..streaming.config import ROOT


if __name__ == "__main__":
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
        cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    packages = {name: importlib.metadata.version(name) for name in
                ("numpy","scipy","av","soundfile","psutil","faster-whisper","ctranslate2","gradio","fastapi","uvicorn","websockets")}
    info={"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
          "cpu":cpu,"logical_cpu_count":psutil.cpu_count(), "physical_cores":psutil.cpu_count(logical=False),
          "ram_bytes":psutil.virtual_memory().total,"cuda_devices":ctranslate2.get_cuda_device_count(),
          "cpu_compute_types":sorted(ctranslate2.get_supported_compute_types("cpu")),"packages":packages}
    path=ROOT/"results/step2/environment.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(info,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(info,indent=2))
