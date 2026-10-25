Set-Location -LiteralPath $PSScriptRoot
$env:GRADIO_TEMP_DIR = Join-Path $PSScriptRoot '.cache\gradio'
$env:GRADIO_ANALYTICS_ENABLED = 'False'
$env:HF_HUB_OFFLINE = '1'
& (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') -u -m app.main
