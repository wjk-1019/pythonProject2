# PowerShell 启动脚本
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting the multimodal retrieval API server..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 激活conda环境
conda activate multimodel

# 启动服务
python api_server.py
