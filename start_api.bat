@echo off
REM 启动多模态检索 API 服务器
echo ========================================
echo 正在启动多模态检索 API 服务器...
echo ========================================

REM 激活conda环境
call conda activate multimodel

REM 启动服务
python api_server.py

pause
