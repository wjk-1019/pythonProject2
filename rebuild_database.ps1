# PowerShell 脚本：重建向量数据库
# 使用 Qwen2-VL 生成图片描述，Chinese-CLIP 生成向量

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "重建多模态检索向量数据库" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 设置环境变量
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# 激活 conda 环境
conda activate multimodel

# 参数说明：
# --device cpu  : 使用 CPU（显存不足时）
# --device cuda : 使用 GPU（推荐，速度更快）
# --max_images N : 限制处理图片数量（测试用，不指定则处理全部）

Write-Host ""
Write-Host "开始处理..." -ForegroundColor Yellow
Write-Host "数据集包含约 3969 张图片" -ForegroundColor Yellow
Write-Host "预计耗时：CPU 约 6-8 小时，GPU 约 1-2 小时" -ForegroundColor Yellow
Write-Host ""

# 运行重建脚本（默认使用 CPU）
# 如有 GPU，请将 cpu 改为 cuda
python regenerate_descriptions.py --device cpu

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "数据库重建完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "下一步：启动 API 服务器" -ForegroundColor Cyan
Write-Host "  .\start_api.ps1" -ForegroundColor White
