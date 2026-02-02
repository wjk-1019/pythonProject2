# 多模态行人检索系统

基于 Chinese-CLIP 和 Qwen2-VL 的多模态行人检索与评估系统，支持文本搜图和以图搜图功能。

---

## 🎯 项目功能

1. **文本描述生成**：使用 Qwen2-VL-2B 为行人图片生成详细中文描述
2. **向量检索**：基于 Chinese-CLIP 的跨模态向量检索
3. **对齐评估**：计算 Recall@K 指标评估图文对齐效果
4. **Web API**：FastAPI 后端提供搜索接口

---

## 📊 当前数据集

- **图片数量**: 3936 张行人图片
- **数据来源**: Avenue, VIRAT, MOT16, Normal_Videos 等多个数据集
- **数据格式**: 支持两种格式（日期格式和 _frames 格式）

---

## 🚀 快速开始

### 1. 环境配置

```powershell
# 激活 Conda 环境
conda activate multimodel

# 如需重新安装依赖
pip install -r requirements.txt
```

### 2. 生成描述并重建数据库

```powershell
# 测试模式 (30 张图片，约 20 分钟)
python regenerate_descriptions.py --device cpu --max_images 30

# 完整处理 (3936 张图片，约 40 小时)
python regenerate_descriptions.py --device cpu

# 使用 GPU 加速 (快 10-20 倍)
python regenerate_descriptions.py --device cuda
```

**输出文件**:
- `output/regenerated_pairs.json` - 图文对数据
- `my_lancedb/multimodal_data.lance` - 向量数据库

### 3. 启动 API 服务

```powershell
# 方法 1：使用启动脚本
.\start_api.ps1

# 方法 2：直接运行
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python api_server.py
```

**服务地址**: 
- API 服务: `http://localhost:8000`
- API 文档: `http://localhost:8000/docs`

---

## 📈 评估结果示例

基于 30 张测试图片的评估结果：

| 指标 | Recall@1 | Recall@5 | Recall@10 |
|------|----------|----------|-----------|
| **以图搜文** | 33.33% | 63.33% | 83.33% |
| **以文搜图** | 40.00% | 83.33% | 93.33% |

### 指标说明

- **Image-to-Text Recall@K**: 给定图片，正确文本在前 K 个检索结果中的概率
- **Text-to-Image Recall@K**: 给定文本，正确图片在前 K 个检索结果中的概率

---

## 🛠️ API 接口

### 1. 文本搜索图片

```powershell
# PowerShell 调用
$params = @{ query = "穿红色衣服的男性"; limit = 5 }
Invoke-RestMethod -Uri "http://localhost:8000/search_by_text" -Method Post -Body $params
```

**响应示例**:
```json
{
  "success": true,
  "query": "穿红色衣服的男性",
  "results": [
    {
      "image_url": "http://localhost:8000/data/.../image.jpg",
      "similarity": 71.74,
      "video_url": "http://localhost:8000/data/.../video.mp4"
    }
  ]
}
```

### 2. 其他接口

- `GET /health` - 健康检查
- `GET /system/info` - 系统信息
- `POST /search_by_image` - 图片搜索图片
- `GET /docs` - 交互式 API 文档

---

## 📁 项目结构

```
D:\pythonProject2/
├── api_server.py                    # FastAPI 后端服务器
├── regenerate_descriptions.py       # 描述生成 + 数据库重建
├── config.py                        # 配置文件
├── requirements.txt                 # Python 依赖
├── README.md                        # 项目说明
├── DEPLOYMENT_GUIDE.md              # 详细部署指南
│
├── core/                            # 核心模块
│   └── clip_model.py               # Chinese-CLIP 封装
│
├── utils/                           # 工具模块
│   ├── data_utils.py               # 数据加载
│   ├── lancedb_utils.py            # 向量数据库
│   └── vector_utils.py             # 向量计算
│
├── text_generation/                 # 文本生成模块
│   └── vlm_generator.py            # Qwen2-VL 描述生成器
│
├── evaluation/                      # 评估模块
│   └── recall_evaluator.py         # Recall@K 评估
│
├── data/                            # 数据集 (3936 张图片)
├── my_lancedb/                      # 向量数据库
├── output/                          # 输出目录
└── evaluation_results/              # 评估结果
```

---

## 🔬 技术方案

### 工作流程

```
行人图片 → Qwen2-VL 生成描述 → Chinese-CLIP 编码 → LanceDB 向量库 → API 检索
```

### 模型选择

| 任务 | 模型 | 说明 |
|------|------|------|
| 描述生成 | **Qwen2-VL-2B-Instruct** | 视觉语言模型 |
| 向量编码 | **Chinese-CLIP-ViT-H-14** | 跨模态检索模型 |
| 向量存储 | **LanceDB** | 向量数据库 |

---

## 📝 完整工作流示例

```powershell
# 步骤 1: 生成描述 (测试 30 张)
python regenerate_descriptions.py --device cpu --max_images 30

# 步骤 2: 启动 API 服务
python api_server.py

# 步骤 3: 测试搜索
$params = @{ query = "穿红色衣服的男性"; limit = 5 }
Invoke-RestMethod -Uri "http://localhost:8000/search_by_text" -Method Post -Body $params
```

---

## 🐛 常见问题

### 1. OpenMP 库冲突
**错误**: `OMP: Error #15`  
**解决**: 已在 `config.py` 中设置环境变量

### 2. 显存不足
**解决**: 使用 CPU 模式
```powershell
python regenerate_descriptions.py --device cpu
```

### 3. 数据库为空
**解决**: 先运行描述生成脚本
```powershell
python regenerate_descriptions.py --max_images 30
```

---

## 📚 详细文档

完整的部署和使用指南请参考：[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

---

## 💡 性能参考

| 配置 | 设备 | 处理速度 | 总耗时 (3936 张) |
|------|------|----------|------------------|
| Qwen2-VL-2B | CPU | ~40 秒/张 | ~40 小时 |
| Qwen2-VL-2B | GPU | ~4 秒/张 | ~4 小时 |

---

**最后更新**: 2026-01-31 | **版本**: 1.0.0
