import os
# 设置环境变量，解决 OpenMP 库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 获取当前脚本所在目录的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# LanceDB 配置
LANCEDB_DIR = os.path.join(BASE_DIR, "my_lancedb")  # LanceDB 数据库的存储目录
TABLE_NAME = "multimodal_data"  # LanceDB 表的名称

# Chinese-CLIP 模型配置
# 使用 chinese-clip-vit-huge-patch14 模型以获得最佳性能
CLIP_MODEL_NAME = "chinese-clip-vit-huge-patch14"
CLIP_MODEL_TYPE = "chinese-clip"  # 标识使用 Chinese-CLIP
CHINESE_CLIP_PRETRAIN = "CN-CLIP:ViT-H-14"  # 预训练权重标识

DEVICE = "cuda"

# 数据配置 - 使用绝对路径
DATA_DIR = os.path.join(BASE_DIR, "data")

# 注解和输出配置
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "annotations")

# 评测配置
EVALUATION_DIR = os.path.join(BASE_DIR, "evaluation_results")

# 模型配置
MODELS_DIR = os.path.join(BASE_DIR, "models")
