"""
向量处理模块 - 使用官方 cn_clip 库
用于生成图像和文本的特征向量
"""
import torch
import numpy as np
from PIL import Image
from typing import List, Union, Tuple, Optional
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 导入配置
from config import CLIP_MODEL_NAME, DEVICE

# 全局变量，用于存储加载好的模型和预处理器，避免重复加载
_model = None
_preprocess = None
_tokenize = None
_device = None


def load_clip_model() -> Tuple[Optional[object], Optional[object]]:
    """
    加载 Chinese-CLIP 模型（使用官方 cn_clip 库）
    
    Returns:
        (模型, 预处理函数) 元组
    """
    global _model, _preprocess, _tokenize, _device
    
    if _model is None:
        try:
            import cn_clip.clip as clip
            from cn_clip.clip import load_from_name, available_models
            
            _device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # 使用 ViT-H-14 模型（1024维向量）
            model_name = "ViT-H-14"
            print(f"正在加载 cn_clip 模型 '{model_name}'...")
            print(f"可用模型: {available_models()}")
            
            _model, _preprocess = load_from_name(
                model_name, 
                device=_device, 
                download_root='./models'
            )
            _model.eval()
            
            # 获取 tokenize 函数
            _tokenize = clip.tokenize
            
            print(f"✓ cn_clip 模型加载成功，设备: {_device}")
            
        except Exception as e:
            print(f"错误: 加载 cn_clip 模型失败: {e}")
            print("请确保已安装 cn_clip: pip install cn_clip")
            raise
    
    return _model, _preprocess


def get_vector_dimension() -> int:
    """获取 CLIP 模型的向量维度"""
    # ViT-H-14 的维度是 1024
    return 1024


def get_image_vector(image: Image.Image) -> Union[List[float], None]:
    """
    将 PIL Image 对象转换为向量
    
    Args:
        image: PIL Image 对象
        
    Returns:
        正规化后的浮点列表，或 None
    """
    global _preprocess, _device
    model, preprocess = load_clip_model()
    
    if model is None or preprocess is None:
        return None

    try:
        # 确保图像是 RGB 模式
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # 使用 cn_clip 预处理
        image_input = preprocess(image).unsqueeze(0).to(_device)
        
        with torch.no_grad():
            # 使用 encode_image 获取图像特征
            image_features = model.encode_image(image_input)
            # L2 归一化
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # 转换为列表
        return image_features.cpu().numpy().astype(np.float32).tolist()[0]
    except Exception as e:
        print(f"生成图像向量错误: {e}")
        return None


def get_text_vector(text: str) -> Union[List[float], None]:
    """
    将文本字符串转换为向量
    
    Args:
        text: 文本字符串
        
    Returns:
        正规化后的浮点列表，或 None
    """
    global _tokenize, _device
    model, _ = load_clip_model()
    
    if model is None or _tokenize is None:
        return None

    try:
        # 使用 cn_clip 的 tokenize 函数
        text_input = _tokenize([text]).to(_device)
        
        with torch.no_grad():
            # 使用 encode_text 获取文本特征
            text_features = model.encode_text(text_input)
            # L2 归一化
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 转换为列表
        return text_features.cpu().numpy().astype(np.float32).tolist()[0]
    except Exception as e:
        print(f"生成文本向量错误: {e}")
        return None


def get_batch_image_vectors(images: List[Image.Image]) -> List[Union[List[float], None]]:
    """
    批量获取图像向量（提高效率）
    
    Args:
        images: PIL Image 对象列表
        
    Returns:
        向量列表
    """
    global _preprocess, _device
    model, preprocess = load_clip_model()
    
    if model is None or preprocess is None:
        return [None] * len(images)

    try:
        # 预处理所有图像
        image_inputs = []
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            image_inputs.append(preprocess(img))
        
        # 堆叠成批次
        image_batch = torch.stack(image_inputs).to(_device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_batch)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().astype(np.float32).tolist()
    except Exception as e:
        print(f"批量生成图像向量错误: {e}")
        return [None] * len(images)


def get_batch_text_vectors(texts: List[str]) -> List[Union[List[float], None]]:
    """
    批量获取文本向量（提高效率）
    
    Args:
        texts: 文本字符串列表
        
    Returns:
        向量列表
    """
    global _tokenize, _device
    model, _ = load_clip_model()
    
    if model is None or _tokenize is None:
        return [None] * len(texts)

    try:
        # 批量 tokenize
        text_inputs = _tokenize(texts).to(_device)
        
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy().astype(np.float32).tolist()
    except Exception as e:
        print(f"批量生成文本向量错误: {e}")
        return [None] * len(texts)


def compute_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    计算两个向量的余弦相似度
    
    Args:
        vec1: 向量1
        vec2: 向量2
        
    Returns:
        余弦相似度 (0-1)
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    
    # 由于向量已经 L2 归一化，直接计算点积即可
    similarity = np.dot(v1, v2)
    
    return float(similarity)
