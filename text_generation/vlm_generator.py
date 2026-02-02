"""
VLM图像描述生成器
使用独立的视觉语言模型为行人图片生成中文描述
确保与Chinese-CLIP完全独立，保证评估公正性

支持模型：
- Qwen2-VL (推荐，中文最好)
- BLIP-2 (备选，更小更稳定)
"""
import torch
from PIL import Image
from typing import List, Optional
import os
from pathlib import Path
from tqdm import tqdm
import json


class VLMGenerator:
    """视觉语言模型描述生成器"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct", device: str = "cuda"):
        """
        初始化VLM
        Args:
            model_name: 模型名称，支持:
                - "Qwen/Qwen2-VL-7B-Instruct" (推荐，中文效果最好)
                - "Qwen/Qwen2-VL-2B-Instruct" (轻量版)
                - "blip2" (备选，更小更稳定)
            device: 运行设备
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self.model_type = "qwen" if "Qwen" in model_name else "blip2"
        
        print(f"正在加载VLM模型: {model_name}")
        self._load_model()
        print(f"模型加载完成，设备: {device}")
    
    def _load_model(self):
        """加载模型"""
        if self.model_type == "qwen":
            self._load_qwen()
        else:
            self._load_blip2()
    
    def _load_qwen(self):
        """加载Qwen2-VL模型"""
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
    
    def _load_blip2(self):
        """加载BLIP-2模型（备选方案）"""
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        
        model_id = "Salesforce/blip2-opt-2.7b"
        self.processor = Blip2Processor.from_pretrained(model_id)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
    
    def generate(self, image: Image.Image, prompt: str = None, max_tokens: int = 200) -> str:
        """
        生成图像描述
        """
        if self.model_type == "qwen":
            return self._generate_qwen(image, prompt, max_tokens)
        else:
            return self._generate_blip2(image, prompt, max_tokens)
    
    def _generate_qwen(self, image: Image.Image, prompt: str = None, max_tokens: int = 200) -> str:
        """使用Qwen2-VL生成"""
        from qwen_vl_utils import process_vision_info
        
        if prompt is None:
            prompt = (
                "请用一句话简洁描述这个行人的外观特征，包括："
                "性别、年龄段、上衣颜色和类型、裤子颜色和类型、发型、配饰。"
                "直接描述，不要分点。"
            )
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        
        output_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        return response.strip()
    
    def _generate_blip2(self, image: Image.Image, prompt: str = None, max_tokens: int = 200) -> str:
        """使用BLIP-2生成（英文，然后翻译）"""
        if prompt is None:
            prompt = "Describe this pedestrian in detail, including gender, age, clothing colors and types, hairstyle."
        
        inputs = self.processor(image, prompt, return_tensors="pt").to(self.device, torch.float16)
        
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        
        response = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return response.strip()
    
    def batch_generate(
        self,
        image_dir: str,
        output_file: str,
        max_images: int = None,
        prompt: str = None
    ) -> List[dict]:
        """
        批量生成图片描述并保存
        Args:
            image_dir: 图片目录
            output_file: 输出JSON文件路径
            max_images: 最大处理数量（用于测试）
            prompt: 自定义提示词
        Returns:
            图片-文本对列表
        """
        # 查找所有图片
        image_dir = Path(image_dir)
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = []
        for ext in extensions:
            image_paths.extend(list(image_dir.rglob(f"*{ext}")))
            image_paths.extend(list(image_dir.rglob(f"*{ext.upper()}")))
        image_paths = sorted(list(set(image_paths)))
        
        if max_images:
            image_paths = image_paths[:max_images]
        
        print(f"找到 {len(image_paths)} 张图片")
        
        # 生成描述
        results = []
        for i, img_path in enumerate(tqdm(image_paths, desc="生成描述")):
            try:
                image = Image.open(img_path).convert("RGB")
                desc = self.generate(image, prompt)
                results.append({
                    "id": f"ped_{i:05d}",
                    "image_path": str(img_path),
                    "text": desc
                })
            except Exception as e:
                print(f"\n警告: {img_path} 处理失败: {e}")
                results.append({
                    "id": f"ped_{i:05d}",
                    "image_path": str(img_path),
                    "text": "[生成失败]"
                })
        
        # 保存结果
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n已保存 {len(results)} 对图片-文本到: {output_file}")
        return results


def main():
    """命令行入口"""
    import argparse
    parser = argparse.ArgumentParser(description="行人图片描述生成")
    parser.add_argument("--image_dir", type=str, required=True, help="图片目录")
    parser.add_argument("--output", type=str, default="output/image_text_pairs.json", help="输出文件")
    parser.add_argument("--max_images", type=int, default=None, help="最大处理数（测试用）")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-7B-Instruct", help="VLM模型")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    args = parser.parse_args()
    
    generator = VLMGenerator(model_name=args.model, device=args.device)
    generator.batch_generate(args.image_dir, args.output, args.max_images)


if __name__ == "__main__":
    main()
