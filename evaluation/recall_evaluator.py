"""
Chinese-CLIP 多模态对齐评估（使用 cn_clip 库）
计算 Recall@K 指标：以图搜文、以文搜图
支持不同数据集规模的召回率可视化

改进版本：
1. 使用多样性采样，避免选择重复或高度相似的文本
2. 支持去重评估模式
3. 更公平地评估图文对齐质量
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
import json
from PIL import Image
from tqdm import tqdm
from collections import Counter
import random


class AlignmentEvaluator:
    """Chinese-CLIP对齐效果评估器（使用 cn_clip）"""
    
    def __init__(self, device: str = "cuda"):
        """初始化，加载Chinese-CLIP模型"""
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self.tokenize = None
        
        print("加载 cn_clip 模型...")
        self._load_clip()
        print("cn_clip 加载完成")
    
    def _load_clip(self):
        """加载cn_clip"""
        import cn_clip.clip as clip
        from cn_clip.clip import load_from_name
        
        self.model, self.preprocess = load_from_name(
            "ViT-H-14", 
            device=self.device, 
            download_root='./models'
        )
        self.model.eval()
        self.tokenize = clip.tokenize
    
    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        """编码图片为embedding"""
        features = []
        batch_size = 16
        for i in tqdm(range(0, len(images), batch_size), desc="编码图片"):
            batch_imgs = images[i:i+batch_size]
            batch_inputs = []
            for img in batch_imgs:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                batch_inputs.append(self.preprocess(img))
            
            image_input = torch.stack(batch_inputs).to(self.device)
            with torch.no_grad():
                feat = self.model.encode_image(image_input)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            features.append(feat.cpu().numpy())
        return np.vstack(features)
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """编码文本为embedding"""
        features = []
        batch_size = 32
        for i in tqdm(range(0, len(texts), batch_size), desc="编码文本"):
            batch = texts[i:i+batch_size]
            text_inputs = self.tokenize(batch).to(self.device)
            
            with torch.no_grad():
                feat = self.model.encode_text(text_inputs)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            features.append(feat.cpu().numpy())
        return np.vstack(features)
    
    def compute_recall(self, query_feats: np.ndarray, gallery_feats: np.ndarray, k_list: List[int] = [1, 5, 10]) -> Dict[str, float]:
        """
        计算Recall@K
        Args:
            query_feats: 查询特征 (N, D)
            gallery_feats: 库特征 (N, D)
            k_list: K值列表
        Returns:
            {"recall@1": 0.xx, "recall@5": 0.xx, ...}
        """
        # 余弦相似度
        similarity = np.matmul(query_feats, gallery_feats.T)
        n = similarity.shape[0]
        
        # 排序获取top-k索引
        sorted_indices = np.argsort(-similarity, axis=1)
        
        results = {}
        for k in k_list:
            k = min(k, n)
            correct = sum(1 for i in range(n) if i in sorted_indices[i, :k])
            results[f"recall@{k}"] = correct / n
        
        return results
    
    def sample_diverse_pairs(
        self, 
        pairs: List[dict], 
        target_size: int,
        text_similarity_threshold: float = 0.85
    ) -> List[dict]:
        """
        多样性采样：选择文本差异较大的图文对
        避免选择重复或高度相似的文本
        
        Args:
            pairs: 原始图文对列表
            target_size: 目标采样数量
            text_similarity_threshold: 文本相似度阈值，超过此值被认为相似
            
        Returns:
            采样后的图文对列表
        """
        if target_size >= len(pairs):
            return pairs
        
        # 首先按文本去重
        text_to_pairs = {}
        for p in pairs:
            text = p['text']
            if text not in text_to_pairs:
                text_to_pairs[text] = p
        
        unique_pairs = list(text_to_pairs.values())
        print(f"  去重后唯一文本: {len(unique_pairs)} / {len(pairs)}")
        
        if target_size >= len(unique_pairs):
            # 如果目标数量大于唯一文本数，随机补充
            result = unique_pairs.copy()
            remaining = target_size - len(unique_pairs)
            if remaining > 0:
                duplicates = [p for p in pairs if p['text'] in text_to_pairs and p != text_to_pairs[p['text']]]
                if duplicates:
                    result.extend(random.sample(duplicates, min(remaining, len(duplicates))))
            return result[:target_size]
        
        # 使用贪婪多样性采样
        # 随机打乱后取样，优先选择不同文本开头的
        random.shuffle(unique_pairs)
        
        # 按文本前30字符分组
        prefix_groups = {}
        for p in unique_pairs:
            prefix = p['text'][:30]
            if prefix not in prefix_groups:
                prefix_groups[prefix] = []
            prefix_groups[prefix].append(p)
        
        # 轮流从每个组中选取
        result = []
        group_iterators = {k: iter(v) for k, v in prefix_groups.items()}
        keys = list(group_iterators.keys())
        random.shuffle(keys)
        
        while len(result) < target_size and group_iterators:
            for key in list(keys):
                if len(result) >= target_size:
                    break
                try:
                    result.append(next(group_iterators[key]))
                except StopIteration:
                    keys.remove(key)
        
        return result[:target_size]
    
    def evaluate(
        self,
        image_text_pairs: List[dict],
        k_list: List[int] = [1, 5, 10]
    ) -> Dict[str, Dict[str, float]]:
        """
        完整评估
        Args:
            image_text_pairs: [{"id": "ped_00000", "image_path": "...", "text": "..."}]
            k_list: K值列表
        Returns:
            评估结果字典
        """
        print(f"\n开始评估，共 {len(image_text_pairs)} 对数据")
        
        # 加载图片
        images = []
        texts = []
        for pair in image_text_pairs:
            img = Image.open(pair["image_path"])
            images.append(img)
            texts.append(pair["text"])
        
        # 编码
        print("\n[1/3] 编码图片...")
        image_feats = self.encode_images(images)
        
        print("\n[2/3] 编码文本...")
        text_feats = self.encode_texts(texts)
        
        # 计算Recall
        print("\n[3/3] 计算Recall@K...")
        results = {}
        
        # 以图搜文 (Image-to-Text)
        results["image_to_text"] = self.compute_recall(image_feats, text_feats, k_list)
        
        # 以文搜图 (Text-to-Image)
        results["text_to_image"] = self.compute_recall(text_feats, image_feats, k_list)
        
        return results
    
    def evaluate_at_different_scales(
        self,
        image_text_pairs: List[dict],
        scales: List[int] = [10, 50, 100, 500, 1000],
        k_list: List[int] = [1, 5, 10],
        use_diverse_sampling: bool = True
    ) -> Dict[int, Dict[str, Dict[str, float]]]:
        """
        在不同数据规模下评估召回率
        
        Args:
            image_text_pairs: 完整的图文对列表
            scales: 要测试的数据规模列表
            k_list: K值列表
            use_diverse_sampling: 是否使用多样性采样（去重+分组采样）
            
        Returns:
            {规模: {"image_to_text": {...}, "text_to_image": {...}}}
        """
        total = len(image_text_pairs)
        results_by_scale = {}
        
        print(f"\n{'='*60}")
        print(f"多规模评估: 总数据量 {total}")
        print(f"测试规模: {scales}")
        print(f"采样模式: {'多样性采样' if use_diverse_sampling else '随机采样'}")
        print(f"{'='*60}")
        
        for scale in scales:
            if scale > total:
                print(f"\n⚠ 规模 {scale} 超过总数据量 {total}，跳过")
                continue
            
            print(f"\n{'='*40}")
            print(f"评估规模: {scale}")
            print(f"{'='*40}")
            
            # 采样
            if use_diverse_sampling:
                sampled = self.sample_diverse_pairs(image_text_pairs, scale)
            else:
                sampled = random.sample(image_text_pairs, scale)
            
            # 评估
            results = self.evaluate(sampled, k_list)
            results_by_scale[scale] = results
            
            # 打印结果
            self.print_results(results, scale)
        
        return results_by_scale
    
    def print_results(self, results: Dict, scale: Optional[int] = None):
        """打印评估结果"""
        title = f"Chinese-CLIP 对齐评估结果"
        if scale:
            title += f" (规模: {scale})"
        
        print("\n" + "=" * 50)
        print(title)
        print("=" * 50)
        for task, metrics in results.items():
            print(f"\n【{task}】")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f} ({v*100:.2f}%)")
        print("=" * 50)
    
    def save_results(self, results: Dict, output_path: str):
        """保存结果到JSON"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {output_path}")
    
    def generate_scale_report(
        self,
        results_by_scale: Dict[int, Dict],
        output_path: str = "evaluation_results/scale_report.html"
    ):
        """
        生成规模与召回率的可视化HTML报告
        
        Args:
            results_by_scale: 不同规模的评估结果
            output_path: 输出HTML文件路径
        """
        scales = sorted(results_by_scale.keys())
        
        # 提取数据
        i2t_r1 = [results_by_scale[s]["image_to_text"]["recall@1"] * 100 for s in scales]
        i2t_r5 = [results_by_scale[s]["image_to_text"]["recall@5"] * 100 for s in scales]
        i2t_r10 = [results_by_scale[s]["image_to_text"]["recall@10"] * 100 for s in scales]
        
        t2i_r1 = [results_by_scale[s]["text_to_image"]["recall@1"] * 100 for s in scales]
        t2i_r5 = [results_by_scale[s]["text_to_image"]["recall@5"] * 100 for s in scales]
        t2i_r10 = [results_by_scale[s]["text_to_image"]["recall@10"] * 100 for s in scales]
        
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chinese-CLIP 召回率评估报告</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
        .chart-container {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px 15px;
            text-align: center;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .metric-good {{ color: #4CAF50; font-weight: bold; }}
        .metric-medium {{ color: #FF9800; font-weight: bold; }}
        .metric-poor {{ color: #f44336; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>🎯 Chinese-CLIP 召回率评估报告</h1>
    
    <div class="chart-container">
        <h2>📊 以图搜文 (Image-to-Text) 召回率</h2>
        <canvas id="i2tChart"></canvas>
    </div>
    
    <div class="chart-container">
        <h2>📊 以文搜图 (Text-to-Image) 召回率</h2>
        <canvas id="t2iChart"></canvas>
    </div>
    
    <h2>📋 详细数据表格</h2>
    <table>
        <thead>
            <tr>
                <th>数据规模</th>
                <th>I2T R@1</th>
                <th>I2T R@5</th>
                <th>I2T R@10</th>
                <th>T2I R@1</th>
                <th>T2I R@5</th>
                <th>T2I R@10</th>
            </tr>
        </thead>
        <tbody>
            {"".join(f'''
            <tr>
                <td><strong>{s}</strong></td>
                <td class="{_get_metric_class(i2t_r1[i])}">{i2t_r1[i]:.2f}%</td>
                <td class="{_get_metric_class(i2t_r5[i])}">{i2t_r5[i]:.2f}%</td>
                <td class="{_get_metric_class(i2t_r10[i])}">{i2t_r10[i]:.2f}%</td>
                <td class="{_get_metric_class(t2i_r1[i])}">{t2i_r1[i]:.2f}%</td>
                <td class="{_get_metric_class(t2i_r5[i])}">{t2i_r5[i]:.2f}%</td>
                <td class="{_get_metric_class(t2i_r10[i])}">{t2i_r10[i]:.2f}%</td>
            </tr>
            ''' for i, s in enumerate(scales))}
        </tbody>
    </table>
    
    <script>
        const scales = {json.dumps(scales)};
        
        // 以图搜文图表
        new Chart(document.getElementById('i2tChart'), {{
            type: 'line',
            data: {{
                labels: scales,
                datasets: [
                    {{
                        label: 'Recall@1',
                        data: {json.dumps(i2t_r1)},
                        borderColor: '#4CAF50',
                        backgroundColor: 'rgba(76, 175, 80, 0.1)',
                        fill: true,
                        tension: 0.3
                    }},
                    {{
                        label: 'Recall@5',
                        data: {json.dumps(i2t_r5)},
                        borderColor: '#2196F3',
                        backgroundColor: 'rgba(33, 150, 243, 0.1)',
                        fill: true,
                        tension: 0.3
                    }},
                    {{
                        label: 'Recall@10',
                        data: {json.dumps(i2t_r10)},
                        borderColor: '#FF9800',
                        backgroundColor: 'rgba(255, 152, 0, 0.1)',
                        fill: true,
                        tension: 0.3
                    }}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: '以图搜文召回率随数据规模变化'
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: '数据集规模'
                        }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: '召回率 (%)'
                        }},
                        min: 0,
                        max: 100
                    }}
                }}
            }}
        }});
        
        // 以文搜图图表
        new Chart(document.getElementById('t2iChart'), {{
            type: 'line',
            data: {{
                labels: scales,
                datasets: [
                    {{
                        label: 'Recall@1',
                        data: {json.dumps(t2i_r1)},
                        borderColor: '#4CAF50',
                        backgroundColor: 'rgba(76, 175, 80, 0.1)',
                        fill: true,
                        tension: 0.3
                    }},
                    {{
                        label: 'Recall@5',
                        data: {json.dumps(t2i_r5)},
                        borderColor: '#2196F3',
                        backgroundColor: 'rgba(33, 150, 243, 0.1)',
                        fill: true,
                        tension: 0.3
                    }},
                    {{
                        label: 'Recall@10',
                        data: {json.dumps(t2i_r10)},
                        borderColor: '#FF9800',
                        backgroundColor: 'rgba(255, 152, 0, 0.1)',
                        fill: true,
                        tension: 0.3
                    }}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: '以文搜图召回率随数据规模变化'
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: '数据集规模'
                        }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: '召回率 (%)'
                        }},
                        min: 0,
                        max: 100
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n✅ 可视化报告已生成: {output_path}")
        return output_path


def _get_metric_class(value: float) -> str:
    """根据召回率值返回CSS类"""
    if value >= 70:
        return "metric-good"
    elif value >= 40:
        return "metric-medium"
    else:
        return "metric-poor"


def main():
    """命令行入口"""
    import argparse
    parser = argparse.ArgumentParser(description="Chinese-CLIP对齐评估")
    parser.add_argument("--pairs_file", type=str, default="output/regenerated_pairs.json", help="图片-文本对JSON文件")
    parser.add_argument("--output", type=str, default="evaluation_results/evaluation_results.json", help="输出文件")
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 5, 10], help="K值列表")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--multi_scale", action="store_true", help="进行多规模评估")
    parser.add_argument("--scales", type=int, nargs="+", default=[10, 50, 100, 500, 1000, 2000], help="测试规模列表")
    args = parser.parse_args()
    
    # 加载数据
    with open(args.pairs_file, 'r', encoding='utf-8') as f:
        pairs = json.load(f)
    
    # 评估
    evaluator = AlignmentEvaluator(device=args.device)
    
    if args.multi_scale:
        # 多规模评估
        results_by_scale = evaluator.evaluate_at_different_scales(
            pairs, 
            scales=args.scales,
            k_list=args.k_values
        )
        
        # 保存结果
        evaluator.save_results(results_by_scale, args.output)
        
        # 生成可视化报告
        evaluator.generate_scale_report(
            results_by_scale,
            "evaluation_results/scale_report.html"
        )
    else:
        # 单次评估
        results = evaluator.evaluate(pairs, args.k_values)
        evaluator.print_results(results)
        evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()
