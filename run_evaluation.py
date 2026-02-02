"""
快速评估脚本 - 测试多规模召回率可视化
改进版本：使用多样性采样，获得更公平的评估结果
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import random
from pathlib import Path

def load_pairs_from_db():
    """从数据库加载图文对"""
    import lancedb
    
    db = lancedb.connect("my_lancedb")
    table = db.open_table("multimodal_data")
    df = table.to_pandas()
    
    pairs = []
    for _, row in df.iterrows():
        pairs.append({
            "id": f"img_{len(pairs):05d}",
            "image_path": row["image_path"],
            "text": row["text"]
        })
    
    return pairs


def run_multi_scale_evaluation(use_diverse_sampling: bool = True):
    """
    运行多规模评估
    
    Args:
        use_diverse_sampling: 是否使用多样性采样（去重+分组采样），提高召回率的公平性
    """
    from evaluation.recall_evaluator import AlignmentEvaluator
    
    print("=" * 60)
    print("多规模召回率评估")
    print(f"采样模式: {'多样性采样（去重+分组）' if use_diverse_sampling else '随机采样'}")
    print("=" * 60)
    
    # 加载数据
    print("\n加载数据...")
    pairs = load_pairs_from_db()
    print(f"✓ 加载 {len(pairs)} 条图文对")
    
    # 创建评估器
    evaluator = AlignmentEvaluator(device="cpu")
    
    # 定义测试规模 - 增加更多规模点
    scales = [10, 30, 50, 100, 200, 500, 1000, 2000, 3000]
    
    # 过滤超过总数的规模
    total = len(pairs)
    scales = [s for s in scales if s <= total]
    
    print(f"\n测试规模: {scales}")
    
    # 多规模评估（使用多样性采样）
    results_by_scale = evaluator.evaluate_at_different_scales(
        pairs,
        scales=scales,
        k_list=[1, 5, 10],
        use_diverse_sampling=use_diverse_sampling
    )
    
    # 保存结果
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    suffix = "_diverse" if use_diverse_sampling else "_random"
    evaluator.save_results(
        results_by_scale, 
        str(output_dir / f"multi_scale_results{suffix}.json")
    )
    
    # 生成可视化报告
    report_path = evaluator.generate_scale_report(
        results_by_scale,
        str(output_dir / f"scale_report{suffix}.html")
    )
    
    print(f"\n✅ 评估完成!")
    print(f"   结果文件: evaluation_results/multi_scale_results{suffix}.json")
    print(f"   可视化报告: {report_path}")
    print(f"\n打开浏览器查看报告: start {report_path}")
    
    return results_by_scale


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="多规模召回率评估")
    parser.add_argument("--random", action="store_true", 
                        help="使用随机采样而非多样性采样")
    args = parser.parse_args()
    
    run_multi_scale_evaluation(use_diverse_sampling=not args.random)
