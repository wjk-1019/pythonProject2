"""
视频兼容性检查和批量转码脚本

功能：
1. 扫描data目录下所有视频文件
2. 检查视频编码是否与浏览器兼容（H.264）
3. 可选：将不兼容的视频转码为H.264格式

用法：
    # 仅检查，不转码
    python check_video_compatibility.py
    
    # 检查并转码不兼容的视频
    python check_video_compatibility.py --transcode
    
    # 检查特定视频文件
    python check_video_compatibility.py --file <视频路径>
"""

import os
# 设置环境变量，解决 OpenMP 库冲突问题（必须在导入其他库之前设置）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import argparse
from typing import List, Tuple

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.video_utils import (
    get_video_info,
    is_browser_compatible,
    check_ffmpeg_available,
    transcode_to_h264,
    scan_videos_in_directory
)
from config import DATA_DIR


def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / 1024 / 1024:.1f}MB"
    else:
        return f"{size_bytes / 1024 / 1024 / 1024:.2f}GB"


def format_duration(seconds: float) -> str:
    """格式化时长"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{int(seconds // 60)}分{int(seconds % 60)}秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}时{minutes}分"


def check_single_video(video_path: str) -> Tuple[bool, dict]:
    """检查单个视频文件"""
    if not os.path.exists(video_path):
        print(f"❌ 文件不存在: {video_path}")
        return False, {}
    
    info = get_video_info(video_path)
    if not info:
        print(f"❌ 无法读取视频信息: {video_path}")
        return False, {}
    
    compatible = is_browser_compatible(video_path)
    
    print(f"\n📹 视频: {os.path.basename(video_path)}")
    print(f"   路径: {video_path}")
    print(f"   编码: {info['codec']}")
    print(f"   分辨率: {info['width']}x{info['height']}")
    print(f"   帧率: {info['fps']:.2f} fps")
    print(f"   时长: {format_duration(info['duration'])}")
    print(f"   大小: {format_size(info['file_size'])}")
    print(f"   浏览器兼容: {'✅ 是' if compatible else '❌ 否'}")
    
    return compatible, info


def scan_and_report(data_dir: str) -> Tuple[List[str], List[str]]:
    """扫描目录并生成报告"""
    print(f"\n🔍 扫描目录: {data_dir}")
    print("=" * 60)
    
    videos = scan_videos_in_directory(data_dir)
    
    if not videos:
        print("未找到任何视频文件")
        return [], []
    
    compatible_videos = []
    incompatible_videos = []
    
    total_size = 0
    incompatible_size = 0
    
    for path, info in sorted(videos.items()):
        total_size += info.get('file_size', 0)
        if info.get('compatible', True):
            compatible_videos.append(path)
        else:
            incompatible_videos.append(path)
            incompatible_size += info.get('file_size', 0)
    
    # 打印摘要
    print(f"\n📊 扫描结果摘要:")
    print(f"   视频总数: {len(videos)}")
    print(f"   兼容视频: {len(compatible_videos)} ✅")
    print(f"   不兼容视频: {len(incompatible_videos)} ❌")
    print(f"   总文件大小: {format_size(total_size)}")
    
    if incompatible_videos:
        print(f"\n⚠️ 不兼容视频列表 ({len(incompatible_videos)} 个):")
        for path in incompatible_videos:
            info = videos[path]
            print(f"   - {os.path.relpath(path, data_dir)}")
            print(f"     编码: {info.get('codec', 'unknown')}, 大小: {format_size(info.get('file_size', 0))}")
        
        print(f"\n   不兼容视频总大小: {format_size(incompatible_size)}")
        print(f"\n💡 提示: 使用 --transcode 参数转码不兼容的视频")
    else:
        print("\n✅ 所有视频都与浏览器兼容!")
    
    return compatible_videos, incompatible_videos


def transcode_videos(video_list: List[str], overwrite: bool = True) -> int:
    """转码视频列表"""
    if not check_ffmpeg_available():
        print("\n❌ 错误: FFmpeg未安装")
        print("请从 https://ffmpeg.org/download.html 下载并安装FFmpeg")
        print("Windows用户可以使用: winget install ffmpeg")
        return 0
    
    print(f"\n🔄 开始转码 {len(video_list)} 个视频...")
    print("=" * 60)
    
    success_count = 0
    for i, video_path in enumerate(video_list, 1):
        print(f"\n[{i}/{len(video_list)}] 转码: {os.path.basename(video_path)}")
        
        success, msg = transcode_to_h264(video_path, overwrite=overwrite)
        if success:
            print(f"   ✅ 成功: {msg}")
            success_count += 1
        else:
            print(f"   ❌ 失败: {msg}")
    
    print(f"\n📊 转码完成: {success_count}/{len(video_list)} 成功")
    return success_count


def main():
    parser = argparse.ArgumentParser(
        description="视频兼容性检查和转码工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 扫描所有视频并显示报告
    python check_video_compatibility.py
    
    # 转码所有不兼容的视频
    python check_video_compatibility.py --transcode
    
    # 检查单个视频文件
    python check_video_compatibility.py --file data/video.mp4
        """
    )
    
    parser.add_argument(
        '--transcode', '-t',
        action='store_true',
        help='转码不兼容的视频为H.264格式'
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='检查单个视频文件'
    )
    
    parser.add_argument(
        '--dir', '-d',
        type=str,
        default=DATA_DIR,
        help=f'指定数据目录 (默认: {DATA_DIR})'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📹 视频兼容性检查工具")
    print("=" * 60)
    
    # 检查单个文件
    if args.file:
        compatible, info = check_single_video(args.file)
        if not compatible and args.transcode:
            print("\n🔄 开始转码...")
            success, msg = transcode_to_h264(args.file, overwrite=True)
            print(f"   {'✅ 成功' if success else '❌ 失败'}: {msg}")
        return
    
    # 扫描目录
    compatible, incompatible = scan_and_report(args.dir)
    
    # 转码
    if args.transcode and incompatible:
        confirm = input(f"\n确认转码 {len(incompatible)} 个视频? (y/N): ")
        if confirm.lower() == 'y':
            transcode_videos(incompatible)
        else:
            print("已取消转码")


if __name__ == "__main__":
    main()
