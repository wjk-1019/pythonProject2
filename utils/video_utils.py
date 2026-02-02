"""
视频处理工具模块

功能：
1. 检查视频编码格式
2. 转码视频为浏览器兼容格式（H.264）
3. 视频流支持
"""

import os
import subprocess
import shutil
from typing import Optional, Dict, Tuple
import cv2


def get_video_codec(video_path: str) -> Optional[str]:
    """
    获取视频编码格式
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        FourCC编码字符串，如 'h264', 'FMP4', 'XVID' 等
    """
    if not os.path.exists(video_path):
        return None
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = ''.join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        cap.release()
        return codec
    except Exception as e:
        print(f"获取视频编码失败: {e}")
        return None


def get_video_info(video_path: str) -> Optional[Dict]:
    """
    获取视频详细信息
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        包含编码、帧率、尺寸等信息的字典
    """
    if not os.path.exists(video_path):
        return None
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = ''.join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        info = {
            'codec': codec,
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0,
            'file_size': os.path.getsize(video_path)
        }
        cap.release()
        return info
    except Exception as e:
        print(f"获取视频信息失败: {e}")
        return None


def is_browser_compatible(video_path: str) -> bool:
    """
    检查视频是否与浏览器兼容
    
    浏览器通常支持的编码：
    - H.264 (avc1, h264)
    - VP8/VP9 (在WebM容器中)
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        是否浏览器兼容
    """
    codec = get_video_codec(video_path)
    if not codec:
        return False
    
    # 浏览器兼容的编码
    compatible_codecs = ['h264', 'H264', 'avc1', 'AVC1', 'x264', 'X264']
    return codec in compatible_codecs


def check_ffmpeg_available() -> bool:
    """检查系统是否安装了ffmpeg"""
    return shutil.which('ffmpeg') is not None


def transcode_to_h264(input_path: str, output_path: Optional[str] = None, 
                      overwrite: bool = False) -> Tuple[bool, str]:
    """
    将视频转码为H.264格式（浏览器兼容）
    
    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径，默认为 {原文件名}_h264.mp4
        overwrite: 是否覆盖原文件
        
    Returns:
        (成功标志, 消息或输出路径)
    """
    if not check_ffmpeg_available():
        return False, "FFmpeg未安装，请先安装FFmpeg"
    
    if not os.path.exists(input_path):
        return False, f"输入文件不存在: {input_path}"
    
    # 生成输出路径
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_h264.mp4"
    
    # 如果已经是H.264，跳过转码
    if is_browser_compatible(input_path):
        return True, f"视频已是浏览器兼容格式，无需转码: {input_path}"
    
    try:
        # FFmpeg 转码命令
        # -c:v libx264 - 使用H.264编码
        # -preset fast - 快速预设（平衡速度和质量）
        # -crf 23 - 质量等级（18-28，越小质量越好）
        # -c:a aac - 音频使用AAC编码
        # -movflags +faststart - 支持流式播放
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            '-y',  # 覆盖输出文件
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode != 0:
            return False, f"转码失败: {result.stderr}"
        
        # 如果需要覆盖原文件
        if overwrite and output_path != input_path:
            backup_path = f"{input_path}.backup"
            os.rename(input_path, backup_path)
            os.rename(output_path, input_path)
            os.remove(backup_path)
            return True, input_path
        
        return True, output_path
        
    except subprocess.TimeoutExpired:
        return False, "转码超时"
    except Exception as e:
        return False, f"转码出错: {str(e)}"


def scan_videos_in_directory(data_dir: str) -> Dict[str, Dict]:
    """
    扫描目录中的所有视频文件，检查编码兼容性
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        {视频路径: {info, compatible}} 字典
    """
    results = {}
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV')
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(video_extensions):
                video_path = os.path.join(root, file)
                info = get_video_info(video_path)
                if info:
                    info['compatible'] = is_browser_compatible(video_path)
                    results[video_path] = info
    
    return results


def batch_transcode_directory(data_dir: str, dry_run: bool = True) -> Dict[str, str]:
    """
    批量转码目录中不兼容的视频
    
    Args:
        data_dir: 数据目录路径
        dry_run: 是否只检测不实际转码
        
    Returns:
        {视频路径: 状态信息} 字典
    """
    results = {}
    videos = scan_videos_in_directory(data_dir)
    
    incompatible_count = 0
    for path, info in videos.items():
        if not info.get('compatible', True):
            incompatible_count += 1
            if dry_run:
                results[path] = f"需要转码 (当前编码: {info.get('codec', 'unknown')})"
            else:
                success, msg = transcode_to_h264(path, overwrite=True)
                results[path] = "转码成功" if success else f"转码失败: {msg}"
    
    if dry_run:
        print(f"\n扫描完成: 共 {len(videos)} 个视频, {incompatible_count} 个需要转码")
    
    return results


if __name__ == "__main__":
    # 测试代码
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        print(f"检查视频: {video_path}")
        info = get_video_info(video_path)
        if info:
            print(f"  编码: {info['codec']}")
            print(f"  分辨率: {info['width']}x{info['height']}")
            print(f"  帧率: {info['fps']:.2f}")
            print(f"  时长: {info['duration']:.2f}秒")
            print(f"  文件大小: {info['file_size'] / 1024 / 1024:.2f}MB")
            print(f"  浏览器兼容: {'是' if is_browser_compatible(video_path) else '否'}")
        else:
            print("  无法读取视频信息")
    else:
        print("用法: python video_utils.py <视频文件路径>")
