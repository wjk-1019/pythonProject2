# 数据目录说明

本目录用于存放视频数据和提取的行人图片。

## 目录结构

```
data/
├── {video_name}_frames/
│   ├── {video_name}.mp4 或 .avi    # 原始视频文件
│   └── photos/
│       └── fullbody/
│           ├── track_0001_fullbody.jpg
│           ├── track_0002_fullbody.jpg
│           └── ...
```

## 数据来源

- **Avenue 数据集**: 街道监控视频
- **MOT16 数据集**: 多目标追踪数据集
- **VIRAT 数据集**: 公共场所行为识别数据集
- **KITTI 数据集**: 自动驾驶场景数据集
- **UCF-Crime 数据集**: 异常行为检测数据集

## 数据要求

- 视频格式: `.mp4`, `.avi`, `.mov`, `.mkv`
- 图片格式: `.jpg`, `.jpeg`, `.png`
- 视频编码: 推荐 H.264（浏览器兼容）
- 图片尺寸: 建议不小于 224x224

## 获取数据

由于数据集较大（约 36GB），不包含在 Git 仓库中。请按照以下方式获取：

1. **下载公开数据集**:
   - Avenue: http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html
   - MOT16: https://motchallenge.net/data/MOT16/
   - VIRAT: https://viratdata.org/
   - KITTI: http://www.cvlibs.net/datasets/kitti/

2. **提取行人图片**:
   ```bash
   # 使用目标检测和追踪算法提取行人全身照
   python extract_pedestrians.py --video data/{video_name}.mp4
   ```

3. **放置到正确位置**:
   ```bash
   data/
   └── {video_name}_frames/
       ├── {video_name}.mp4
       └── photos/fullbody/
   ```

## 注意事项

- ⚠️ 数据集仅用于学术研究，请遵守相应的使用协议
- ⚠️ 大型视频文件不会被 Git 追踪（已在 `.gitignore` 中排除）
- ⚠️ 如果需要共享数据，建议使用云存储（如 Google Drive、百度网盘）

## 数据统计

当前系统支持的数据规模：

- 视频数量: 417 个
- 图片数量: 3936 张
- 总数据大小: 约 36.55 GB
