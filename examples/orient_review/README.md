# 文本行方向复核工具（OCR 识别数据集版）

逐张复核方向模型挑出的「疑似上下颠倒」小图，人工确认后**原地旋转 180° 覆盖**，
用于清洗 line_marker 等 OCR **识别**数据集里方向反的样本（混进识别训练会让 loss 难收敛）。

> 与 `PaddleX/annotation_review` 的区别：那个改的是方向**分类**数据集的标签（txt 里的 0/180）；
> 这里数据集标签是**识别文本**，与方向无关，所以「确认」做的是**旋转图片像素**，
> 全程**不动** `train.txt`/`val.txt`。

## 工作流

```bash
cd plugins/vie-plugin-panel-label/examples/orient_review

# 1) 方向模型批量扫全量，挑出疑似反向的候选 -> candidates.json
#    --batch-size 控制批量推理加速；--score-thresh 控制候选严格度
python predict_candidates.py --batch-size 64 --score-thresh 0.7

# 2) 启动复核服务，浏览器开 http://127.0.0.1:5001
python review_tool.py
```

网页里：

- 每张候选显示缩略图 + 模型置信度（score）。
- 点 **「确认翻转 180°」** → 后台旋转该图并覆盖，缩略图随即刷新成翻转后的样子核对。
- 点 **「撤销」** → 从备份还原原图。
- 顶部可筛选 全部 / 仅未处理 / 仅已翻转；**「翻转当前可见的未处理项」** 批量一键翻转（核查后高效处理）。

## 安全 & 可回滚

- 每张被翻转的图，覆盖前会把**原图快照**备份到 `<images>/../orient_backup/`（`backup_once`：
  只备份第一次，保住最原始像素）。
- 复核进度记录在 `review_state.json`，重开工具不丢、断点续核。
- 识别标签与方向无关，`train.txt`/`val.txt` **无需也不会**改动。

## 配置

路径常量在 `review_io.py` 顶部：

- `IMAGES_DIR` — 待复核图片目录（默认指向 line_marker train/ocr/all/images）
- `BACKUP_DIR` — 原图备份目录（默认 `<images>/../orient_backup`）
- `CANDIDATES_JSON` / `STATE_JSON` — 候选文件 / 进度文件

`predict_candidates.py` 也支持 `--images-dir / --model-dir / --score-thresh / --batch-size / --out` 覆盖。
方向模型默认自动定位仓库内 `weights/panel_label/textline_ori_lcnet_v*`（最新版）。

## 结构

- `review_io.py` — 纯函数核心：旋转/备份/还原、候选与进度读写（可单测）
- `predict_candidates.py` — 方向模型批量扫描，生成候选（需模型权重，建议 GPU）
- `review_tool.py` — Flask 服务：画廊页 + `/`、`/image`、`/apply`、`/apply_batch`、`/undo`
- `test_review_io.py` — `review_io` 单测（`python -m pytest test_review_io.py -v`）

## 测试

```bash
cd orient_review && python -m pytest test_review_io.py -v
```
