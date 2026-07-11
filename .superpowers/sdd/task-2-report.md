# Task 2 报告：简化直送 OCR 主流程

## 状态

- 完成提交：`65d76f1 refactor(panel_label_ocr): 简化直送OCR数据流`
- 代码改动严格限于 `vie_plugin_panel_label/panel_label_detect.py` 和新增测试
  `tests/test_direct_ocr_pipeline.py`。

## RED

命令：

```bash
PYTHONPATH=/home/zhanggong/workspace/VisInferEngine/mobile_vision:/home/zhanggong/workspace/VisInferEngine/mobile_vision/plugins/vie-plugin-panel-label/.worktrees/direct-ocr-cleanup \
  conda run -n ppocr python -m pytest tests/test_direct_ocr_pipeline.py -q
```

结果：`3 failed`。三个失败均为预期的 `AttributeError`，分别证明
`_orient_crops`、`_recognize_with_fallback`、`_extract_texts` 尚不存在。

## GREEN

提交后重新运行：

```bash
PYTHONPATH=/home/zhanggong/workspace/VisInferEngine/mobile_vision:/home/zhanggong/workspace/VisInferEngine/mobile_vision/plugins/vie-plugin-panel-label/.worktrees/direct-ocr-cleanup \
  conda run -n ppocr python -m pytest \
  tests/test_direct_ocr_pipeline.py tests/test_dedup.py \
  tests/test_panel_label_sort.py tests/test_rec_export.py -q
```

结果：`33 passed in 1.54s`。

## 实现与自审

- 构造函数签名已核对，与 brief 接口完全一致。
- 三个私有方法已实现；低置信反向识别仅在 `rec_score` 严格更高时替换。
- `text_crops` 使用列表流保持与 ROI 1:1 顺序，空 ROI 时输出空列表。
- 删除主流程中的文本检测参数、实例、裁剪器、映射结构和旧裁图类。
- 删除通配导入，显式导入 `cv2` 及实际依赖。
- `git diff HEAD^ --check` 通过；提交仅包含两个目标文件。

## 静态残留检查说明

目标生产文件与本任务新测试的残留检查无匹配：

```bash
rg -n "text_det_|text_det_points|text_det_model|OCRPipelineCrop|CropByPolys|TextDetection" \
  vie_plugin_panel_label/panel_label_detect.py tests/test_direct_ocr_pipeline.py
```

brief 中对 `vie_plugin_panel_label tests examples` 的全目录命令仍会命中本任务范围外的既有内容：

- `tests/test_auto_annotate.py`、`tests/test_ocr_dataset_converter.py`：独立数据工具仍合法使用文本检测/多边形裁剪。
- 若干测试的 PaddleOCR 导入桩仍定义旧依赖，以支持其被测模块。
- `tests/test_rec_export.py` 含 Task 1 删除字段的负向断言。

按“只改目标生产文件和新测试文件”的限制未修改上述既有文件。

## 审查修复（2026-07-11）

### 修复内容

- `tests/test_rec_export.py` 使用动态拼接的旧字段名执行负向断言，测试源码不再保留旧字段字面量。
- 新增 `infer()` 级假模型集成测试，覆盖混合 line/非 line 检测、ROI 排序重排、稀疏低置信方向 fallback、低分文字置 `None`，并验证 `Points/index/class_id/confidence/texts/text_crops` 与原检测结果 1:1 映射。
- 方向分类、首次文字识别及 fallback 文字识别均增加结果数量检查；数量与输入裁图不一致时显式抛出 `ValueError`，避免 `zip` 或索引流程静默截断。
- 未使用的 `roi_transforms` 接收变量改为 `_`。
- fallback 单测改用稀疏索引 `[0, 2]`，同时断言 fallback 模型的输入内容及结果写回位置。

### RED / GREEN

新增数量契约测试后运行：

```bash
env PYTHONPATH=/home/zhanggong/workspace/VisInferEngine/mobile_vision \
  conda run -n ppocr python -m pytest \
  tests/test_direct_ocr_pipeline.py tests/test_rec_export.py -q
```

首次结果：`3 failed, 7 passed`。其中两个预期失败分别证明方向和识别结果数量不等时未抛错；第三个失败是测试将 dataclass 的 `index` 列表误当成 ndarray，修正测试断言后不涉及生产行为。

实现保护并补齐 fallback 数量测试后，同命令结果：`11 passed in 1.64s`。

### 覆盖验证

```bash
env PYTHONPATH=/home/zhanggong/workspace/VisInferEngine/mobile_vision \
  conda run -n ppocr python -m pytest \
  tests/test_direct_ocr_pipeline.py tests/test_dedup.py \
  tests/test_panel_label_sort.py tests/test_rec_export.py -q
```

完整结果：`37 passed in 1.49s`。

尝试运行插件全量 `tests/`，收集阶段因本工作树中不存在既有测试依赖的 `scripts.convert_ocr_dataset_to_ppocr` 而中止：`1 error in 1.67s`。该模块属于独立数据转换工具，按任务边界未修改；本次覆盖测试均通过。

### 正式运行代码残留扫描

```bash
rg -n "text_det_|text_det_points|text_det_model|OCRPipelineCrop|CropByPolys|TextDetection" \
  vie_plugin_panel_label
```

完整结果：`0 matches`（`rg` 退出码 1，包装检查命令输出 `production residual scan: 0 matches` 并以退出码 0 完成）。
