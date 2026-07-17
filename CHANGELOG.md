# Changelog

本插件变更记录遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)
规范，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 变更

- 方向分类与 CTC 识别改为消费框架类型化结果；OCR pipeline 显式接收模型 metadata
  路径，模型对象不再持有或接收仅供 runner 加载的 ONNX 路径。
- 场景注册切换至框架 `ScenarioRegistry`，保持检测、方向仲裁、文字识别和 API
  响应行为不变。
- 检测、方向分类和文字识别 ONNX runner 改为由场景业务层统一创建并注入；
  三个子模型支持随服务关闭统一释放，初始化失败会回滚已创建资源。
- 面板标签分割检测默认模型由 YOLO `best.onnx` 切换至
  `rfdetr-seg-nano.onnx`；OCR、去重、排序和 API 响应契约保持不变。
- 新增默认开启的 `PANEL_LABEL_CPU_FAST_PATH`：RF-DETR 仅解码检测框局部 mask，
  ROI 展平仅在 polygon 紧致区域内分配和计算，保持文本、类别、顺序、置信度和
  位置反映射契约。
- `PANEL_LABEL_CPU_FAST_PATH=false` 可同时恢复完整 mask 和旧 ROI 展平路径；
  单候选局部处理失败时自动回退旧路径并限频记录 warning。

## [1.1.1] - 2026-07-14

### 修复

- 修复 XB3 两排线标使用默认线性排序导致识别顺序错误的问题。

### 变更

- 线标 OCR 的 ONNX CTC 识别改为按请求批量执行，避免逐个 ROI 调用推理后端。
- 线标 OCR 的 ONNX CTC 识别使用顺序执行模式，降低单路图调度开销。

## [1.1.0] - 2026-07-13

### 新增

- 新增 PP-LCNet 文本行方向分类与 PP-OCRv5 文字识别 ONNX 适配器。
- 新增 Paddle GPU 与 ONNX Runtime CUDA 的真实样本一致性测试。

### 变更

- 删除失效的文本检测配置、数据字段和裁图管线，将 ROI、文字与回流裁图
  收敛为 1:1 列表流。
- 方向分类和文字识别统一切换至 ONNX Runtime，移除插件运行时的
  PaddleOCR/PaddleX 推理依赖。
- 模型配置改为直接引用 `weights/panel_label/v2` 下的 ONNX 文件。
- 保持现有 API、返回字段、阈值、排序和低置信方向双向仲裁行为不变。
