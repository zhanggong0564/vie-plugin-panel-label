# Changelog

本插件变更记录遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)
规范，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

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
