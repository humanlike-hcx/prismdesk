# 评测说明

`datasets/route_dev_v1.jsonl` 是三路业务路由的快速回归集：`general`、`technical`、`billing` 各 8 条，共 24 条。样本不复用 `core/intent_recognizer.py` 的 Few-shot 模板。

它用于发现开发回归，不足以证明“准确率提升 10%”。旧的 `intent_smoke_v1.jsonl` 保留为历史资料，不再由评测器读取。

`POST /eval/run` 可用 `route_sources` 做消融，例如：

```json
{
  "route_sources": ["embedding"],
  "dialog_cases": []
}
```

正式对外报告需要独立的、人工复核且未参与模板或阈值调整的盲测集。盲测应分别报告三路由的样本数、逐类 F1、混淆矩阵和置信区间，并将人工交接与澄清需求作为单独的二分类评测。
