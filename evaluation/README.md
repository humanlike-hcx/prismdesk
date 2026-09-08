# 评测说明

`datasets/intent_smoke_v1.jsonl` 是快速回归集，覆盖 19 个意图，每类 3 条，共 57 条。样本不复用 `core/intent_recognizer.py` 的 Few-shot 模板。

它用于发现明显退化，不足以证明“准确率提升 10%”。对外报告模型收益时，应另建至少 200 条、由人工复核且从未参与模板或阈值调整的盲测集，并同时报告样本数、逐类 F1、混淆矩阵和置信区间。

`POST /eval/run` 的 `intent_sources` 可控制消融模式，例如：

```json
{
  "intent_sources": ["embedding"],
  "dialog_cases": []
}
```

可分别运行 `pattern`、`embedding`、`llm` 和三路融合。只有在同一份冻结测试集上比较这些结果，才能把变化归因到对应模块。

远端 Embedding 接口已经完成连通性验证，但尚未记录可对外报告的性能数字。性能验证必须先冻结标签定义与模板，再在未参与调优的盲测集上运行。
