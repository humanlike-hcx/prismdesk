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

2026-09-08 的远端 Embedding 开发集运行记录：

| 模式 | 正确数 | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| 初始模板 + Qwen3-Embedding-4B | 33/57 | 57.89% | 0.5331 |
| 类别边界整理后 + Qwen3-Embedding-4B | 49/57 | 85.96% | 0.8535 |

这两次运行使用的是同一份已参与模板迭代的开发 smoke 集，不能用来宣称模型在未知数据上的提升。最终结论必须来自未参与调优的盲测集。
