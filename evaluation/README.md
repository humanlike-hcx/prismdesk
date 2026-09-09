# 评测说明

`datasets/route_dev_v1.jsonl` 是快速回归集，每类 8 条；`route_validation_v1.jsonl` 是参数选择用验证集，每类 20 条；`route_blind_v1.jsonl` 是参数冻结后才运行的留出集，每类 20 条。三者均不复用 Embedding 模板或 LLM Few-shot，且彼此不重复。该留出集由项目构造，不能称为真正的业务盲测；真实盲测应由未参与开发的人独立抽样和标注。

它用于发现开发回归，不足以证明“准确率提升 10%”。旧的 `intent_smoke_v1.jsonl` 保留为历史资料，不再由评测器读取。

`POST /eval/run` 可用 `route_sources` 做 LLM、Embedding 和融合消融，例如：

```json
{
  "route_suite": "validation",
  "route_sources": ["embedding"],
  "dialog_cases": []
}
```

Pattern 不参与业务路由投票，只负责显式人工交接、紧急度和实体提取。正式对外报告前，三份项目内构造样本仍需由业务人员人工复核或替换为脱敏真实问题；盲测应分别报告三路由的样本数、逐类 F1、混淆矩阵和置信区间，并将人工交接与澄清需求作为单独的二分类评测。

调用真实模型做消融并将结果写入本地文件：

```bash
python -m evaluation.route_ablation --suite validation --output evaluation/results/route_validation_live.json
```

评测脚本会分别报告 LLM、Embedding 与 LLM 权重为 0.25、0.5、0.75 的融合结果。仅依据 validation 选择参数；完成后再运行 `--suite blind`，结果文件被 Git 忽略。
