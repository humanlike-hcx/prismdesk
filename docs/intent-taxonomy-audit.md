# 意图标签体系审计

## 当前状态

`IntentCategory` 将三种不同层级的信息放进了一个互斥分类器：

| 信息层级 | 当前标签 |
| --- | --- |
| 会话行为 | `greeting`、`feedback`、`complaint`、`escalation`、`human_handoff` |
| 宽泛事项 | `query`、`request`、`technical`、`billing`、`account` |
| 细粒度业务事项 | `order_status`、`logistics`、`refund`、`invoice`、`payment_issue`、`account_security`、`technical_login`、`technical_crash` |

这不是单层、互斥的意图体系。例如“登录失败”既是 `technical`，又是 `technical_login`；“退款”既是 `billing`，又是 `refund`；“投诉并转人工”同时包含 `complaint`、`escalation` 和 `human_handoff`。因此用单个最近模板做平铺分类时，重叠是标签定义导致的，不是单纯扩大模板可以解决的问题。

## 保持现有 API 时的最小规范

在不改动 API 返回字段的前提下，必须先为每个标签写入包含条件、排除条件和优先级：

| 标签 | 包含条件 | 排除条件 / 优先级 |
| --- | --- | --- |
| `human_handoff` | 用户明确要求真人、人工坐席或人工客服 | 优先于其他业务事项 |
| `escalation` | 用户明确要求主管、升级、复核或正式申诉 | 没有明确人工诉求时才使用 |
| `complaint` | 表达不满，但没有明确要求人工或升级 | 低于前两者 |
| 细粒度业务标签 | 明确出现对应业务任务 | 优先于 `technical`、`billing`、`account` |
| `technical`、`billing`、`account` | 已知业务领域，但无法归入任何细粒度事项 | 只作领域兜底 |
| `query`、`request` | 无法识别业务领域的通用咨询或操作请求 | 仅作最终兜底 |
| `other` | 与客服系统无关或无法判断 | 不作为任何业务类别的替代品 |

该规范能让平铺分类可运行，但不能消除多意图消息的信息损失。

## 推荐的下一版输出结构

将识别结果拆为可组合字段，而不是让一个标签承担全部含义：

```json
{
  "dialog_act": "handoff_request",
  "domain": "billing",
  "task": "refund",
  "urgency": "high"
}
```

例如“我要找人工处理退款”可同时保留人工交接需求和退款任务。路由器依据 `dialog_act` 决定是否升级，RAG 依据 `domain/task` 决定是否检索，业务 Agent 依据 `task` 决定处理流程。

## 合格的评测流程

1. 先冻结标签定义、优先级和模板；模板不得根据盲测错误再调整。
2. 从不同来源构造样本并人工复核：真实脱敏问题、独立改写、边界案例和无关问题。
3. 按意图分层切分为开发集、验证集和盲测集；同一语义簇不得跨集合。
4. 盲测集至少覆盖每个最终路由标签 10 至 15 条，并报告样本数、Accuracy、Macro-F1、逐类 Precision/Recall/F1 和混淆矩阵。
5. 分别比较 Pattern、Embedding、LLM 和融合模式。每次对模板、阈值或标签优先级调整后，只看开发集与验证集；盲测仅在版本冻结后运行。

在完成上述流程前，任何 57 条 smoke 集上的数字都只能用于开发排错，不能作为项目总体准确率或性能提升结论。
