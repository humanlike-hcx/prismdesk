# 三路业务路由设计

## 设计结论

入口不再把问候、投诉、转人工、退款、发票、登录失败等放进同一个互斥标签集合。它只输出稳定的业务路由：

| 路由 | 处理范围 | 主 Agent |
| --- | --- | --- |
| `general` | 订单、物流、账户、常规咨询、反馈与投诉 | GeneralAgent |
| `technical` | 登录、验证码、报错、崩溃、功能异常、配置 | TechnicalAgent |
| `billing` | 支付、扣款、退款、发票、订阅金额 | BillingAgent |

每次识别还会输出两个可组合字段：`handoff_requested` 表示用户明确请求人工处理；`needs_clarification` 表示三路由置信度不足且消息不是简单寒暄。它们不改变业务域，因此“转人工处理退款”会保留 `billing + handoff_requested=true`，不会丢失退款信息。

## 路由与 RAG

`technical` 和 `billing` 在主链路中预取知识库，因为回答通常依赖产品说明、故障排查规则或售后政策。`handoff_requested=true` 时禁止预取，交接节点只整理已知会话信息。`general` 使用 `optional`，由通用 Agent 根据上下文决定是否调用知识库。

## 评测边界

默认开发回归集是 `evaluation/datasets/route_dev_v1.jsonl`：三类各 8 条，不复用 Few-shot 模板。它用于检测代码和路由规则回归，不用于宣称总体准确率。

正式评测应冻结路由定义后，独立构造并人工复核开发集、验证集和盲测集；同一语义簇不能跨集合。盲测集每个最终路由至少保留 10 至 15 条，并报告样本数、Accuracy、Macro-F1、逐类 Precision/Recall/F1 和混淆矩阵。对于“是否转人工”和“是否需要澄清”，应分别以二分类指标报告，而不是混入路由准确率。
