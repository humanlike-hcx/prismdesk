"""对三路由识别执行可复现的 LLM / Embedding / 融合消融评测。

仅用 validation 集选择融合权重和澄清阈值；holdout 集只在参数冻结后运行。
运行前从项目根目录的 .env 读取模型和 Embedding 配置：
    python -m evaluation.route_ablation --suite validation
"""
import argparse
import asyncio
import json
import os
import pathlib
from collections import Counter
from typing import Any, Dict, List, Sequence

from dotenv import load_dotenv

from core.intent_recognizer import IntentRecognizer, RouteCategory
from evaluation.evaluator import ROUTE_BLIND_CASES, ROUTE_DEV_CASES, ROUTE_VALIDATION_CASES, RouteTestCase


def _recognizer() -> IntentRecognizer:
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("未设置 ANTHROPIC_API_KEY")
    return IntentRecognizer(
        api_key=key,
        base_url=os.getenv("ANTHROPIC_BASE_URL", "").strip() or None,
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022").strip(),
        embedding_api_key=os.getenv("EMBEDDING_API_KEY", "").strip(),
        embedding_base_url=os.getenv("EMBEDDING_BASE_URL", "").strip(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "").strip(),
        embedding_timeout_s=float(os.getenv("EMBEDDING_TIMEOUT_S", "20")),
    )


def _fuse(llm: Dict[str, Any], embedding: Dict[str, Any], llm_weight: float) -> tuple[str, float, float]:
    """与生产路由相同的置信度加权规则，用于离线比较候选权重。"""
    if llm.get("failed"):
        confidence = float(embedding.get("confidence", 0.0) or 0.0)
        return str(embedding.get("route", RouteCategory.GENERAL).value), confidence, confidence

    weighted: Dict[str, float] = {}
    for result, weight in ((llm, llm_weight), (embedding, 1.0 - llm_weight)):
        route = result.get("route", RouteCategory.GENERAL)
        route_name = route.value if isinstance(route, RouteCategory) else str(route)
        weighted[route_name] = weighted.get(route_name, 0.0) + weight * float(result.get("confidence", 0.0) or 0.0)
    ranked = sorted(weighted.items(), key=lambda item: item[1], reverse=True)
    confidence = ranked[0][1]
    margin = confidence - (ranked[1][1] if len(ranked) > 1 else 0.0)
    return ranked[0][0], confidence, margin


async def _collect(cases: Sequence[RouteTestCase], concurrency: int) -> List[Dict[str, Any]]:
    recognizer = _recognizer()
    # 先加载一次模板，避免并发任务重复初始化远端向量。
    await recognizer._load_template_embeddings()
    gate = asyncio.Semaphore(concurrency)

    async def one(case: RouteTestCase) -> Dict[str, Any]:
        async with gate:
            llm, embedding = await asyncio.gather(
                recognizer._llm_recognize(case.message, None),
                recognizer._embedding_recognize(case.message),
            )
        return {"id": case.test_id, "expected": case.expected_route, "llm": llm, "embedding": embedding}

    return await asyncio.gather(*(one(case) for case in cases))


def _summary(rows: Sequence[Dict[str, Any]], llm_weight: float | None, clarification_margin: float) -> Dict[str, Any]:
    predictions = []
    margins = []
    failures = 0
    for row in rows:
        if llm_weight is None:
            result = row["llm"]
            prediction = result.get("route", RouteCategory.GENERAL).value
            confidence = float(result.get("confidence", 0.0) or 0.0)
            margin = confidence
            failures += int(bool(result.get("failed")))
        elif llm_weight < 0:
            result = row["embedding"]
            prediction = result.get("route", RouteCategory.GENERAL).value
            confidence = float(result.get("confidence", 0.0) or 0.0)
            margin = confidence
        else:
            prediction, confidence, margin = _fuse(row["llm"], row["embedding"], llm_weight)
            failures += int(bool(row["llm"].get("failed")))
        predictions.append((prediction, row["expected"], confidence))
        margins.append(margin)
    correct = sum(prediction == expected for prediction, expected, _ in predictions)
    decided = [(prediction, expected) for (prediction, expected, _), margin in zip(predictions, margins) if margin >= clarification_margin]
    return {
        "accuracy": round(correct / len(rows), 4),
        "correct": correct,
        "total": len(rows),
        "llm_failures": failures,
        "clarification_margin": clarification_margin,
        "decided_coverage": round(len(decided) / len(rows), 4),
        "decided_accuracy": round(sum(prediction == expected for prediction, expected in decided) / len(decided), 4) if decided else None,
        "confusion": dict(Counter(f"{expected}->{prediction}" for prediction, expected, _ in predictions if prediction != expected)),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("dev", "validation", "blind"), default="validation")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--clarification-margin", type=float, default=0.12)
    parser.add_argument("--output", type=pathlib.Path, help="可选的 JSON 结果文件")
    args = parser.parse_args()
    load_dotenv()
    suites = {"dev": ROUTE_DEV_CASES, "validation": ROUTE_VALIDATION_CASES, "blind": ROUTE_BLIND_CASES}
    rows = await _collect(suites[args.suite], max(1, args.concurrency))
    report = {
        "suite": args.suite,
        "cases": len(rows),
        "clarification_margin": args.clarification_margin,
        "llm": _summary(rows, None, args.clarification_margin),
        "embedding": _summary(rows, -1, args.clarification_margin),
        "fusion_llm_weight": {
            str(weight): _summary(rows, weight, args.clarification_margin)
            for weight in (0.25, 0.5, 0.75)
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    asyncio.run(main())
