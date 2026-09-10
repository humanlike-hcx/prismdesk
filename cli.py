"""Interactive CLI client for the PrismDesk local demo."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


API_BASE = os.getenv("PRISMDESK_API_BASE", "http://127.0.0.1:8000")
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RED = "\033[91m"


def _configure_terminal() -> None:
    if sys.platform == "win32":
        os.system("")
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")


def _request(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
        method = "POST"
    request = urllib.request.Request(
        f"{API_BASE}{path}", data=data, headers=headers, method=method
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _banner() -> None:
    print(f"{CYAN}{BOLD}")
    print("  ┌──────────────────────────────────────────────┐")
    print("  │                  PrismDesk                   │")
    print("  │        Multi-Agent Customer Service CLI      │")
    print("  └──────────────────────────────────────────────┘")
    print(RESET)
    print(f"{DIM}输入问题开始对话。输入 /help 查看命令，/quit 退出。{RESET}\n")


def _route_details(result: dict[str, Any]) -> str:
    route = result.get("route", "general")
    agents = result.get("agent_types") or [result.get("agent_type", "general")]
    confidence = float(result.get("route_confidence") or result.get("routing_confidence") or 0)
    rag_policy = result.get("rag_policy", "optional")
    rag_state = "命中" if result.get("retrieval_success") else (
        "未命中" if result.get("retrieval_attempted") else "未调用"
    )
    tools = result.get("tools_used") or []
    latency = float(result.get("latency_ms") or 0) / 1000
    fields = [
        f"route={route}",
        f"agent={'+'.join(str(agent) for agent in agents)}",
        f"confidence={confidence:.1%}",
        f"rag={rag_policy}/{rag_state}",
        f"latency={latency:.2f}s",
    ]
    if tools:
        fields.append(f"tools={','.join(str(tool) for tool in tools)}")
    if result.get("escalated"):
        fields.append("escalated=true")
    return "  ".join(fields)


def _show_help() -> None:
    print(f"{YELLOW}可用命令{RESET}")
    print("  /new     开始一段新会话")
    print("  /status  查看后端与 Agent 状态")
    print("  /help    查看命令")
    print("  /quit    退出 PrismDesk\n")


def _show_status() -> None:
    try:
        health = _request("/health")
        agents = health.get("agents", {})
        print(f"{GREEN}服务正常{RESET}  API={API_BASE}")
        if isinstance(agents, dict):
            print(f"{DIM}Agents: {', '.join(agents.keys())}{RESET}\n")
    except Exception as exc:
        print(f"{RED}服务不可用：{exc}{RESET}\n")


def main() -> int:
    _configure_terminal()
    _banner()
    conversation_id: str | None = None
    user_id = "local-demo-user"

    while True:
        try:
            message = input(f"{BLUE}{BOLD}你 > {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}会话已结束。{RESET}")
            return 0

        if not message:
            continue
        command = message.lower()
        if command in {"/quit", "/exit", "quit", "exit"}:
            print(f"{DIM}会话已结束。{RESET}")
            return 0
        if command == "/help":
            _show_help()
            continue
        if command == "/new":
            conversation_id = None
            print(f"{GREEN}已开始新会话。{RESET}\n")
            continue
        if command == "/status":
            _show_status()
            continue

        print(f"{DIM}PrismDesk 正在处理…{RESET}", flush=True)
        try:
            result = _request(
                "/chat",
                {
                    "message": message,
                    "user_id": user_id,
                    "conv_id": conversation_id,
                },
            )
            conversation_id = result.get("conv_id") or conversation_id
            answer = result.get("response") or "未返回回答。"
            print(f"\n{CYAN}{BOLD}PrismDesk >{RESET} {answer}\n")
            print(f"{DIM}{_route_details(result)}{RESET}\n")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"{RED}请求失败（HTTP {exc.code}）：{detail}{RESET}\n")
        except Exception as exc:
            print(f"{RED}连接失败：{exc}{RESET}\n")


if __name__ == "__main__":
    raise SystemExit(main())
