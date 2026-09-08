"""让纯单元测试可在未安装外部 SDK 的最小环境中收集。"""
import sys
import types


try:
    import anthropic  # noqa: F401
except ModuleNotFoundError:
    module = types.ModuleType("anthropic")

    class AsyncAnthropic:  # pragma: no cover - 仅供无依赖环境导入
        def __init__(self, **kwargs):
            self.messages = None

    module.AsyncAnthropic = AsyncAnthropic
    sys.modules["anthropic"] = module
