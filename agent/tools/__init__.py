from typing import Dict, List

from .code_interpreter import CodeInterpreterTool
from .shell import ShellTool
from .fetch import FetchTool


def build_tools(cfg: Dict[str, bool], paths: Dict[str, str]) -> List[dict]:
    tools = []
    if cfg.get("code_interpreter"):
        tools.append(CodeInterpreterTool(paths.get("python_tool", "python")))
    if cfg.get("shell"):
        tools.append(ShellTool())
    if cfg.get("search"):
        tools.append(FetchTool(paths.get("search_api_key", "")))
    return [t.as_mcp() for t in tools]

CODE_TESTS: str = """def test_add():\n    assert add(2,2) == 4\n"""


def run_code_variant(code: str, tests: str = CODE_TESTS):
    full_code = code + "\n" + tests + "\nif __name__ == '__main__':\n    import pytest, sys; sys.exit(pytest.main(['-q']))\n"
    tool = CodeInterpreterTool()
    output = tool.run(full_code)
    success = '1 passed' in output
    return success, 0.0
