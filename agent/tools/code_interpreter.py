import subprocess
import tempfile
from pathlib import Path

class CodeInterpreterTool:
    def __init__(self, python_path: str = "python"):
        self.python_path = python_path

    def as_mcp(self) -> dict:
        return {
            "name": "code_interpreter",
            "description": "Execute Python code in a sandboxed environment",
            "parameters": {"type": "string", "name": "code"},
            "call": self.run,
        }

    def run(self, code: str) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            p = subprocess.run(
                [self.python_path, "-I", "-S", "-"],
                input=code.encode(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=tmp,
                timeout=10,
            )
            return p.stdout.decode()
