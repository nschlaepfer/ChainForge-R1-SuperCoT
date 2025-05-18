import subprocess

WHITELIST = {"ls", "/usr/bin/time"}

class ShellTool:
    def as_mcp(self) -> dict:
        return {
            "name": "shell",
            "description": "Execute whitelisted shell commands",
            "parameters": {"type": "string", "name": "command"},
            "call": self.run,
        }

    def run(self, command: str) -> str:
        cmd = command.split()
        if cmd[0] not in WHITELIST:
            return "Command not allowed"
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=5)
        return res.stdout.decode()
