import tomllib
import time
import os
from pathlib import Path
from llama_cpp import Llama
from qwen_agent import Assistant
from agent.tools import build_tools

CFG_PATH = os.environ.get("AGENT_CONFIG", Path(__file__).with_suffix('.toml'))
CFG = tomllib.loads(Path(CFG_PATH).read_text())

os.environ["OMP_NUM_THREADS"] = str(CFG["performance"]["threads"])

llm = Llama(
    model_path=CFG["model"]["gguf_path"],
    n_ctx=CFG["model"]["context_length"],
    n_gpu_layers=CFG["performance"]["gpu_layers"],
    temperature=CFG["model"]["temperature"],
    top_p=CFG["model"]["top_p"],
    top_k=CFG["model"]["top_k"],
    logits_all=False,
)

assistant = Assistant(
    llm=llm,
    function_list=build_tools(CFG["tools"], CFG.get("paths", {})),
    enable_thinking=CFG["model"]["enable_thinking"],
)

def chat(prompt: str) -> str:
    start = time.perf_counter()
    rsp = assistant.chat(prompt)["content"]
    print(f"[{(time.perf_counter()-start):.2f}s]")
    return rsp

if __name__ == "__main__":
    while True:
        try:
            print(chat(input(">>> ")))
        except (EOFError, KeyboardInterrupt):
            break
