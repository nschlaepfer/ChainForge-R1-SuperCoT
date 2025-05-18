import time
from agent.main import chat

prompts = ["2+2?", "Tell me a joke.", "Summarize the plot of Romeo and Juliet."]

for p in prompts:
    start = time.time()
    ans = chat(p)
    print(f"Prompt: {p}\nAnswer: {ans.strip()}\nElapsed: {time.time()-start:.2f}s\n")
