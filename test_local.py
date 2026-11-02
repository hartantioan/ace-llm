"""Step-by-step check that ACE runs fully locally on Ollama (qwen3.5:9b + nomic-embed-text).

Run from the root of your ace-llm clone:   python test_local.py
Stops at the first failing step and tells you what to fix.
"""

import json
import os
import sys
import time
import urllib.request

from dotenv import load_dotenv

load_dotenv()
BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def step(n, title):
    print(f"\n[{n}] {title}")


def ok(msg):
    print(f"    PASS  {msg}")


def fail(msg, hint=""):
    print(f"    FAIL  {msg}")
    if hint:
        print(f"    HINT  {hint}")
    sys.exit(1)


# 1. Ollama server reachable and models installed
step(1, f"Ollama server at {BASE}")
try:
    with urllib.request.urlopen(f"{BASE}/api/tags", timeout=5) as r:
        installed = [m["name"] for m in json.load(r)["models"]]
except Exception as e:
    fail(f"cannot reach Ollama ({e})", "start it with `ollama serve` (or open the Ollama app)")
ok(f"installed: {installed}")

from ace.llm.embeddings import EmbeddingsEnum, create_embedding  # noqa: E402
from ace.llm.base import create_llm  # noqa: E402
from ace.llm.models import ModelsEnum  # noqa: E402

for needed in (ModelsEnum.QWEN_3_5_9B.value, EmbeddingsEnum.OLLAMA_NOMIC_EMBED_TEXT.value):
    if not any(n == needed or n.startswith(needed + ":") for n in installed):
        fail(f"'{needed}' not installed", f"run `ollama pull {needed}`, or fix the tag in models.py to match `ollama list`")
ok("both models present")

# 2. Plain chat call
step(2, "Chat model (create_llm -> ChatOllama)")
llm = create_llm(ModelsEnum.QWEN_3_5_9B, temperature=0.0)
t = time.time()
try:
    out = llm.invoke("Reply with exactly one word: pong")
except Exception as e:
    fail(f"chat call failed: {e}")
ok(f"{time.time() - t:.1f}s -> {out.content.strip()[:80]!r}")
if "<think>" in out.content or not out.content.strip():
    print("    WARN  thinking text/empty reply. Try create_llm(..., reasoning=False) in setup.py")

# 3. Embeddings
step(3, "Embeddings (create_embedding -> OllamaEmbeddings)")
emb = create_embedding(EmbeddingsEnum.OLLAMA_NOMIC_EMBED_TEXT)
try:
    vecs = emb.embed_documents(["send an email", "book a taxi"])
except Exception as e:
    fail(f"embedding call failed: {e}")
ok(f"{len(vecs)} vectors, dim={len(vecs[0])}")

# 4. Structured output (what AbstractPlanner needs: abstract.py:106)
step(4, "Structured output (with_structured_output)")
from pydantic import BaseModel  # noqa: E402


class Tool(BaseModel):
    name: str
    description: str


class Resp(BaseModel):
    tools: list[Tool]


try:
    res = llm.with_structured_output(Resp).invoke("List two tools an email assistant needs. Give name and description.")
except Exception as e:
    fail(f"structured output failed: {e}", "try reasoning=False, or raise num_ctx (e.g. num_ctx=8192) on ChatOllama")
ok(f"parsed {len(res.tools)} tools: {[t.name for t in res.tools]}")

# 5. Full agent on a bundled example query (ace.setup.setup_app)
step(5, "Full ACE agent end to end")
from ace.setup import AppConfig, setup_app  # noqa: E402

try:
    agent = setup_app(AppConfig())
except Exception as e:
    fail(f"setup_app failed: {e}")
ok("agent built (tool embeddings generated via FAISS)")
query = open("examples/queries/fare_estimation.txt").read().strip() if os.path.exists("examples/queries/fare_estimation.txt") else "Draft an email to my boss saying I will be late."
print(f"    query: {query[:100]!r}")
t = time.time()
try:
    agent.run_query(query)
except Exception as e:
    fail(f"run_query failed: {e}", "re-run `python -m ace -vv` and read ace.log for the failing chain")
ok(f"completed in {time.time() - t:.0f}s")

print("\nAll steps passed. Confirm it stayed local: `ollama ps` shows qwen3.5:9b loaded.")
