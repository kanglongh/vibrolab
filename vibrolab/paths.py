"""集中管理项目路径."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / 'data'
OUT  = ROOT / 'outputs'
CACHE = OUT / 'cache'
LLM_MODELS = ROOT / 'llm' / 'models'

for d in [OUT, CACHE]:
    d.mkdir(parents=True, exist_ok=True)
