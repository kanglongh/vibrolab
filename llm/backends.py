"""
vibrolab.llm.backends — LLM 后端切换.

支持两种后端:
    local   本地 Qwen 系列 (transformers, CPU 可跑)
    api     任何 OpenAI-Compatible 服务 (DashScope / DeepSeek / 智谱 / Moonshot /
            OpenAI 官方 / vLLM 自部署 等), 通过 base_url + api_key + model 三件套配置

用法:
    from llm.backends import get_backend
    llm = get_backend()          # 自动根据环境变量选择
    text = llm.generate("你好")
    llm.last_usage               # 上一次调用的 token 消耗 (仅 api 后端)
"""
from __future__ import annotations

import os


class LocalQwenBackend:
    """本地 Qwen 系列 (通过 transformers 加载, 通过 QWEN_MODEL_PATH 指定路径)."""

    def __init__(self, model_path: str = None):
        model_path = model_path or os.environ.get('QWEN_MODEL_PATH')
        if not model_path:
            raise RuntimeError(
                "Local model path not set. Set env var QWEN_MODEL_PATH to point at "
                "your local Qwen (or compatible) model directory, or use the API "
                "backend (LLM_BACKEND=api, see .env.example). See llm/README.md for "
                "model download instructions."
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Local model not found at {model_path}. "
                f"Verify QWEN_MODEL_PATH points to a valid model directory, "
                f"or use API backend (LLM_BACKEND=api)."
            )
        print(f"[LLM] Loading local model from {model_path}...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,   # CPU 默认 fp32; 有 GPU 可换 bfloat16
            trust_remote_code=True,
        ).eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        self.last_usage = None
        self.total_tokens = 0
        self.total_cost_rmb = 0.0
        print(f"[LLM] Ready on {self.device}")

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.3) -> str:
        """用 chat template 生成回复."""
        import torch
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


class OpenAICompatibleBackend:
    """兼容 OpenAI Chat Completions 协议的通用后端.

    只要服务商暴露 /v1/chat/completions 端点即可接入 (阿里云百炼 / DeepSeek /
    智谱 / Moonshot / OpenAI 官方 / vLLM 自部署 ...).

    通过环境变量配置:
        LLM_API_KEY       — 服务商 API key
        LLM_BASE_URL      — API 基础 URL (示例见 .env.example)
        LLM_MODEL         — 模型 ID
        LLM_PRICE_IN      — [可选] 输入价格 (元 / 百万 token), 用于成本核算
        LLM_PRICE_OUT     — [可选] 输出价格 (元 / 百万 token)

    调用后通过 self.last_usage 读取上次调用的 token 与成本;
    self.total_tokens / self.total_cost_rmb 累计本 session 全部调用.
    """

    def __init__(self, model: str = None, api_key: str = None,
                 base_url: str = None,
                 price_in_per_million: float = None,
                 price_out_per_million: float = None):
        self.api_key  = api_key  or os.environ.get('LLM_API_KEY')
        self.base_url = base_url or os.environ.get('LLM_BASE_URL')
        self.model    = model    or os.environ.get('LLM_MODEL')
        if not self.api_key:
            raise RuntimeError("Env var LLM_API_KEY not set (see .env.example)")
        if not self.base_url:
            raise RuntimeError("Env var LLM_BASE_URL not set (see .env.example)")
        if not self.model:
            raise RuntimeError("Env var LLM_MODEL not set (see .env.example)")

        # 价格 (元 / 百万 token); 未设置则成本记 0, 不影响功能
        self.price_in  = price_in_per_million  if price_in_per_million  is not None \
                         else float(os.environ.get('LLM_PRICE_IN',  '0') or 0)
        self.price_out = price_out_per_million if price_out_per_million is not None \
                         else float(os.environ.get('LLM_PRICE_OUT', '0') or 0)

        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        self.last_usage = None       # dict: {input_tokens, output_tokens, cost_rmb}
        self.total_tokens = 0
        self.total_cost_rmb = 0.0

    def _price(self, input_tok: int, output_tok: int) -> float:
        return (input_tok / 1_000_000.0) * self.price_in \
             + (output_tok / 1_000_000.0) * self.price_out

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.3) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        u = resp.usage
        in_tok  = int(getattr(u, 'prompt_tokens',     0) or 0)
        out_tok = int(getattr(u, 'completion_tokens', 0) or 0)
        cost = self._price(in_tok, out_tok)
        self.last_usage = {
            'input_tokens':  in_tok,
            'output_tokens': out_tok,
            'total_tokens':  in_tok + out_tok,
            'cost_rmb':      cost,
        }
        self.total_tokens   += in_tok + out_tok
        self.total_cost_rmb += cost
        return resp.choices[0].message.content.strip()


_backend_singleton = None


def get_backend(backend: str = None):
    """获取 LLM 后端单例. backend='local' | 'api' | None (自动读 LLM_BACKEND 环境变量)."""
    global _backend_singleton
    if _backend_singleton is not None:
        return _backend_singleton
    if backend is None:
        backend = os.environ.get('LLM_BACKEND', 'local')
    if backend == 'local':
        _backend_singleton = LocalQwenBackend()
    elif backend == 'api':
        _backend_singleton = OpenAICompatibleBackend()
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return _backend_singleton
