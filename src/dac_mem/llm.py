"""
Unified LLM interface for DAC-Mem.

Providers: OpenAI (GPT-4o), Anthropic (Claude), Google (Gemini),
HuggingFace Transformers (Qwen, Llama), and vLLM for fast local inference.

Every provider exposes the same generate() / batch_generate() API so the
rest of DAC-Mem is provider-agnostic.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── abstract base ──────────────────────────────────────────────────────────

class BaseLLM(ABC):
    """Minimal interface every LLM backend must implement."""

    @abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 512) -> str: ...

    def batch_generate(self, prompts: List[str], system: Optional[str] = None,
                       temperature: float = 0.0, max_tokens: int = 512) -> List[str]:
        """Default sequential; backends like vLLM override for parallelism."""
        return [self.generate(p, system, temperature, max_tokens) for p in prompts]

    @property
    @abstractmethod
    def model_name(self) -> str: ...


# ── OpenAI ─────────────────────────────────────────────────────────────────

class OpenAILLM(BaseLLM):
    def __init__(self, model: str = 'gpt-4o',
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        import openai
        self._model = model
        kwargs: Dict[str, Any] = {}
        if api_key or os.getenv('OPENAI_API_KEY'):
            kwargs['api_key'] = api_key or os.getenv('OPENAI_API_KEY')
        if base_url:
            kwargs['base_url'] = base_url
        self.client = openai.OpenAI(**kwargs)

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, prompt: str, system: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 512) -> str:
        msgs: list = []
        if system:
            msgs.append({'role': 'system', 'content': system})
        msgs.append({'role': 'user', 'content': prompt})
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self._model, messages=msgs,
                    temperature=temperature, max_tokens=max_tokens,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        return ''


# ── Anthropic ──────────────────────────────────────────────────────────────

class AnthropicLLM(BaseLLM):
    def __init__(self, model: str = 'claude-sonnet-4-20250514',
                 api_key: Optional[str] = None):
        import anthropic
        self._model = model
        self.client = anthropic.Anthropic(
            api_key=api_key or os.getenv('ANTHROPIC_API_KEY'))

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, prompt: str, system: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 512) -> str:
        kw: Dict[str, Any] = dict(
            model=self._model, max_tokens=max_tokens,
            temperature=temperature,
            messages=[{'role': 'user', 'content': prompt}],
        )
        if system:
            kw['system'] = system
        for attempt in range(3):
            try:
                resp = self.client.messages.create(**kw)
                return resp.content[0].text.strip()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        return ''


# ── Google Gemini ──────────────────────────────────────────────────────────

class GoogleLLM(BaseLLM):
    def __init__(self, model: str = 'gemini-1.5-pro',
                 api_key: Optional[str] = None):
        import google.generativeai as genai
        genai.configure(api_key=api_key or os.getenv('GOOGLE_API_KEY'))
        self._model = model
        self.client = genai.GenerativeModel(model)

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, prompt: str, system: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 512) -> str:
        full = f"{system}\n\n{prompt}" if system else prompt
        resp = self.client.generate_content(
            full,
            generation_config={'temperature': temperature,
                               'max_output_tokens': max_tokens},
        )
        return resp.text.strip()


# ── HuggingFace Transformers (Qwen, Llama, …) ─────────────────────────────

class HuggingFaceLLM(BaseLLM):
    """Runs a chat model locally via transformers.  Works with
    Qwen2-7B-Instruct, Llama-3-8B-Instruct, etc."""

    def __init__(self, model: str = 'Qwen/Qwen2-7B-Instruct',
                 device: str = 'auto', torch_dtype: str = 'auto',
                 load_in_4bit: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._name = model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, trust_remote_code=True)
        load_kw: Dict[str, Any] = dict(
            torch_dtype=torch_dtype, device_map=device,
            trust_remote_code=True)
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kw['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(model, **load_kw)

    @property
    def model_name(self) -> str:
        return self._name

    def generate(self, prompt: str, system: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 512) -> str:
        msgs: list = []
        if system:
            msgs.append({'role': 'system', 'content': system})
        msgs.append({'role': 'user', 'content': prompt})
        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors='pt').to(self.model.device)
        gen_kw: Dict[str, Any] = dict(max_new_tokens=max_tokens,
                                       do_sample=temperature > 0)
        if temperature > 0:
            gen_kw['temperature'] = temperature
        out = self.model.generate(**inputs, **gen_kw)
        new_tok = out[0][inputs['input_ids'].shape[-1]:]
        return self.tokenizer.decode(new_tok, skip_special_tokens=True).strip()


# ── vLLM (high-throughput local inference) ─────────────────────────────────

class VLLMBackend(BaseLLM):
    """Uses the vllm library for fast batched inference on local GPUs."""

    def __init__(self, model: str = 'Qwen/Qwen2-72B-Instruct',
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.90):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        self._name = model
        self.llm = LLM(model=model,
                       tensor_parallel_size=tensor_parallel_size,
                       gpu_memory_utilization=gpu_memory_utilization,
                       trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, trust_remote_code=True)
        self.SamplingParams = SamplingParams

    @property
    def model_name(self) -> str:
        return self._name

    def _format(self, prompt: str, system: Optional[str] = None) -> str:
        msgs: list = []
        if system:
            msgs.append({'role': 'system', 'content': system})
        msgs.append({'role': 'user', 'content': prompt})
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)

    def generate(self, prompt: str, system: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 512) -> str:
        return self.batch_generate([prompt], system, temperature, max_tokens)[0]

    def batch_generate(self, prompts: List[str], system: Optional[str] = None,
                       temperature: float = 0.0, max_tokens: int = 512) -> List[str]:
        texts = [self._format(p, system) for p in prompts]
        params = self.SamplingParams(temperature=max(temperature, 0.01),
                                     max_tokens=max_tokens)
        outs = self.llm.generate(texts, params)
        return [o.outputs[0].text.strip() for o in outs]


# ── Disk cache ─────────────────────────────────────────────────────────────

class LLMCache:
    def __init__(self, cache_dir: str = '.cache/llm'):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, model: str, prompt: str, system: str,
              temperature: float) -> Path:
        h = hashlib.md5(
            f"{model}\n{system}\n{prompt}\n{temperature}".encode()
        ).hexdigest()
        return self.dir / f"{h}.json"

    def get(self, model: str, prompt: str, system: str = '',
            temperature: float = 0.0) -> Optional[str]:
        p = self._path(model, prompt, system, temperature)
        if p.exists():
            return json.loads(p.read_text())['response']
        return None

    def put(self, model: str, prompt: str, response: str,
            system: str = '', temperature: float = 0.0) -> None:
        p = self._path(model, prompt, system, temperature)
        p.write_text(json.dumps({
            'model': model, 'response': response,
            'prompt_head': prompt[:200],
        }, ensure_ascii=False))


class CachedLLM(BaseLLM):
    """Transparent caching wrapper — only caches deterministic (temp=0) calls."""
    def __init__(self, llm: BaseLLM, cache_dir: str = '.cache/llm'):
        self._llm = llm
        self.cache = LLMCache(cache_dir)

    @property
    def model_name(self) -> str:
        return self._llm.model_name

    def generate(self, prompt: str, system: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 512) -> str:
        if temperature == 0.0:
            hit = self.cache.get(self._llm.model_name, prompt, system or '')
            if hit is not None:
                return hit
        resp = self._llm.generate(prompt, system, temperature, max_tokens)
        if temperature == 0.0:
            self.cache.put(self._llm.model_name, prompt, resp, system or '')
        return resp

    def batch_generate(self, prompts: List[str], system: Optional[str] = None,
                       temperature: float = 0.0, max_tokens: int = 512) -> List[str]:
        return self._llm.batch_generate(prompts, system, temperature, max_tokens)


# ── Factory ────────────────────────────────────────────────────────────────

_DEFAULTS: Dict[str, str] = {
    'openai':     'gpt-4o',
    'anthropic':  'claude-sonnet-4-20250514',
    'google':     'gemini-1.5-pro',
    'huggingface':'Qwen/Qwen2-7B-Instruct',
    'qwen':       'Qwen/Qwen2-72B-Instruct',
    'qwen-small': 'Qwen/Qwen2-7B-Instruct',
    'llama':      'meta-llama/Meta-Llama-3-70B-Instruct',
    'llama-small':'meta-llama/Meta-Llama-3-8B-Instruct',
    'vllm':       'Qwen/Qwen2-72B-Instruct',
}


def get_llm(provider: str = 'openai', model: Optional[str] = None,
            use_cache: bool = True, **kwargs: Any) -> BaseLLM:
    """Create an LLM by provider name.

    Examples
    --------
    >>> llm = get_llm('openai', 'gpt-4o')
    >>> llm = get_llm('anthropic')            # defaults to Claude Sonnet
    >>> llm = get_llm('qwen')                 # Qwen2-72B via HF
    >>> llm = get_llm('vllm', 'Qwen/Qwen2-72B-Instruct', tensor_parallel_size=4)
    """
    model = model or _DEFAULTS.get(provider, 'gpt-4o')

    if provider == 'openai':
        llm = OpenAILLM(model=model, **kwargs)
    elif provider == 'anthropic':
        llm = AnthropicLLM(model=model, **kwargs)
    elif provider == 'google':
        llm = GoogleLLM(model=model, **kwargs)
    elif provider in ('huggingface', 'hf', 'qwen', 'qwen-small',
                      'llama', 'llama-small'):
        llm = HuggingFaceLLM(model=model, **kwargs)
    elif provider == 'vllm':
        llm = VLLMBackend(model=model, **kwargs)
    else:
        raise ValueError(f'Unknown LLM provider: {provider}')

    return CachedLLM(llm) if use_cache else llm
