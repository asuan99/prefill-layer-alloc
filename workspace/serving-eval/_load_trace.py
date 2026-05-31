"""
_load_trace.py — ShareGPT / LongBench / smoke trace loader for serving-eval.

Returns list of (prompt_text, expected_output_tokens) tuples suitable for
driving an OpenAI-compatible vLLM server.

ShareGPT  : real conversational prompts, short-to-medium length (64–1024 tok)
LongBench : document summarisation / QA, long prompts (2K–8K tok)
Smoke     : synthetic 10-prompt trace for pipeline validation on any GPU
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# ShareGPT
# ---------------------------------------------------------------------------

_SHAREGPT_HF_DATASET = "anon8231489123/ShareGPT_Vicuna_unfiltered"
_SHAREGPT_HF_FILE    = "ShareGPT_V3_unfiltered_cleaned_split.json"


def load_sharegpt(
    n_samples: int = 500,
    seed: int = 42,
    min_prompt_tokens: int = 64,
    max_prompt_tokens: int = 1024,
    cache_dir: Optional[Path] = None,
) -> list[tuple[str, int]]:
    """Load ShareGPT prompts for serving evaluation.

    Each returned tuple is (prompt_text, expected_output_tokens).
    Prompts are drawn from the first human turn; output_len is estimated
    from the corresponding assistant turn length.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets  # required for ShareGPT loading")

    ds = load_dataset(
        _SHAREGPT_HF_DATASET,
        data_files=_SHAREGPT_HF_FILE,
        split="train",
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    rng = random.Random(seed)
    candidates: list[tuple[str, int]] = []

    for item in ds:
        convs = item.get("conversations", [])
        if len(convs) < 2:
            continue
        human = next((c["value"] for c in convs if c.get("from") == "human"), None)
        gpt   = next((c["value"] for c in convs if c.get("from") == "gpt"), None)
        if not human or not gpt:
            continue

        prompt_tok = len(human) // 4     # ~4 chars per token
        output_tok = max(64, min(512, len(gpt) // 4))

        if min_prompt_tokens <= prompt_tok <= max_prompt_tokens:
            candidates.append((human, output_tok))

        if len(candidates) >= n_samples * 4:
            break

    rng.shuffle(candidates)
    return candidates[:n_samples]


# ---------------------------------------------------------------------------
# LongBench
# ---------------------------------------------------------------------------

_LONGBENCH_TASKS = ("gov_report", "summ_screen_fd", "qasper")


def load_longbench_subset(
    n_samples: int = 100,
    seed: int = 42,
    tasks: tuple[str, ...] = _LONGBENCH_TASKS,
) -> list[tuple[str, int]]:
    """Load LongBench tasks for long-context prefill workloads (2K–8K tokens).

    Tasks exercise the prefill path heavily.  Each prompt includes the full
    document context, stressing the hybrid model's SSM prefill stages.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets  # required for LongBench loading")

    rng = random.Random(seed)
    candidates: list[tuple[str, int]] = []

    for task in tasks:
        try:
            ds = load_dataset("THUDM/LongBench", task, split="test")
            for item in ds:
                ctx = item.get("context", "") or item.get("input", "")
                qst = item.get("input", "")
                ans = (item.get("answers") or [""])[0]
                if not ctx:
                    continue
                prompt = f"Context:\n{ctx}\n\nQuestion:\n{qst}\n\nAnswer:"
                out_tok = max(32, min(256, len(ans) // 4))
                candidates.append((prompt, out_tok))
        except Exception as exc:
            print(f"  Warning: failed to load LongBench/{task}: {exc}")

    rng.shuffle(candidates)
    return candidates[:n_samples]


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

_SMOKE_TEMPLATES = [
    ("Summarize in one sentence: {text}", 64),
    ("What is the capital of {country}? Answer in one sentence.", 32),
    ("Write a haiku about {topic}.", 32),
    ("Translate 'Hello, world!' to {language}.", 16),
    ("List two benefits of {concept}.", 48),
    ("Define {term} in simple terms.", 40),
    ("Compare {a} and {b} briefly.", 64),
    ("Give an example of {concept} in everyday life.", 56),
    ("What is {topic} and why does it matter?", 64),
    ("Finish this sentence: 'The best way to learn is ...'", 48),
]

_SMOKE_FILLERS = [
    {"text": "Machine learning is a subset of artificial intelligence."},
    {"country": "France"},
    {"topic": "the ocean"},
    {"language": "Spanish"},
    {"concept": "regular exercise"},
    {"term": "entropy"},
    {"a": "cats", "b": "dogs"},
    {"concept": "parallel processing"},
    {"topic": "photosynthesis"},
    {},  # for the sentence-completion template
]


def make_smoke_trace(n_samples: int = 10) -> list[tuple[str, int]]:
    """Generate n_samples synthetic short prompts for pipeline validation.

    Valid on any GPU; use only for verifying end-to-end plumbing, not
    for drawing latency/throughput conclusions.
    """
    results: list[tuple[str, int]] = []
    for i in range(n_samples):
        tmpl, out_tok = _SMOKE_TEMPLATES[i % len(_SMOKE_TEMPLATES)]
        filler = _SMOKE_FILLERS[i % len(_SMOKE_FILLERS)]
        prompt = tmpl
        for k, v in filler.items():
            prompt = prompt.replace(f"{{{k}}}", v)
        results.append((prompt, out_tok))
    return results


# ---------------------------------------------------------------------------
# Local JSONL fallback
# ---------------------------------------------------------------------------

def load_local_jsonl(
    path: Path,
    prompt_key: str = "prompt",
    output_len_key: str = "output_len",
    default_output_len: int = 128,
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> list[tuple[str, int]]:
    """Load a local JSONL file where each line is a request dict.

    Fields used: ``prompt_key`` (required) and ``output_len_key`` (optional,
    defaults to ``default_output_len`` if absent).
    """
    records: list[tuple[str, int]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompt = item.get(prompt_key, "")
            out_len = item.get(output_len_key, default_output_len)
            if prompt:
                records.append((prompt, int(out_len)))

    if n_samples is not None:
        rng = random.Random(seed)
        rng.shuffle(records)
        records = records[:n_samples]
    return records
