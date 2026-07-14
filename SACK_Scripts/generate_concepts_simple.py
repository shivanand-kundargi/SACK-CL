#!/usr/bin/env python3
"""Simple concept generation script using an OpenAI-compatible chat endpoint.

This script reuses the prompt template from the project's `generate_concept_costs.py`.
It is intentionally small and robust: it posts chat completions to a compatible endpoint
(default: https://api.openai.com/v1) and records latency, token usage (if returned),
and parsed concept lists for CIFAR-100, CUB-200, and ImageNet-R classes.

Usage example:
  export OPENAI_API_KEY=...
  python SACK_Scripts/generate_concepts_simple.py --out-dir results/concepts --model gpt-5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests


DEFAULT_DATASETS = "cifar100,cub200,imagenet-r"

# Minimal embedded helpers copied/adapted from generate_concept_costs.py to avoid
# importing that module (which executes large top-level code and can cause
# dataclass/import issues when loaded dynamically).
SYSTEM_PROMPT = "You are a helpful assistant that lists short, fine-grained visual features."


def build_user_prompt(category_name: str) -> str:
    return (
        f"List short (1-2 word) visual fine-grained features that describe a {category_name}. "
        f"Focus and pay attention to detail on its appearance, structure, texture, and environment. "
        f"Give 10-20 concepts per {category_name} such that the concepts strongly define {category_name}. "
        "Return only one bullet list, with each concept on a separate line beginning with '-'."
    )


DEFAULT_CLASS_SOURCES = {
    "cifar100": Path("concept_sets/gpt3_init_dict/gpt3_cifar100_important.json"),
    "cub200": Path("concept_sets/gpt3_init_dict/gpt3_cub_important.json"),
    "imagenet-r": Path("concept_sets/gpt3_init_dict/gpt3_imagenet_r_important_new.json"),
}


def normalize_dataset(value: str) -> str:
    value = value.strip().lower().replace("_", "-")
    aliases = {
        "cifar-100": "cifar100",
        "cifar": "cifar100",
        "cub-200": "cub200",
        "cub": "cub200",
        "imagenetr": "imagenet-r",
        "imagenet-rendition": "imagenet-r",
    }
    return aliases.get(value, value)


def load_classes(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Class source not found: {path}")
    if path.suffix.lower() == ".json":
        with path.open() as file:
            data = json.load(file)
        if isinstance(data, dict):
            classes = list(data.keys())
        elif isinstance(data, list):
            classes = [str(item) for item in data]
        else:
            raise ValueError(f"Unsupported JSON class source shape in {path}")
    else:
        classes = [line.strip() for line in path.read_text().splitlines() if line.strip()]

    seen = set()
    clean_classes = []
    for class_name in classes:
        class_name = str(class_name).replace("_", " ").strip()
        if class_name and class_name not in seen:
            clean_classes.append(class_name)
            seen.add(class_name)
    return clean_classes


def parse_concepts(text: str) -> List[str]:
    concepts: List[str] = []
    seen = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        line = __import__("re").sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        line = line.strip("\"'` ")
        line = __import__("re").sub(r"\s+", " ", line)
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith(("visual features", "concepts:", "features:")):
            continue
        if len(line.split()) > 6:
            continue
        if line not in seen:
            concepts.append(line)
            seen.add(line)
    if not concepts:
        for chunk in __import__("re").split(r"[,;]", text):
            line = chunk.strip().strip("\"'` .")
            line = __import__("re").sub(r"\s+", " ", line)
            if line and len(line.split()) <= 6 and line not in seen:
                concepts.append(line)
                seen.add(line)
    return concepts


def estimate_tokens(text: str) -> int:
    return max(1, int((len(text) + 3) // 4))


def token_usage_or_estimate(class_name: str, raw_text: str, usage: Dict[str, int]):
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if prompt_tokens is not None and completion_tokens is not None:
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens
        return prompt_tokens, completion_tokens, total_tokens, False

    prompt_text = SYSTEM_PROMPT + "\n" + build_user_prompt(class_name)
    prompt_tokens = estimate_tokens(prompt_text)
    completion_tokens = estimate_tokens(raw_text)
    total_tokens = prompt_tokens + completion_tokens
    return prompt_tokens, completion_tokens, total_tokens, True


def endpoint_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return f"{base}/chat/completions"


def call_chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: float = 60.0,
    dump_payload: bool = False,
    use_max_completion_tokens: bool = False,
) -> (str, Dict[str, int]):
    url = endpoint_url(base_url)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    # Some OpenAI-compatible endpoints (newer OpenAI models) expect
    # 'max_completion_tokens' instead of 'max_tokens'. Use the
    # caller-provided flag to select the correct parameter name.
    if use_max_completion_tokens:
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
    if dump_payload:
        print("[debug] POST", url)
        try:
            print(json.dumps(payload, indent=2))
        except Exception:
            print("[debug] <failed to dump payload>")
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        resp.raise_for_status()
    except Exception as exc:
        body = None
        try:
            body = resp.text
        except Exception:
            body = "<failed to read response body>"
        raise RuntimeError(f"HTTP {resp.status_code} when calling {url}: {body}") from exc
    data = resp.json()
    # Extract text
    choices = data.get("choices") or []
    if not choices:
        return "", {}
    message = choices[0].get("message") or {}
    text = message.get("content") or choices[0].get("text", "")
    usage = data.get("usage") or {}
    usage_int: Dict[str, int] = {}
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        v = usage.get(k)
        if isinstance(v, (int, float)):
            usage_int[k] = int(v)
    return str(text), usage_int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple concept generation (OpenAI-compatible or local unsloth/transformers).")
    parser.add_argument("--datasets", type=str, default=DEFAULT_DATASETS)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--backend", choices=("openai-compatible", "api", "unsloth", "transformers"), default="openai-compatible")
    parser.add_argument("--base-url", type=str, default=os.environ.get("SACK_LLM_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Optional cache dir for Hugging Face files (use a Lustre path if default quota is exceeded).")
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY") or os.environ.get("SACK_LLM_API_KEY"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--limit-classes", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dump-payload", action="store_true",
                        help="Print the JSON payload sent to the API (for debugging 400 errors).")
    parser.add_argument("--api-allow-temperature", action="store_true",
                        help="Allow sending the requested --temperature to the remote API; some models only accept the default value.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Only require an API key for remote OpenAI-compatible endpoints (including alias 'api').
    if args.backend in ("openai-compatible", "api") and args.api_key is None and not args.dry_run:
        print("ERROR: --api-key or OPENAI_API_KEY must be set for --backend=openai-compatible/--backend=api non-dry runs.")
        raise SystemExit(2)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.backend in ("openai-compatible", "api"):
        print(f"[info] Using remote API backend '{args.backend}' -> base_url={args.base_url} model={args.model}")

    # If a custom cache directory is provided, prefer it for huggingface/transformers
    if getattr(args, "cache_dir", None):
        cache_path = Path(args.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_path)
        os.environ["TRANSFORMERS_CACHE"] = str(cache_path)
        # HF_HOME typically contains hub/; set to parent so hub cache is at HF_HOME/hub
        os.environ.setdefault("HF_HOME", str(cache_path))

    datasets = [normalize_dataset(d) for d in args.datasets.split(",") if d.strip()]

    all_records = []
    generator = None
    if args.backend in ("unsloth", "transformers") and not args.dry_run:
        print(f"[info] Initializing local backend '{args.backend}' model: {args.model}", flush=True)
        if args.backend == "transformers":
            # Simple transformers-based generator (minimal)
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                class SimpleTransformersGenerator:
                    def __init__(self, model_name: str, cache_dir: Optional[str] = None):
                        trust_remote_code = True
                        load_kwargs = {"trust_remote_code": trust_remote_code}
                        if cache_dir:
                            load_kwargs["cache_dir"] = cache_dir
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
                        if self.tokenizer.pad_token_id is None and getattr(self.tokenizer, "eos_token_id", None) is not None:
                            self.tokenizer.pad_token = self.tokenizer.eos_token
                        model_kwargs = {"device_map": "auto"}
                        if cache_dir:
                            model_kwargs["cache_dir"] = cache_dir
                        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                        self.model.eval()
                        self.torch = torch

                    def generate(self, class_name: str, temperature: float, max_tokens: int):
                        try:
                            encoded = self.tokenizer.apply_chat_template(
                                [
                                    {"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user", "content": build_user_prompt(class_name)},
                                ],
                                add_generation_prompt=True,
                                return_tensors="pt",
                                return_dict=True,
                            )
                        except Exception:
                            prompt = (
                                f"System: {SYSTEM_PROMPT}\n"
                                f"User: {build_user_prompt(class_name)}\n"
                                "Assistant:\n"
                            )
                            encoded = self.tokenizer(prompt, return_tensors="pt")

                        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
                        prompt_tokens = int(encoded["input_ids"].shape[-1])
                        do_sample = temperature > 0
                        # Only pass input_ids and attention_mask to avoid shape mismatches
                        generation_kwargs = {
                            "input_ids": encoded["input_ids"],
                            "attention_mask": encoded.get("attention_mask"),
                            "max_new_tokens": max_tokens,
                            "do_sample": do_sample,
                            "pad_token_id": self.tokenizer.pad_token_id or getattr(self.tokenizer, "eos_token_id", None),
                            "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
                        }
                        if do_sample:
                            generation_kwargs["temperature"] = temperature

                        with self.torch.no_grad():
                            output_ids = self.model.generate(**generation_kwargs)

                        new_tokens = output_ids[0, prompt_tokens:]
                        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                        completion_tokens = int(new_tokens.numel())
                        usage = {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        }
                        return text, usage

                generator = SimpleTransformersGenerator(args.model, cache_dir=args.cache_dir)
            except Exception as exc:
                print(f"[error] Failed to init transformers generator: {exc}")
                raise
        else:
            # unsloth backend
            try:
                # Disable TorchDynamo/torch.compile optimizations which can cause
                # FX/fake-tensor shape-mismatch errors for some MoE kernels.
                os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
                print("[info] TORCHDYNAMO_DISABLE=1 set to avoid FX fake-tensor errors.")
                import torch
                from unsloth import FastLanguageModel

                model_obj, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=args.model,
                    dtype=None,
                    max_seq_length=4096,
                    load_in_4bit=False,
                    full_finetuning=False,
                )
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                class UnslothGenerator:
                    def __init__(self, model, tokenizer, device):
                        self.model = model
                        self.tokenizer = tokenizer
                        self.device = device

                    def generate(self, class_name: str, temperature: float, max_tokens: int):
                        messages = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": build_user_prompt(class_name)},
                        ]
                        try:
                            encoded = self.tokenizer.apply_chat_template(
                                messages,
                                add_generation_prompt=True,
                                return_tensors="pt",
                                return_dict=True,
                            )
                        except Exception:
                            prompt = (
                                f"System: {SYSTEM_PROMPT}\n"
                                f"User: {build_user_prompt(class_name)}\n"
                                "Assistant:\n"
                            )
                            encoded = self.tokenizer(prompt, return_tensors="pt")

                        encoded = {k: v.to(self.device) for k, v in encoded.items()}
                        prompt_tokens = int(encoded["input_ids"].shape[-1])
                        do_sample = temperature > 0
                        # Only pass input_ids and attention_mask to avoid shape mismatches
                        generation_kwargs = {
                            "input_ids": encoded["input_ids"],
                            "attention_mask": encoded.get("attention_mask"),
                            "max_new_tokens": max_tokens,
                            "do_sample": do_sample,
                            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None) or getattr(self.tokenizer, "eos_token_id", None),
                            "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
                        }
                        if do_sample:
                            generation_kwargs["temperature"] = temperature

                        with torch.no_grad():
                            output_ids = self.model.generate(**generation_kwargs)

                        new_tokens = output_ids[0, prompt_tokens:]
                        try:
                            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                        except Exception:
                            text = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
                        completion_tokens = int(new_tokens.numel())
                        usage = {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        }
                        return text, usage

                generator = UnslothGenerator(model_obj, tokenizer, device)
            except Exception as exc:
                print(f"[error] Failed to init unsloth generator: {exc}")
                raise
    for dataset in datasets:
        source = DEFAULT_CLASS_SOURCES[dataset]
        repo_root = Path(__file__).resolve().parents[1]
        if not source.is_absolute():
            source = repo_root / source
        classes = load_classes(source)
        if args.limit_classes:
            classes = classes[: args.limit_classes]
        print(f"[info] {dataset}: generating for {len(classes)} classes from {source}")

        dataset_out = out_dir / dataset
        dataset_out.mkdir(parents=True, exist_ok=True)
        results = {}
        per_class = []

        for idx, class_name in enumerate(classes):
            start = time.perf_counter()
            if args.dry_run:
                raw_text = "- dry-run concept"
                usage = {}
            else:
                try:
                    if generator is not None:
                        raw_text, usage = generator.generate(
                            class_name=class_name,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                        )
                    else:
                        # For some hosted models the API rejects non-default temperature
                        # values (see error 'unsupported_value'). Unless the user passed
                        # --api-allow-temperature, force temperature to 1 for remote API.
                        temp_to_use = args.temperature if getattr(args, "api_allow_temperature", False) else 1.0
                        if temp_to_use != args.temperature:
                            print(f"[info] Forcing temperature=1.0 for remote API call (model may not accept custom temperatures).")
                        raw_text, usage = call_chat_completion(
                                base_url=args.base_url,
                                api_key=args.api_key,
                                model=args.model,
                                system_prompt=SYSTEM_PROMPT,
                                user_prompt=build_user_prompt(class_name),
                                temperature=temp_to_use,
                                max_tokens=args.max_tokens,
                                timeout=args.timeout,
                                dump_payload=getattr(args, "dump_payload", False),
                                use_max_completion_tokens=True,
                            )
                except Exception as exc:
                    raw_text = ""
                    usage = {}
                    print(f"[error] {dataset} {class_name}: {exc}")

            latency = time.perf_counter() - start
            concepts = parse_concepts(raw_text)
            prompt_tokens, completion_tokens, total_tokens, estimated = token_usage_or_estimate(
                class_name, raw_text, usage
            )

            rec = {
                "dataset": dataset,
                "class_index": idx,
                "class_name": class_name,
                "latency_s": latency,
                "concepts": concepts,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "tokens_are_estimated": estimated,
                "raw_text": raw_text,
            }
            per_class.append(rec)
            if concepts:
                results[class_name] = concepts

            print(f"[{dataset}] {idx+1}/{len(classes)} {class_name} | latency={latency:.2f}s concepts={len(concepts)} tokens={total_tokens}")

        (out_dir / f"{dataset}_concepts.json").write_text(json.dumps(results, indent=2))
        (out_dir / f"{dataset}_per_class.json").write_text(json.dumps(per_class, indent=2))
        all_records.extend(per_class)

    # Write a simple overall summary
    summary = {}
    for dataset in set(r["dataset"] for r in all_records):
        group = [r for r in all_records if r["dataset"] == dataset]
        completed = [r for r in group if r.get("concepts")]
        total_tokens = sum((r.get("total_tokens") or 0) for r in completed)
        total_time = sum((r.get("latency_s") or 0.0) for r in completed)
        summary[dataset] = {
            "num_classes": len(group),
            "completed": len(completed),
            "avg_concepts_per_class": (sum(len(r.get("concepts") or []) for r in completed) / len(completed)) if completed else 0,
            "total_tokens": total_tokens,
            "total_time_s": total_time,
            "mean_latency_s": (total_time / len(completed)) if completed else 0,
        }

    (out_dir / "concept_generation_simple_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
