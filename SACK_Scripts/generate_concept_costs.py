#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib import error, request


DEFAULT_CLASS_SOURCES = {
    "cifar100": Path("concept_sets/gpt3_init_dict/gpt3_cifar100_important.json"),
    "cub200": Path("concept_sets/gpt3_init_dict/gpt3_cub_important.json"),
    "imagenet-r": Path("concept_sets/gpt3_init_dict/gpt3_imagenet_r_important_new.json"),
}

DATASET_LABELS = {
    "cifar100": "CIFAR-100",
    "cub200": "CUB-200",
    "imagenet-r": "ImageNet-R",
}

SYSTEM_PROMPT = "You are a helpful assistant that lists short, fine-grained visual features."


def build_user_prompt(category_name: str) -> str:
    """Prompt template adapted from concept_sets/concept_generation.py."""
    return (
        f"List short (1-2 word) visual fine-grained features that describe a {category_name}. "
        f"Focus and pay attention to detail on its appearance, structure, texture, and environment. "
        f"Give 10-20 concepts per {category_name} such that the concepts strongly define {category_name}. "
        "Return only one bullet list, with each concept on a separate line beginning with '-'."
    )


@dataclass
class GenerationRecord:
    dataset: str
    class_index: int
    class_name: str
    latency_s: float
    concepts: List[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    tokens_are_estimated: bool
    raw_text: str
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SACK concept banks with a local LLM and summarize preprocessing cost."
    )
    parser.add_argument("--datasets", type=str, default="cifar100,cub200,imagenet-r",
                        help="Comma-separated datasets: cifar100,cub200,imagenet-r.")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory for generated concepts and cost tables.")
    parser.add_argument("--backend", choices=("transformers", "openai-compatible"), default="transformers",
                        help="Generation backend. Default loads the model directly with Hugging Face Transformers.")
    parser.add_argument("--model", type=str, required=True,
                        help="Hugging Face model id or local path. Default launcher uses openai/gpt-oss-120b.")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Only for --backend=openai-compatible, e.g. http://localhost:8000/v1.")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for the endpoint. Defaults to SACK_LLM_API_KEY, OPENAI_API_KEY, then EMPTY.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--sleep-secs", type=float, default=0.0,
                        help="Optional delay between requests.")
    parser.add_argument("--retries", type=int, default=2,
                        help="Retries per class after the first failed attempt.")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse existing generated classes in <out-dir>/<dataset>_concepts.json.")
    parser.add_argument("--limit-classes", type=int, default=None,
                        help="Debug option: generate only the first N classes for each dataset.")
    parser.add_argument("--input-price-per-1m", type=float, default=0.0,
                        help="Input token price used for cost estimate. Use 0 for local/open models.")
    parser.add_argument("--output-price-per-1m", type=float, default=0.0,
                        help="Output token price used for cost estimate. Use 0 for local/open models.")
    parser.add_argument("--class-source", action="append", default=[],
                        help="Override class source as dataset=/path/to/classes.json or .txt. Can be repeated.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Do not call the model; only write class lists and prompt previews.")
    parser.add_argument("--skip-endpoint-check", action="store_true",
                        help="Skip the /models preflight check before generation.")
    parser.add_argument("--torch-dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto",
                        help="Torch dtype for the transformers backend.")
    parser.add_argument("--device-map", type=str, default="auto",
                        help="Device map for the transformers backend.")
    parser.add_argument("--model-cache-dir", type=str, default=None,
                        help="Optional Hugging Face cache dir for the transformers backend.")
    parser.add_argument("--auto-download", action="store_true",
                        help="Download the transformers model into --model-cache-dir if it is not cached.")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="Optional Hugging Face token. Defaults to HF_TOKEN/HUGGING_FACE_HUB_TOKEN.")
    parser.add_argument("--local-files-only", action="store_true",
                        help="Load only local model files with the transformers backend.")
    parser.add_argument("--no-trust-remote-code", action="store_true",
                        help="Disable trust_remote_code when loading the transformers model/tokenizer.")
    return parser.parse_args()


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
    value = aliases.get(value, value)
    if value not in DEFAULT_CLASS_SOURCES:
        raise ValueError(f"Unsupported dataset '{value}'. Expected one of {sorted(DEFAULT_CLASS_SOURCES)}.")
    return value


def load_class_source_overrides(items: Sequence[str]) -> Dict[str, Path]:
    overrides: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--class-source must look like dataset=/path/to/file, got: {item}")
        dataset_raw, path_raw = item.split("=", 1)
        overrides[normalize_dataset(dataset_raw)] = Path(path_raw)
    return overrides


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


def endpoint_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def models_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        base_url = base_url[: -len("/chat/completions")]
    return f"{base_url}/models"


def check_endpoint(base_url: str, api_key: str, timeout: float) -> None:
    headers = {"Authorization": f"Bearer {api_key}"}
    req = request.Request(models_url(base_url), headers=headers, method="GET")
    try:
        with request.urlopen(req, timeout=min(timeout, 15.0)) as response:
            response.read(512)
    except Exception as exc:
        raise RuntimeError(
            "Could not reach the OpenAI-compatible endpoint. "
            f"Tried {models_url(base_url)} and got: {exc}. "
            "Start your vLLM/OpenAI-compatible server first, or set SACK_LLM_BASE_URL to the correct host/port."
        ) from exc


def likely_cache_dirs(model_cache_dir: Optional[str]) -> List[Path]:
    candidates: List[Path] = []
    if model_cache_dir:
        candidates.append(Path(model_cache_dir))
    env_cache_keys = ("HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")
    for key in env_cache_keys:
        value = os.environ.get(key)
        if value:
            candidates.append(Path(value))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home) / "hub")
    home = Path.home()
    candidates.extend([
        home / ".cache/huggingface/hub",
        Path("/g/g91/kundargi1/.cache/huggingface/hub"),
        Path("/p/lustre1/kundargi1/.cache/huggingface/hub"),
    ])

    unique: List[Path] = []
    seen = set()
    for path in candidates:
        path = path.expanduser()
        path_key = str(path)
        if path_key not in seen:
            unique.append(path)
            seen.add(path_key)
    return unique


def cached_snapshot_candidates(model_name: str, cache_dirs: Sequence[Path]) -> List[Path]:
    if "/" not in model_name:
        return []
    repo_cache_name = "models--" + model_name.replace("/", "--")
    candidates: List[Path] = []
    for cache_dir in cache_dirs:
        snapshots_dir = cache_dir / repo_cache_name / "snapshots"
        if not snapshots_dir.exists():
            continue
        for config_path in sorted(snapshots_dir.glob("*/config.json")):
            candidates.append(config_path.parent)
    return candidates


def is_path_like_model_ref(model_name: str) -> bool:
    return model_name.startswith(("/", "./", "../")) or os.sep in model_name and Path(model_name).exists()


def snapshot_download_with_token(*, repo_id: str, cache_dir: Optional[str], token: Optional[str]) -> str:
    try:
        import inspect
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "Auto-download requires huggingface_hub, which should normally come with transformers. "
            "Install/activate an environment with huggingface_hub or set SACK_AUTO_DOWNLOAD_MODEL=0 "
            "to use an already cached model."
        ) from exc

    kwargs = {
        "repo_id": repo_id,
        "cache_dir": cache_dir,
        "local_files_only": False,
        "resume_download": True,
    }
    if token:
        params = inspect.signature(snapshot_download).parameters
        if "token" in params:
            kwargs["token"] = token
        elif "use_auth_token" in params:
            kwargs["use_auth_token"] = token

    return str(snapshot_download(**kwargs))


def resolve_model_reference(args: argparse.Namespace) -> str:
    model_name = args.model
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return str(model_path)

    if is_path_like_model_ref(model_name):
        raise RuntimeError(
            f"Model path does not exist: {model_name}. "
            "Leave SACK_LLM_MODEL unset to use the default Hugging Face id and cache/download path."
        )

    cache_dirs = likely_cache_dirs(args.model_cache_dir)
    candidates = cached_snapshot_candidates(model_name, cache_dirs)
    if candidates:
        newest = max(candidates, key=lambda path: path.stat().st_mtime)
        print(f"[info] Found cached model snapshot: {newest}", flush=True)
        return str(newest)

    if args.auto_download and not args.local_files_only:
        if not args.model_cache_dir:
            raise RuntimeError("--auto-download requires --model-cache-dir so the model is stored in a known path.")
        Path(args.model_cache_dir).mkdir(parents=True, exist_ok=True)
        print(f"[info] Model not found in cache. Downloading {model_name} to {args.model_cache_dir}", flush=True)
        token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        try:
            snapshot_path = snapshot_download_with_token(
                repo_id=model_name,
                cache_dir=args.model_cache_dir,
                token=token,
            )
        except Exception as exc:
            raise RuntimeError(
                "Model auto-download failed. The launcher is configured to manage the checkpoint path, "
                f"but this node could not download '{model_name}' into {args.model_cache_dir}.\n\n"
                "Run the same command on a node with Hugging Face access. If the model requires authentication, "
                "export HF_TOKEN before running the script.\n\n"
                f"Original error: {exc}"
            ) from exc
        print(f"[info] Download complete. Using snapshot: {snapshot_path}", flush=True)
        return snapshot_path

    return model_name


def build_transformers_load_error(
    *,
    model_name: str,
    model_cache_dir: Optional[str],
    stage: str,
    exc: Exception,
) -> str:
    cache_lines = []
    for cache_dir in likely_cache_dirs(model_cache_dir):
        status = "exists" if cache_dir.exists() else "missing"
        cache_lines.append(f"  - {cache_dir} ({status})")
    cache_block = "\n".join(cache_lines) if cache_lines else "  - no cache directories inferred"
    torch_xpu = "unknown"
    try:
        import torch
        torch_xpu = str(hasattr(torch, "xpu"))
    except Exception:
        pass
    version_lines = []
    for package in ("transformers", "tokenizers", "accelerate", "kernels", "triton", "torch"):
        try:
            from importlib import metadata
            version = metadata.version(package)
        except Exception:
            version = "not installed"
        version_lines.append(f"  - {package}: {version}")
    version_lines.append(f"  - torch_has_xpu: {torch_xpu}")
    version_lines.append(f"  - python: {sys.version.split()[0]}")
    version_block = "\n".join(version_lines)
    return (
        f"Failed to load {stage} for model '{model_name}'.\n"
        "The launcher now manages the checkpoint path automatically. It checks the configured cache first "
        "and downloads the Hugging Face model there when possible.\n\n"
        "For GPT-OSS, the local Transformers stack must be recent enough. The launcher can install these "
        "into data/w3_pydeps automatically; run the bash script directly instead of calling this Python file.\n\n"
        "Detected package versions:\n"
        f"{version_block}\n\n"
        "Run the same bash script on a node with Hugging Face access. If authentication is needed, "
        "export HF_TOKEN before running it.\n\n"
        "Cache directories checked:\n"
        f"{cache_block}\n\n"
        f"Original error: {exc}"
    )


def chat_completion_request(
    *,
    base_url: str,
    api_key: str,
    model: str,
    class_name: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> Tuple[str, Dict[str, int]]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(class_name)},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = request.Request(endpoint_url(base_url), data=body, headers=headers, method="POST")
    with request.urlopen(req, timeout=timeout) as response:
        response_body = response.read().decode("utf-8")
    data = json.loads(response_body)
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"Endpoint response had no choices: {response_body[:500]}")
    message = choices[0].get("message") or {}
    text = message.get("content")
    if text is None:
        text = choices[0].get("text", "")
    usage = data.get("usage") or {}
    usage_int = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, (int, float)):
            usage_int[key] = int(value)
    return str(text), usage_int


class TransformersGenerator:
    def __init__(self, args: argparse.Namespace):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "The transformers backend requires torch and transformers. "
                "Install/use an environment with those packages, or use --backend=openai-compatible."
            ) from exc

        self.torch = torch
        self.model_name = resolve_model_reference(args)
        trust_remote_code = not args.no_trust_remote_code
        load_kwargs = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": args.local_files_only,
        }
        if args.model_cache_dir:
            load_kwargs["cache_dir"] = args.model_cache_dir

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **load_kwargs)
        except Exception as fast_exc:
            slow_kwargs = dict(load_kwargs)
            slow_kwargs["use_fast"] = False
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **slow_kwargs)
                print("[warn] Fast tokenizer failed; loaded slow tokenizer fallback.", flush=True)
            except Exception as slow_exc:
                combined_exc = RuntimeError(
                    f"fast tokenizer error: {fast_exc}\nslow tokenizer fallback error: {slow_exc}"
                )
                raise RuntimeError(
                    build_transformers_load_error(
                        model_name=args.model,
                        model_cache_dir=args.model_cache_dir,
                        stage="tokenizer/config",
                        exc=combined_exc,
                    )
                ) from slow_exc
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = dict(load_kwargs)
        model_kwargs["device_map"] = args.device_map
        dtype = self._resolve_dtype(args.torch_dtype)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        except Exception as exc:
            raise RuntimeError(
                build_transformers_load_error(
                    model_name=args.model,
                    model_cache_dir=args.model_cache_dir,
                    stage="model weights",
                    exc=exc,
                )
            ) from exc
        self.model.eval()

    def _resolve_dtype(self, value: str):
        if value == "auto":
            return None
        if value == "bfloat16":
            return self.torch.bfloat16
        if value == "float16":
            return self.torch.float16
        if value == "float32":
            return self.torch.float32
        return None

    def _build_prompt_inputs(self, class_name: str):
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
            return encoded
        except Exception:
            prompt = (
                f"System: {SYSTEM_PROMPT}\n"
                f"User: {build_user_prompt(class_name)}\n"
                "Assistant:\n"
            )
            return self.tokenizer(prompt, return_tensors="pt")

    def generate(self, class_name: str, temperature: float, max_tokens: int) -> Tuple[str, Dict[str, int]]:
        encoded = self._build_prompt_inputs(class_name)
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        do_sample = temperature > 0
        generation_kwargs = {
            **encoded,
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
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


def parse_concepts(text: str) -> List[str]:
    concepts: List[str] = []
    seen = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        line = line.strip("\"'` ")
        line = re.sub(r"\s+", " ", line)
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
        for chunk in re.split(r"[,;]", text):
            line = chunk.strip().strip("\"'` .")
            line = re.sub(r"\s+", " ", line)
            if line and len(line.split()) <= 6 and line not in seen:
                concepts.append(line)
                seen.add(line)
    return concepts


def estimate_tokens(text: str) -> int:
    # Conservative, dependency-free fallback when endpoint usage is unavailable.
    return max(1, int(math.ceil(len(text) / 4.0)))


def token_usage_or_estimate(class_name: str, raw_text: str, usage: Dict[str, int]) -> Tuple[Optional[int], Optional[int], Optional[int], bool]:
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


def load_existing_concepts(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}
    with path.open() as file:
        data = json.load(file)
    if not isinstance(data, dict):
        return {}
    return {str(key): list(value) for key, value in data.items() if isinstance(value, list)}


def write_json(path: Path, payload) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as file:
        json.dump(payload, file, indent=2)
    tmp_path.replace(path)


def generate_dataset(
    *,
    dataset: str,
    classes: List[str],
    args: argparse.Namespace,
    generator: Optional[TransformersGenerator],
) -> Tuple[List[GenerationRecord], Dict[str, List[str]]]:
    out_dir = args.out_dir
    concepts_path = out_dir / f"{dataset}_concepts.json"
    raw_path = out_dir / f"{dataset}_raw_responses.json"
    generated = load_existing_concepts(concepts_path) if args.resume else {}

    records: List[GenerationRecord] = []
    raw_payload: Dict[str, Dict[str, object]] = {}
    if raw_path.exists() and args.resume:
        try:
            raw_payload = json.loads(raw_path.read_text())
        except Exception:
            raw_payload = {}

    for class_index, class_name in enumerate(classes):
        if args.limit_classes is not None and class_index >= args.limit_classes:
            break

        if args.resume and class_name in generated:
            concepts = generated[class_name]
            record = GenerationRecord(
                dataset=dataset,
                class_index=class_index,
                class_name=class_name,
                latency_s=0.0,
                concepts=concepts,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                tokens_are_estimated=False,
                raw_text="",
                error="resumed",
            )
            records.append(record)
            continue

        if args.dry_run:
            raw_text = "- dry-run concept"
            concepts = ["dry-run concept"]
            prompt_tokens, completion_tokens, total_tokens, estimated = token_usage_or_estimate(class_name, raw_text, {})
            record = GenerationRecord(
                dataset=dataset,
                class_index=class_index,
                class_name=class_name,
                latency_s=0.0,
                concepts=concepts,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                tokens_are_estimated=estimated,
                raw_text=raw_text,
            )
        else:
            last_error = ""
            record = None
            for attempt in range(args.retries + 1):
                start = time.perf_counter()
                try:
                    if args.backend == "transformers":
                        if generator is None:
                            raise RuntimeError("Transformers backend selected but local generator was not initialized.")
                        raw_text, usage = generator.generate(
                            class_name=class_name,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                        )
                    else:
                        raw_text, usage = chat_completion_request(
                            base_url=args.base_url,
                            api_key=args.api_key,
                            model=args.model,
                            class_name=class_name,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                            timeout=args.timeout,
                        )
                    latency = time.perf_counter() - start
                    concepts = parse_concepts(raw_text)
                    prompt_tokens, completion_tokens, total_tokens, estimated = token_usage_or_estimate(class_name, raw_text, usage)
                    record = GenerationRecord(
                        dataset=dataset,
                        class_index=class_index,
                        class_name=class_name,
                        latency_s=latency,
                        concepts=concepts,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        tokens_are_estimated=estimated,
                        raw_text=raw_text,
                    )
                    break
                except (error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
                    latency = time.perf_counter() - start
                    last_error = str(exc)
                    if attempt < args.retries:
                        time.sleep(min(2.0 * (attempt + 1), 10.0))
                    else:
                        record = GenerationRecord(
                            dataset=dataset,
                            class_index=class_index,
                            class_name=class_name,
                            latency_s=latency,
                            concepts=[],
                            prompt_tokens=None,
                            completion_tokens=None,
                            total_tokens=None,
                            tokens_are_estimated=False,
                            raw_text="",
                            error=last_error,
                        )

            if args.sleep_secs > 0:
                time.sleep(args.sleep_secs)

        assert record is not None
        records.append(record)
        if not record.error:
            generated[class_name] = record.concepts
        raw_payload[class_name] = {
            "class_index": class_index,
            "latency_s": record.latency_s,
            "concepts": record.concepts,
            "prompt_tokens": record.prompt_tokens,
            "completion_tokens": record.completion_tokens,
            "total_tokens": record.total_tokens,
            "tokens_are_estimated": record.tokens_are_estimated,
            "raw_text": record.raw_text,
            "error": record.error,
        }
        write_json(concepts_path, generated)
        write_json(raw_path, raw_payload)

        status = "ok" if not record.error else f"error={record.error[:80]}"
        print(
            f"[{dataset}] {class_index + 1}/{len(classes)} {class_name} | "
            f"latency={record.latency_s:.2f}s concepts={len(record.concepts)} tokens={record.total_tokens} {status}",
            flush=True,
        )

    return records, generated


def numeric(values: Iterable[Optional[float]]) -> List[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def sum_int(values: Iterable[Optional[int]]) -> int:
    return int(sum(value for value in values if value is not None))


def mean_or_nan(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.mean(values)) if values else math.nan


def std_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def format_float(value: float, digits: int = 2) -> str:
    if not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def write_records_csv(records: List[GenerationRecord], path: Path) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "dataset", "class_index", "class_name", "latency_s", "num_concepts",
            "prompt_tokens", "completion_tokens", "total_tokens", "tokens_are_estimated", "error", "concepts"
        ])
        for record in records:
            writer.writerow([
                record.dataset,
                record.class_index,
                record.class_name,
                f"{record.latency_s:.6f}",
                len(record.concepts),
                "" if record.prompt_tokens is None else record.prompt_tokens,
                "" if record.completion_tokens is None else record.completion_tokens,
                "" if record.total_tokens is None else record.total_tokens,
                int(record.tokens_are_estimated),
                record.error,
                "; ".join(record.concepts),
            ])


def summarize_records(records: List[GenerationRecord], args: argparse.Namespace) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for dataset in sorted({record.dataset for record in records}):
        group = [record for record in records if record.dataset == dataset]
        completed = [record for record in group if not record.error or record.error == "resumed"]
        generated = [record for record in group if not record.error]
        latencies = numeric(record.latency_s for record in generated)
        concept_counts = numeric(len(record.concepts) for record in completed)
        prompt_tokens = sum_int(record.prompt_tokens for record in generated)
        completion_tokens = sum_int(record.completion_tokens for record in generated)
        total_tokens = sum_int(record.total_tokens for record in generated)
        input_cost = prompt_tokens * args.input_price_per_1m / 1_000_000.0
        output_cost = completion_tokens * args.output_price_per_1m / 1_000_000.0
        rows.append({
            "dataset": dataset,
            "dataset_label": DATASET_LABELS.get(dataset, dataset),
            "num_classes": len(group),
            "completed_classes": len(completed),
            "failed_classes": len([record for record in group if record.error and record.error != "resumed"]),
            "generated_classes": len(generated),
            "avg_concepts_per_class": mean_or_nan(concept_counts),
            "total_wall_time_s": sum(latencies),
            "mean_latency_s": mean_or_nan(latencies),
            "std_latency_s": std_or_zero(latencies),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_input_cost": input_cost,
            "estimated_output_cost": output_cost,
            "estimated_total_cost": input_cost + output_cost,
            "token_note": "estimated where endpoint usage was unavailable" if any(record.tokens_are_estimated for record in generated) else "endpoint usage",
        })
    return rows


def write_summary_csv(rows: List[Dict[str, object]], path: Path) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "dataset", "classes", "completed", "failed", "generated_now",
            "avg_concepts_per_class", "total_wall_time_s", "mean_latency_s", "std_latency_s",
            "prompt_tokens", "completion_tokens", "total_tokens",
            "estimated_input_cost", "estimated_output_cost", "estimated_total_cost", "token_note"
        ])
        for row in rows:
            writer.writerow([
                row["dataset_label"],
                row["num_classes"],
                row["completed_classes"],
                row["failed_classes"],
                row["generated_classes"],
                format_float(float(row["avg_concepts_per_class"]), 2),
                format_float(float(row["total_wall_time_s"]), 2),
                format_float(float(row["mean_latency_s"]), 2),
                format_float(float(row["std_latency_s"]), 2),
                row["prompt_tokens"],
                row["completion_tokens"],
                row["total_tokens"],
                format_float(float(row["estimated_input_cost"]), 6),
                format_float(float(row["estimated_output_cost"]), 6),
                format_float(float(row["estimated_total_cost"]), 6),
                row["token_note"],
            ])


def write_summary_markdown(rows: List[Dict[str, object]], path: Path) -> None:
    headers = [
        "Dataset", "# Classes", "Completed", "Failed", "Avg Concepts/Class",
        "Total Time (s)", "Mean Latency/Class (s)", "Prompt Tokens",
        "Completion Tokens", "Total Tokens", "Est. Cost"
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            str(row["dataset_label"]),
            str(row["num_classes"]),
            str(row["completed_classes"]),
            str(row["failed_classes"]),
            format_float(float(row["avg_concepts_per_class"]), 2),
            format_float(float(row["total_wall_time_s"]), 2),
            format_float(float(row["mean_latency_s"]), 2),
            str(row["prompt_tokens"]),
            str(row["completion_tokens"]),
            str(row["total_tokens"]),
            f"${float(row['estimated_total_cost']):.6f}",
        ]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")


def build_overall_summary(
    *,
    rows: List[Dict[str, object]],
    args: argparse.Namespace,
    model_load_time_s: float,
    end_to_end_time_s: float,
) -> Dict[str, object]:
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in rows)
    completion_tokens = sum(int(row["completion_tokens"]) for row in rows)
    total_tokens = sum(int(row["total_tokens"]) for row in rows)
    estimated_input_cost = prompt_tokens * args.input_price_per_1m / 1_000_000.0
    estimated_output_cost = completion_tokens * args.output_price_per_1m / 1_000_000.0
    return {
        "backend": args.backend,
        "model": args.model,
        "datasets": ",".join(str(row["dataset_label"]) for row in rows),
        "classes": sum(int(row["num_classes"]) for row in rows),
        "completed": sum(int(row["completed_classes"]) for row in rows),
        "failed": sum(int(row["failed_classes"]) for row in rows),
        "model_load_time_s": model_load_time_s,
        "class_generation_time_s": sum(float(row["total_wall_time_s"]) for row in rows),
        "end_to_end_time_s": end_to_end_time_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_input_cost": estimated_input_cost,
        "estimated_output_cost": estimated_output_cost,
        "estimated_total_cost": estimated_input_cost + estimated_output_cost,
    }


def write_overall_summary(overall: Dict[str, object], out_dir: Path) -> None:
    csv_path = out_dir / "concept_generation_overall.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "backend", "model", "datasets", "classes", "completed", "failed",
            "model_load_time_s", "class_generation_time_s", "end_to_end_time_s",
            "prompt_tokens", "completion_tokens", "total_tokens", "estimated_total_cost"
        ])
        writer.writerow([
            overall["backend"],
            overall["model"],
            overall["datasets"],
            overall["classes"],
            overall["completed"],
            overall["failed"],
            format_float(float(overall["model_load_time_s"]), 2),
            format_float(float(overall["class_generation_time_s"]), 2),
            format_float(float(overall["end_to_end_time_s"]), 2),
            overall["prompt_tokens"],
            overall["completion_tokens"],
            overall["total_tokens"],
            format_float(float(overall["estimated_total_cost"]), 6),
        ])

    headers = [
        "Backend", "Model", "Datasets", "# Classes", "Completed", "Failed",
        "Model Load (s)", "Generation Time (s)", "End-to-End Time (s)",
        "Prompt Tokens", "Completion Tokens", "Total Tokens", "Est. Cost"
    ]
    values = [
        str(overall["backend"]),
        str(overall["model"]),
        str(overall["datasets"]),
        str(overall["classes"]),
        str(overall["completed"]),
        str(overall["failed"]),
        format_float(float(overall["model_load_time_s"]), 2),
        format_float(float(overall["class_generation_time_s"]), 2),
        format_float(float(overall["end_to_end_time_s"]), 2),
        str(overall["prompt_tokens"]),
        str(overall["completion_tokens"]),
        str(overall["total_tokens"]),
        f"${float(overall['estimated_total_cost']):.6f}",
    ]
    md_path = out_dir / "concept_generation_overall.md"
    md_path.write_text(
        "| " + " | ".join(headers) + " |\n"
        "| " + " | ".join(["---"] * len(headers)) + " |\n"
        "| " + " | ".join(values) + " |\n"
    )


def main() -> None:
    run_start = time.perf_counter()
    args = parse_args()
    if args.api_key is None:
        args.api_key = os.environ.get("SACK_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    if args.backend == "openai-compatible" and not args.base_url:
        raise SystemExit("--base-url is required when --backend=openai-compatible.")
    repo_root = Path(__file__).resolve().parents[1]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "openai-compatible" and not args.dry_run and not args.skip_endpoint_check:
        check_endpoint(args.base_url, args.api_key, args.timeout)

    datasets = [normalize_dataset(item) for item in args.datasets.split(",") if item.strip()]
    overrides = load_class_source_overrides(args.class_source)

    manifest = {
        "backend": args.backend,
        "model": args.model,
        "base_url": args.base_url,
        "datasets": datasets,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "sleep_secs": args.sleep_secs,
        "retries": args.retries,
        "resume": args.resume,
        "limit_classes": args.limit_classes,
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
        "model_cache_dir": args.model_cache_dir,
        "auto_download": args.auto_download,
        "local_files_only": args.local_files_only,
        "input_price_per_1m": args.input_price_per_1m,
        "output_price_per_1m": args.output_price_per_1m,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt_template": build_user_prompt("<CATEGORY>"),
    }
    write_json(args.out_dir / "generation_manifest.json", manifest)

    all_records: List[GenerationRecord] = []
    class_lists: Dict[str, List[str]] = {}
    generator = None
    model_load_time_s = 0.0
    if args.backend == "transformers" and not args.dry_run:
        print(f"[info] Loading transformers model: {args.model}", flush=True)
        load_start = time.perf_counter()
        generator = TransformersGenerator(args)
        model_load_time_s = time.perf_counter() - load_start
        print(f"[info] Model loaded in {model_load_time_s:.2f}s.", flush=True)

    for dataset in datasets:
        source = overrides.get(dataset, DEFAULT_CLASS_SOURCES[dataset])
        if not source.is_absolute():
            source = repo_root / source
        classes = load_classes(source)
        if args.limit_classes is not None:
            classes_to_write = classes[:args.limit_classes]
        else:
            classes_to_write = classes
        class_lists[dataset] = classes_to_write
        write_json(args.out_dir / f"{dataset}_classes.json", classes_to_write)
        print(f"[info] {dataset}: loaded {len(classes_to_write)} classes from {source}", flush=True)
        records, _ = generate_dataset(dataset=dataset, classes=classes, args=args, generator=generator)
        all_records.extend(records)

    write_records_csv(all_records, args.out_dir / "concept_generation_per_class.csv")
    summary_rows = summarize_records(all_records, args)
    write_summary_csv(summary_rows, args.out_dir / "concept_generation_summary.csv")
    write_summary_markdown(summary_rows, args.out_dir / "concept_generation_summary.md")
    write_json(args.out_dir / "concept_generation_summary.json", summary_rows)
    end_to_end_time_s = time.perf_counter() - run_start
    overall = build_overall_summary(
        rows=summary_rows,
        args=args,
        model_load_time_s=model_load_time_s,
        end_to_end_time_s=end_to_end_time_s,
    )
    write_overall_summary(overall, args.out_dir)
    write_json(args.out_dir / "concept_generation_overall.json", overall)

    print("\n===== Concept Generation Cost Summary =====")
    print((args.out_dir / "concept_generation_summary.md").read_text())
    print("===== Overall Timing Summary =====")
    print((args.out_dir / "concept_generation_overall.md").read_text())
    print(f"[done] Wrote outputs to {args.out_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
