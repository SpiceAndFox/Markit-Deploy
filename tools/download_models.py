#!/usr/bin/env python3
"""
Usage:
  python -m pip install -U modelscope
  python tools/download_models.py --source modelscope --model-root /ckpts/markit/models

Hugging Face is still available as a fallback:
  python -m pip install -U huggingface_hub hf_transfer hf_xet
  python tools/download_models.py --source huggingface --model-root /ckpts/markit/models
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


DEFAULT_MODEL_SOURCE = "modelscope"
DEFAULT_QWEN25_VL_REPO = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_SUBJECT_LLM_REPO = "Qwen/Qwen-7B-Chat"
DEFAULT_MODELSCOPE_SUBJECT_LLM_REPO = "Qwen/Qwen-7b-chat"
DEFAULT_HF_YOLOE_REPO = "jameslahm/yoloe"
DEFAULT_YOLOE_WEIGHTS = "yoloe-v8l-seg.pt"


def parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def apply_env_file(path: Path) -> None:
    for key, value in parse_env_file(path).items():
        os.environ.setdefault(key, value)

    proxy_map = {
        "HTTP_PROXY": "DOWNLOAD_HTTP_PROXY",
        "HTTPS_PROXY": "DOWNLOAD_HTTPS_PROXY",
        "ALL_PROXY": "DOWNLOAD_ALL_PROXY",
        "NO_PROXY": "DOWNLOAD_NO_PROXY",
    }
    for target_key, source_key in proxy_map.items():
        if source_key in os.environ and os.environ[source_key]:
            os.environ.setdefault(target_key, os.environ[source_key])
            os.environ.setdefault(target_key.lower(), os.environ[source_key])


def env_or_default(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "y", "on"}


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_source(value: str) -> str:
    source = value.strip().lower()
    aliases = {
        "hf": "huggingface",
        "hugging-face": "huggingface",
        "ms": "modelscope",
        "model-scope": "modelscope",
    }
    source = aliases.get(source, source)
    if source not in {"modelscope", "huggingface"}:
        raise SystemExit(
            f"Unsupported model source: {value}. Use modelscope or huggingface."
        )
    return source


def source_repo(
    *,
    source: str,
    cli_value: str | None,
    source_env: str,
    generic_env: str,
    default: str,
) -> str:
    if cli_value:
        return cli_value
    if source == "modelscope":
        return env_or_default(source_env, default)
    return env_or_default(generic_env, default)


def require_huggingface_hub():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install it first:\n"
            "  python -m pip install -U huggingface_hub hf_transfer hf_xet"
        ) from exc
    return snapshot_download, hf_hub_download


def require_modelscope():
    try:
        from modelscope import snapshot_download
        from modelscope.hub.file_download import model_file_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: modelscope. Install it first:\n"
            "  python -m pip install -U modelscope"
        ) from exc
    return snapshot_download, model_file_download


def download_snapshot(
    *,
    source: str,
    snapshot_download,
    repo_id: str,
    local_dir: Path,
    token: str | None,
) -> dict[str, str]:
    local_dir.mkdir(parents=True, exist_ok=True)

    if source == "modelscope":
        try:
            path = snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                token=token,
            )
        except TypeError:
            path = snapshot_download(
                model_id=repo_id,
                local_dir=str(local_dir),
                token=token,
            )
    else:
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            token=token,
        )

    return {
        "type": f"{source}_snapshot",
        "repo_id": repo_id,
        "path": str(Path(path).resolve()),
    }


def download_repo_files(
    *,
    source: str,
    file_download,
    repo_id: str,
    filenames: list[str],
    local_dir: Path,
    token: str | None,
) -> list[dict[str, str]]:
    local_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for filename in filenames:
        if source == "modelscope":
            path = file_download(
                model_id=repo_id,
                file_path=filename,
                local_dir=str(local_dir),
                token=token,
            )
        else:
            path = file_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(local_dir),
                token=token,
            )

        records.append(
            {
                "type": f"{source}_file",
                "repo_id": repo_id,
                "filename": filename,
                "path": str(Path(path).resolve()),
            }
        )

    return records


def filename_from_url(url: str) -> str:
    filename = Path(unquote(urlparse(url).path)).name
    if not filename:
        raise SystemExit(f"Cannot infer filename from URL: {url}")
    return filename


def download_url_files(*, urls: list[str], local_dir: Path) -> list[dict[str, str]]:
    local_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for url in urls:
        filename = filename_from_url(url)
        output_path = local_dir / filename
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"[cached] {output_path}")
        else:
            print(f"[download] {url}")
            request = Request(url, headers={"User-Agent": "MarkIt-model-download"})
            with urlopen(request) as response, output_path.open("wb") as output:
                shutil.copyfileobj(response, output)

        records.append(
            {
                "type": "url_file",
                "url": url,
                "filename": filename,
                "path": str(output_path.resolve()),
            }
        )

    return records


def write_manifest(model_root: Path, records: list[dict[str, str]]) -> None:
    manifest_path = model_root / "manifest.json"
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_root": str(model_root.resolve()),
        "models": records,
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[ok] wrote {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download MarkIt deployment models to a local model root."
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Optional env file to read paths, tokens, and proxy settings from.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Download backend: modelscope or huggingface. Defaults to MODEL_SOURCE.",
    )
    parser.add_argument(
        "--model-root",
        default=None,
        help="Target model root. Defaults to MARKIT_MODEL_ROOT or ./models.",
    )
    parser.add_argument(
        "--qwen25-vl-repo",
        default=None,
        help=f"Main VLM repo. Default: {DEFAULT_QWEN25_VL_REPO}",
    )
    parser.add_argument(
        "--subject-llm-repo",
        default=None,
        help=(
            "Subject extractor repo. Hugging Face default: "
            f"{DEFAULT_SUBJECT_LLM_REPO}; ModelScope default: "
            f"{DEFAULT_MODELSCOPE_SUBJECT_LLM_REPO}"
        ),
    )
    parser.add_argument(
        "--yoloe-repo",
        default=None,
        help=(
            "YOLOE repo in the selected backend. For ModelScope, prefer "
            "MODELSCOPE_YOLOE_REPO or --yoloe-urls."
        ),
    )
    parser.add_argument(
        "--yoloe-weights",
        default=None,
        help=f"Comma-separated YOLOE weights. Default: {DEFAULT_YOLOE_WEIGHTS}",
    )
    parser.add_argument(
        "--yoloe-urls",
        default=None,
        help="Comma-separated direct URLs for YOLOE weights from a domestic mirror.",
    )
    parser.add_argument(
        "--yoloe-hf-endpoint",
        default=None,
        help=(
            "Hugging Face-compatible endpoint used only for YOLOE when the main "
            "source is ModelScope, for example https://hf-mirror.com."
        ),
    )
    parser.add_argument(
        "--skip-subject-llm",
        action="store_true",
        help="Skip Qwen-7B subject extractor download.",
    )
    parser.add_argument(
        "--skip-yoloe",
        action="store_true",
        help="Skip YOLOE weight download.",
    )
    args = parser.parse_args()

    apply_env_file(Path(args.env_file))

    source = normalize_source(
        args.source or env_or_default("MODEL_SOURCE", DEFAULT_MODEL_SOURCE)
    )
    if source == "modelscope":
        snapshot_download, file_download = require_modelscope()
        token = os.environ.get("MODELSCOPE_TOKEN") or None
    else:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        snapshot_download, file_download = require_huggingface_hub()
        token = os.environ.get("HF_TOKEN") or None

    model_root = Path(
        args.model_root or env_or_default("MARKIT_MODEL_ROOT", "./models")
    ).expanduser()
    model_root.mkdir(parents=True, exist_ok=True)

    qwen25_vl_repo = source_repo(
        source=source,
        cli_value=args.qwen25_vl_repo,
        source_env="MODELSCOPE_QWEN25_VL_REPO",
        generic_env="QWEN25_VL_REPO",
        default=DEFAULT_QWEN25_VL_REPO,
    )
    subject_llm_repo = source_repo(
        source=source,
        cli_value=args.subject_llm_repo,
        source_env="MODELSCOPE_SUBJECT_LLM_REPO",
        generic_env="SUBJECT_LLM_REPO",
        default=(
            DEFAULT_MODELSCOPE_SUBJECT_LLM_REPO
            if source == "modelscope"
            else DEFAULT_SUBJECT_LLM_REPO
        ),
    )
    hf_yoloe_repo = args.yoloe_repo or env_or_default(
        "YOLOE_REPO", DEFAULT_HF_YOLOE_REPO
    )
    if source == "modelscope":
        yoloe_repo = args.yoloe_repo or os.environ.get("MODELSCOPE_YOLOE_REPO", "")
    else:
        yoloe_repo = hf_yoloe_repo
    yoloe_repo = yoloe_repo.strip()
    yoloe_files = split_csv(
        args.yoloe_weights or env_or_default("YOLOE_WEIGHTS", DEFAULT_YOLOE_WEIGHTS)
    )
    yoloe_urls = split_csv(args.yoloe_urls or os.environ.get("YOLOE_URLS", ""))
    yoloe_hf_endpoint = (
        args.yoloe_hf_endpoint or os.environ.get("YOLOE_HF_ENDPOINT", "")
    ).strip()
    skip_subject_llm = args.skip_subject_llm or env_flag("SKIP_SUBJECT_LLM")
    skip_yoloe = args.skip_yoloe or env_flag("SKIP_YOLOE")

    records: list[dict[str, str]] = []

    print(f"[source] {source}")
    print(f"[download] {qwen25_vl_repo}")
    records.append(
        download_snapshot(
            source=source,
            snapshot_download=snapshot_download,
            repo_id=qwen25_vl_repo,
            local_dir=model_root / "Qwen2.5-VL-7B-Instruct",
            token=token,
        )
    )

    if not skip_subject_llm:
        print(f"[download] {subject_llm_repo}")
        records.append(
            download_snapshot(
                source=source,
                snapshot_download=snapshot_download,
                repo_id=subject_llm_repo,
                local_dir=model_root / "Qwen-7B-Chat",
                token=token,
            )
        )

    if skip_yoloe:
        print("[skip] YOLOE")
    elif yoloe_urls:
        records.extend(
            download_url_files(
                urls=yoloe_urls,
                local_dir=model_root / "YOLOE-Large",
            )
        )
    elif yoloe_repo:
        print(f"[download] {yoloe_repo}: {', '.join(yoloe_files)}")
        records.extend(
            download_repo_files(
                source=source,
                file_download=file_download,
                repo_id=yoloe_repo,
                filenames=yoloe_files,
                local_dir=model_root / "YOLOE-Large",
                token=token,
            )
        )
    elif source == "modelscope" and yoloe_hf_endpoint:
        os.environ["HF_ENDPOINT"] = yoloe_hf_endpoint.rstrip("/")
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        _, hf_file_download = require_huggingface_hub()
        print(
            f"[download] {hf_yoloe_repo} via {os.environ['HF_ENDPOINT']}: "
            f"{', '.join(yoloe_files)}"
        )
        records.extend(
            download_repo_files(
                source="huggingface",
                file_download=hf_file_download,
                repo_id=hf_yoloe_repo,
                filenames=yoloe_files,
                local_dir=model_root / "YOLOE-Large",
                token=os.environ.get("HF_TOKEN") or None,
            )
        )
    else:
        print(
            "[skip] YOLOE: no domestic repo configured. Set MODELSCOPE_YOLOE_REPO, "
            "set YOLOE_URLS, set YOLOE_HF_ENDPOINT, or pass --skip-yoloe."
        )

    write_manifest(model_root, records)


if __name__ == "__main__":
    main()
