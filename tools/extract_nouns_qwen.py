#!/usr/bin/env python3
# Usage:
#   python tools/extract_nouns_qwen.py \
#     --input_txt /data/Raw/Charades-STA/raw/charades_sta_test.txt \
#     --model_path /models/local/Qwen-7B-Chat \
#     --output_json /data/MarkIt/nouns_qwen.json
#
# Docker Compose example:
#   docker compose run --rm markit bash -lc '
#   python tools/extract_nouns_qwen.py \
#     --input_txt /data/Raw/Charades-STA/charades_sta_test.txt \
#     --model_path /models/local/Qwen-7B-Chat \
#     --output_json /data/MarkIt/nouns_qwen.json'
#
# Output:
#   Writes a JSON mapping from record_id to noun dict, e.g.:
#   {"001YG_1": {"1": "person", "2": "door"}, ...}
#   Pass this to prepare_charades_sta.py via --nouns_json.
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_: Any):
        return iterable


SYSTEM_PROMPT = (
    "Extract target nouns for visual grounding from an action description. "
    "Return a JSON array of at most 3 short English noun phrases for visible people, "
    "objects, places, or body parts that are physically present in the scene. "
    "Prefer concrete nouns that can be segmented in video frames. "
    "Do not include verbs, actions, timestamps, explanations, or markdown."
)


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def parse_charades_sta_line(line: str, line_number: int) -> dict[str, Any] | None:
    raw = line.strip()
    if not raw or raw.startswith("#"):
        return None
    if "##" not in raw:
        raise ValueError(f"Line {line_number}: expected '<video> <start> <end>##<query>'")

    meta, query = raw.split("##", 1)
    parts = meta.split()
    if len(parts) != 3:
        raise ValueError(f"Line {line_number}: expected three metadata fields before ##")

    video_id, start_text, end_text = parts
    if float(end_text) < float(start_text):
        raise ValueError(f"Line {line_number}: end_time < start_time")

    return {"video_id": video_id, "query": normalize_text(query)}


def load_charades_sta(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").readlines(), start=1):
        item = parse_charades_sta_line(line, line_number)
        if item is not None:
            records.append(item)
    return records


def build_record_ids(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    per_video_counts: dict[str, int] = defaultdict(int)
    result = []
    for item in records:
        video_id = item["video_id"]
        per_video_counts[video_id] += 1
        record_id = f"{video_id}_{per_video_counts[video_id]}"
        result.append({**item, "id": record_id})
    return result


def clean_noun(text: str) -> str:
    noun = re.sub(r"^[\s\d\-\*\.\)\(]+", "", str(text).strip())
    noun = re.sub(r"\s+", " ", noun)
    noun = noun.strip(" \t\r\n\"'`[]{}")
    return noun


def extract_nouns_from_response(response: str, max_nouns: int) -> dict[str, str]:
    text = response.strip()

    nouns: list[str] = []
    try:
        payload = json.loads(text)
        if isinstance(payload, list):
            nouns = [clean_noun(item) for item in payload]
        elif isinstance(payload, dict):
            value = payload.get("nouns", payload.get("objects", payload.get("items", [])))
            if isinstance(value, list):
                nouns = [clean_noun(item) for item in value]
            elif isinstance(value, dict):
                nouns = [clean_noun(item) for _, item in sorted(value.items())]
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*?\]", text)
        if match:
            try:
                payload = json.loads(match.group(0))
                if isinstance(payload, list):
                    nouns = [clean_noun(item) for item in payload]
            except json.JSONDecodeError:
                nouns = []

    if not nouns:
        nouns = [clean_noun(item) for item in re.split(r"[,;\n]+", text)]

    nouns = [noun for noun in nouns if noun and len(noun) > 1]
    seen = set()
    unique = []
    for noun in nouns:
        lower = noun.lower()
        if lower not in seen:
            seen.add(lower)
            unique.append(noun)
            if max_nouns > 0 and len(unique) >= max_nouns:
                break
    return {str(idx): noun for idx, noun in enumerate(unique, start=1)}


def build_prompt(query: str) -> str:
    return (
        f"Description: {query}\n"
        "Return only a JSON array, for example: [\"person\", \"door\"]."
    )


def build_fallback_chat_prompt(query: str) -> str:
    return f"{SYSTEM_PROMPT}\n\n{build_prompt(query)}"


def format_chat_texts(tokenizer: Any, records: list[dict[str, Any]]) -> list[str] | None:
    messages_batch = []
    for rec in records:
        messages_batch.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(rec["query"])},
        ])

    try:
        return [
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in messages_batch
        ]
    except Exception:
        return None


def load_existing_output(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Existing output is not a JSON object: {path}")

    existing: dict[str, dict[str, str]] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            existing[str(key)] = {
                str(noun_id): str(noun).strip()
                for noun_id, noun in value.items()
                if str(noun).strip()
            }
    return existing


def resolve_device_map(device: str) -> str | dict[str, str]:
    normalized = str(device).strip().lower()
    if normalized == "auto":
        return "auto"
    if normalized == "cpu":
        return {"": "cpu"}
    return {"": device}


def resolve_torch_dtype(torch_module: Any, device: str):
    return torch_module.float32 if str(device).strip().lower() == "cpu" else torch_module.bfloat16


def main() -> None:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: torch/transformers. Run inside the Docker image or rebuild it."
        ) from exc

    parser = argparse.ArgumentParser(
        description="Extract nouns from Charades-STA queries using Qwen-7B-Chat."
    )
    parser.add_argument("--input_txt", required=True, help="Path to charades_sta_test.txt")
    parser.add_argument("--model_path", required=True, help="Path to Qwen-7B-Chat model directory")
    parser.add_argument("--output_json", required=True, help="Output JSON mapping record_id -> nouns")
    parser.add_argument("--max_nouns", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda:0", help="cuda device, cpu, or auto")
    parser.add_argument("--batch_size", type=int, default=4, help="Queries per batch")
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing output and regenerate all records")
    args = parser.parse_args()

    input_txt = Path(args.input_txt)
    if not input_txt.exists():
        raise FileNotFoundError(f"Input file not found: {input_txt}")

    raw_records = load_charades_sta(input_txt)
    records = build_record_ids(raw_records)
    print(f"Loaded {len(records)} records from {input_txt}")

    output_path = Path(args.output_json)
    nouns_map: dict[str, dict[str, str]] = {} if args.overwrite else load_existing_output(output_path)
    if nouns_map:
        print(f"Loaded {len(nouns_map)} existing records from {output_path}")

    print(f"Loading Qwen-7B-Chat from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=resolve_torch_dtype(torch, args.device),
        device_map=resolve_device_map(args.device),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if hasattr(model, "generation_config"):
        model.generation_config.do_sample = False

    device = model.device
    pending_records = [record for record in records if record["id"] not in nouns_map]
    total = len(pending_records)

    for batch_start in tqdm(range(0, total, args.batch_size), desc="extracting nouns"):
        batch = pending_records[batch_start : batch_start + args.batch_size]

        texts = format_chat_texts(tokenizer, batch)
        if texts is None:
            if not hasattr(model, "chat"):
                raise RuntimeError(
                    "Tokenizer has no usable chat template and model has no chat() fallback."
                )
            for rec in batch:
                response = model.chat(
                    tokenizer,
                    build_fallback_chat_prompt(rec["query"]),
                    history=None,
                )
                if isinstance(response, tuple):
                    response = response[0]
                nouns_map[rec["id"]] = extract_nouns_from_response(str(response), args.max_nouns)
            continue

        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        responses = tokenizer.batch_decode(
            [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        for rec, response in zip(batch, responses):
            nouns = extract_nouns_from_response(response, args.max_nouns)
            nouns_map[rec["id"]] = nouns

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        ordered = {record["id"]: nouns_map.get(record["id"], {}) for record in records}
        json.dump(ordered, f, ensure_ascii=False, indent=2)
        f.write("\n")

    non_empty = sum(1 for v in nouns_map.values() if v)
    print(f"Wrote {len(nouns_map)} records to {output_path} ({non_empty} with nouns)")

    total_nouns = sum(len(v) for v in nouns_map.values())
    print(f"Total nouns extracted: {total_nouns}")


if __name__ == "__main__":
    main()
