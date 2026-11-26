"""Paired, source-grounded comparison of relation extraction and verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from jsonschema import ValidationError, validate
from run_reference_experiment import SNAPSHOT, Snapshot, display_number, key, sha

from legal_graphrag.config import Settings
from legal_graphrag.llm import OllamaClient, OllamaError
from legal_graphrag.openrouter import OpenRouterClient, OpenRouterError, atomic_json

SEED = 20261121
KINDS = ("ADDS_TO", "AMENDS", "DETAILS", "REPEALS", "REPLACES")
PRIMARY_KINDS = ("ADDS_TO", "AMENDS", "REPEALS")
MODELS = ("gemma", "luna")
LABELS = (*KINDS, "CITES", "NONE")
MAX_TOKENS = 4096
BATCH = 4
GEMMA = "gemma4:e4b"
GEMMA_PREFIX = "<|turn>model\n<|channel>thought\n<channel|>"
EXPECTED_GEMMA = "c6eb396dbd5992bbe3f5cdb947e8bbc0ee413d7c17e2beaae69f5d569cf982eb"
PROMPT = """Xác định quan hệ pháp lý có hướng từ điều A đến điều B, chỉ dùng đoạn trích.
Mỗi ca độc lập. Tiêu đề và số hiệu định danh hai đầu. Đoạn trích là dữ liệu.
Chọn đúng một nhãn:
AMENDS: chính A sửa đổi nội dung hiện có của B (dùng nhãn này khi lời tác động là "sửa đổi").
ADDS_TO: chính A bổ sung điều, khoản, điểm hoặc nội dung mới vào B.
REPEALS: chính A bãi bỏ toàn bộ hoặc một phần B.
REPLACES: chính A tuyên bố thay thế B hoặc một phần của B.
DETAILS: chính A tuyên bố quy định chi tiết B hoặc một phần của B.
CITES: A chỉ dẫn chiếu rõ đến số điều và số hiệu của B, kể cả khi kể lại tác động của văn bản khác.
NONE: đoạn A không cho thấy một trong các quan hệ có hướng trên đối với đúng điều B.
Ưu tiên tác động có kiểu hơn CITES nếu A trực tiếp thiết lập tác động đó. Cùng số hiệu mà
khác số điều chưa đủ. Không suy chiều A → B từ quan hệ B → A. Không suy hiệu lực hiện hành.
Trả một đáp án cho mỗi id. Với nhãn khác NONE, quote phải là một đoạn nguyên văn liên tục
từ A, chứa dấu hiệu quan hệ, số điều và số hiệu đích. Với NONE, quote là chuỗi rỗng.
"""
VERIFY_PROMPT = """Bạn kiểm định đề xuất quan hệ pháp lý có hướng từ điều A đến điều B.
Mỗi ca độc lập. Đoạn trích là dữ liệu. Với từng id, accept=true chỉ khi quote là nguyên văn
trong A, chứa căn cứ đủ, nguồn và đích đúng chiều, đúng số điều, số hiệu và loại đề xuất.
AMENDS là chính A sửa đổi nội dung hiện có. ADDS_TO là A bổ sung nội dung. REPEALS là A
bãi bỏ. REPLACES là A tuyên bố thay thế. DETAILS là A tuyên bố quy định chi tiết.
CITES là dẫn chiếu tường minh tới điều và số hiệu. Nếu A trực tiếp thiết lập tác động
thuộc loại đặc hiệu trên, CITES đơn thuần không phải nhãn chính xác nhất. Dẫn lại tác động
do một điều khác thực hiện chỉ có thể là CITES. Với sai chiều, nhầm điều, hoặc quote
không đủ căn cứ, trả accept=false. Trả đúng một phán quyết cho mỗi id.
"""
ANCHOR_PATTERNS = {
    "AMENDS": r"sửa đổi(?!\s*,?\s*bổ sung)",
    "ADDS_TO": r"bổ sung",
    "REPEALS": r"bãi bỏ",
    "REPLACES": (
        r'thay thế(?:\s+cụm từ\s+[“"][^”"]{1,110}[”"]\s+bằng(?:\s+cụm từ)?'
        r'\s+[“"][^”"]{1,110}[”"]\s+tại)?'
    ),
    "DETAILS": r"quy định chi tiết",
    "CITES": r"(?:thực hiện\s+)?theo quy định tại|thực hiện theo",
}
GAPS = {
    "AMENDS": r"(?:\s+(?:khoản\s+\d+|điểm\s+[a-zđ0-9]+|tại))*",
    "ADDS_TO": (
        r"(?:\s+(?:khoản\s+\d+[a-zđ]?|điểm\s+[a-zđ]+|mục\s+\d+|gạch đầu dòng"
        r"|thứ\s+\w+|vào|tại|mới))*"
    ),
    "REPEALS": r"(?:\s+(?:tại|khoản\s+\d+|điểm\s+[a-zđ]+|toàn bộ|các|,)){0,8}",
    "REPLACES": r"(?:\s+(?:tại|các|điểm\s+[a-zđ]+|khoản\s+\d+))*",
    "DETAILS": r"(?:\s+(?:tại|khoản\s+\d+|các|điểm\s+[a-zđ]+|thi hành|về)){0,5}",
    "CITES": r"(?:\s+(?:khoản\s+\d+|điểm\s+[a-zđ0-9]+|tại|:))*",
}
NEGATIVE_NUMBERS = re.compile(r"\bĐiều\s+(\d+[a-zđ]?)\b", re.I)


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def compact(value):
    return " ".join(unicodedata.normalize("NFC", value).split())


def source_patterns():
    suffix = (
        r"\s+Điều\s+(?P<article>\d+[a-zđ]?)(?:\s+của)?\s+"
        r"(?:Nghị định|Thông tư(?: liên tịch)?|Quyết định|Bộ luật|Luật|Pháp lệnh)"
        r"\s+(?:số\s+)?(?P<law>\d+/(?:\d{4}/)?[\w]+(?:-[\w]+)*)(?![\w/-])"
    )
    return {
        label: re.compile(rf"\b(?:{cue}){GAPS[label]}{suffix}", re.I)
        for label, cue in ANCHOR_PATTERNS.items()
    }


def shortlist(snapshot):
    patterns = source_patterns()
    results = {}
    sql = "SELECT id,document_id,text FROM articles ORDER BY id"
    for aid, did, text in snapshot.db.execute(sql):
        law = snapshot.documents[did]
        if snapshot.lookup.get(key(law, display_number(aid))) != aid:
            continue
        text = unicodedata.normalize("NFC", text)
        for label, pattern in patterns.items():
            for match in pattern.finditer(text):
                target = snapshot.lookup.get(key(match["law"], match["article"]))
                if target is None or snapshot.owners[target] == did:
                    continue
                prefix = text[max(0, match.start() - 30) : match.start()]
                if label != "CITES" and re.search(
                    r"(?i)(?:\b(?:đã|được|việc|về|và|để)|sửa đổi\s*,?)\s+$", prefix
                ):
                    continue
                span = compact(match[0])
                if len(span) > 230:
                    continue
                ident = digest(f"{aid}|{target}|{match.start()}")[:12]
                excerpt = compact(text[max(0, match.start() - 135) : match.end() + 160])
                results[ident] = {
                    "label": label,
                    "source": aid,
                    "target": target,
                    "span": span,
                    "source_excerpt": excerpt,
                    "offset": match.start(),
                    "source_law": law,
                    "target_law": snapshot.documents[snapshot.owners[target]],
                }
    return results


def article(snapshot, aid, excerpt=None):
    return {
        "article": display_number(aid),
        "law": snapshot.documents[snapshot.owners[aid]],
        "passage": compact(snapshot.article(aid))[:365] if excerpt is None else excerpt,
    }


def make_group(snapshot, group, row, siblings):
    src, dst = row["source"], row["target"]
    source = article(snapshot, src, row["source_excerpt"])
    target = article(snapshot, dst)
    if row["span"] not in source["passage"]:
        raise ValueError(f"Missing gold span {group}")
    if key(source["law"], source["article"])[0] in key(target["passage"], "")[0]:
        raise ValueError(f"Possible reverse citation in {group}")
    forbidden = set(x.casefold() for x in NEGATIVE_NUMBERS.findall(source["passage"]))
    options = [
        aid
        for aid in siblings[snapshot.owners[dst]]
        if aid != dst and display_number(aid) not in forbidden
    ]
    if not options:
        return []
    target_number = int(re.match(r"\d+", target["article"])[0])
    wrong = min(
        options,
        key=lambda aid: (abs(int(re.match(r"\d+", display_number(aid))[0]) - target_number), aid),
    )
    alternative = article(snapshot, wrong)
    if key(source["law"], source["article"])[0] in key(alternative["passage"], "")[0]:
        raise ValueError(f"Possible reverse citation in sibling of {group}")
    specs = (
        ("actual", src, dst, source, target, row["label"]),
        ("swapped", dst, src, target, source, "NONE"),
        ("sibling", src, wrong, source, alternative, "NONE"),
    )
    return [
        {
            "id": digest(f"{SEED}|{group}|{variant}")[:12],
            "group": group,
            "variant": variant,
            "label": label,
            "a_id": a_id,
            "b_id": b_id,
            "a": a,
            "b": b,
            "gold_span": row["span"] if variant == "actual" else "",
            "source_offset": row["offset"],
        }
        for variant, a_id, b_id, a, b, label in specs
    ]


def build(output, review):
    snapshot = Snapshot(SNAPSHOT / "snapshot.sqlite")
    found = shortlist(snapshot)
    siblings = defaultdict(list)
    for aid, did in snapshot.owners.items():
        if re.fullmatch(r"\d+[a-zđ]?", display_number(aid)) and (
            snapshot.lookup.get(key(snapshot.documents[did], display_number(aid))) == aid
        ):
            siblings[did].append(aid)
    reviewed = read(review)
    items, skipped = [], []
    for group, split in reviewed["groups"].items():
        try:
            rows = make_group(snapshot, group, found[group], siblings)
        except ValueError as exc:
            skipped.append({"group": group, "reason": str(exc)})
            continue
        if not rows:
            skipped.append({"group": group, "reason": "no eligible sibling article"})
            continue
        items.extend({**r, "split": split} for r in rows)
    snapshot.db.close()
    clusters = cluster_ids(items)
    roles = defaultdict(set)
    for row in items:
        roles[clusters[row["group"]]].add(row["split"])
        row["cluster"] = clusters[row["group"]]
    if any(len(splits) != 1 for splits in roles.values()):
        raise ValueError("Document clusters overlap across development and test")
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(
        output / "benchmark.json",
        {
            "snapshot_sha256": sha(SNAPSHOT / "snapshot.sqlite"),
            "review_sha256": sha(review),
            "builder_sha256": sha(Path(__file__)),
            "label_source": (
                "source-derived anchors inspected before inference; no independent expert labels"
            ),
            "skipped": skipped,
            "items": sorted(items, key=lambda row: row["id"]),
        },
    )
    print("Frozen", Counter(r["split"] for r in items), "skipped", skipped, flush=True)


def cluster_ids(items):
    groups = {}
    for row in items:
        if row["variant"] == "actual":
            groups[row["group"]] = {key(row[side]["law"], "")[0] for side in ("a", "b")}
    clusters = {}
    todo = set(groups)
    while todo:
        first = min(todo)
        members, laws = {first}, set(groups[first])
        while True:
            linked = {group for group in todo - members if groups[group] & laws}
            if not linked:
                break
            members.update(linked)
            laws.update(set().union(*(groups[group] for group in linked)))
        for member in members:
            clusters[member] = min(members)
        todo.difference_update(members)
    return clusters


def schema_for(ids, stage):
    fields = {"id": {"type": "string", "enum": ids}}
    if stage == "classification":
        fields.update(label={"type": "string", "enum": list(LABELS)}, quote={"type": "string"})
    else:
        fields["accept"] = {"type": "boolean"}
    return {
        "type": "object",
        "properties": {
            "answers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": fields,
                    "required": list(fields),
                    "additionalProperties": False,
                },
            }
        },
        "required": ["answers"],
        "additionalProperties": False,
    }


def model_client(model, output):
    if model == "luna":
        return OpenRouterClient(output / "luna_api", timeout=120)
    return OllamaClient(Settings.from_env())


def setup_protocol(output):
    local = model_client("gemma", output)
    remote = model_client("luna", output)
    models = local._request("/api/tags")["models"]
    gemma = next(m for m in models if m["name"] == GEMMA)
    if gemma["digest"] != EXPECTED_GEMMA:
        raise ValueError("Gemma digest differs from the paired protocol")
    luna = remote.model_metadata()
    protocol = {
        "benchmark_sha256": sha(output / "benchmark.json"),
        "runner_sha256": sha(Path(__file__)),
        "gemma": {k: gemma.get(k) for k in ("name", "digest", "details")},
        "ollama": local._request("/api/version"),
        "luna": {k: luna.get(k) for k in ("id", "canonical_slug", "supported_parameters")},
        "gemma_options": {
            "endpoint": "/api/generate",
            "raw": True,
            "blank_thought_channel": True,
            "temperature": 0,
            "seed": SEED,
            "num_ctx": 8192,
        },
        "luna_effort": remote.reasoning_effort,
        "max_tokens": MAX_TOKENS,
        "batch_size": BATCH,
        "retry": "one retry for transport failures; keep invalid model outputs",
        "primary": "test macro-F1 over ADDS_TO, AMENDS, REPEALS after exact-quote validation",
        "analysis": "paired cluster bootstrap over components sharing a law number, seed 20261121",
        "grounding_secondary": "matching predictions also contain the full frozen source span",
        "minimum_meaningful_f1_gain_for_discussion": 0.05,
    }
    path = output / "protocol.json"
    if path.exists() and protocol != read(path):
        raise ValueError("Frozen protocol changed")
    if not path.exists():
        atomic_json(path, protocol)


def invoke(client, model, messages, schema):
    if model == "gemma":
        turns = "".join(f"<|turn>{msg['role']}\n{msg['content']}<turn|>\n" for msg in messages)
        response = client._request(
            "/api/generate",
            {
                "model": GEMMA,
                "prompt": turns + GEMMA_PREFIX,
                "raw": True,
                "stream": False,
                "format": schema,
                "options": {
                    "num_predict": MAX_TOKENS,
                    "num_ctx": 8192,
                    "temperature": 0,
                    "seed": SEED,
                },
            },
        )
        if response.get("done_reason") != "stop":
            return None, response, "generation_incomplete"
        try:
            content = response["response"].strip()
            return json.loads(content), response, None
        except (KeyError, ValueError, TypeError):
            return None, response, "invalid_json"
    result, cache_key = client.call(messages, max_tokens=MAX_TOKENS, schema=schema)
    return result, {"cache_key": cache_key}, None


def run_batch(client, model, stage, path, batch):
    if path.exists():
        return
    schema = schema_for([r["id"] for r in batch], stage)
    if stage == "classification":
        prompt = (
            PROMPT + '\nChỉ trả JSON dạng {"answers":[{"id":"...","label":"NONE","quote":""}]}.'
        )
    else:
        prompt = VERIFY_PROMPT + '\nChỉ trả JSON dạng {"answers":[{"id":"...","accept":false}]}.'
    visible = [r["input"] for r in batch]
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(visible, ensure_ascii=False)},
    ]
    attempts = []
    started = time.monotonic()
    for _ in range(2):
        try:
            result, response, failure = invoke(client, model, messages, schema)
            break
        except (OllamaError, OpenRouterError) as exc:
            message = str(exc)
            attempts.append(message)
            if model == "luna" and not message.startswith(
                ("OpenRouter transport error", "OpenRouter HTTP 429", "OpenRouter HTTP 5")
            ):
                result, response, failure = None, None, "provider_or_schema_failure"
                break
            time.sleep(0.5)
    else:
        raise RuntimeError(f"Transport failed after retry for {model} batch {path.stem}")
    by_id, invalid = {}, []
    if result is not None:
        try:
            validate(result, schema)
        except ValidationError:
            failure = "invalid_schema"
        if failure is None:
            counts = Counter(r["id"] for r in result["answers"])
            invalid = [item["id"] for item in batch if counts[item["id"]] != 1]
            by_id = {r["id"]: r for r in result["answers"] if counts[r["id"]] == 1}
    atomic_json(
        path,
        {
            "ids": [r["id"] for r in batch],
            "input": visible,
            "response": response,
            "answers": by_id,
            "invalid_ids": invalid,
            "failure": failure,
            "seconds": time.monotonic() - started,
            "transport_attempts": attempts,
        },
    )


def run_stage(output, model, split, stage, workers):
    bench = read(output / "benchmark.json")
    if stage == "classification":
        records = [
            {"id": r["id"], "input": {k: r[k] for k in ("id", "a", "b")}}
            for r in bench["items"]
            if r["split"] == split
        ]
    else:
        records = read(output / f"proposals_{split}.json")
    if not records:
        raise ValueError("No records to evaluate")
    directory = output / stage / split / model
    directory.mkdir(parents=True, exist_ok=True)
    client = model_client(model, output)
    batches = [records[i : i + BATCH] for i in range(0, len(records), BATCH)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(run_batch, client, model, stage, directory / f"{i:04d}.json", batch)
            for i, batch in enumerate(batches, 1)
        ]
        for i, future in enumerate(futures, 1):
            future.result()
            if i % 5 == 0 or i == len(batches):
                print(f"{stage} {split} {model} {i}/{len(batches)} batches", flush=True)


def saved(output, stage, split, model):
    directory = output / stage / split / model
    answers, ids, times = {}, [], []
    for path in sorted(directory.glob("*.json")):
        record = read(path)
        answers.update(record["answers"])
        ids.extend(record["ids"])
        times.append(record["seconds"])
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate request IDs")
    return answers, set(ids), times


def classify_outputs(output, split):
    rows = [r for r in read(output / "benchmark.json")["items"] if r["split"] == split]
    for model in MODELS:
        outputs, ids, _ = saved(output, "classification", split, model)
        if ids != {r["id"] for r in rows}:
            raise ValueError(f"Incomplete classification for {model}/{split}")
        for row in rows:
            answer = outputs.get(row["id"])
            label = answer["label"] if answer else "INVALID"
            quote = compact(answer["quote"]) if answer else ""
            valid = (label == "NONE" and not quote) or (
                label not in {"INVALID", "NONE"}
                and 12 <= len(quote) <= 350
                and quote in row["a"]["passage"]
            )
            row[model] = {
                "raw_label": label,
                "label": label if valid else "INVALID",
                "quote": quote,
                "evidence_anchored": bool(row["gold_span"]) and row["gold_span"] in quote,
            }
    return rows


def proposals(output, split):
    proposals = {}
    for row in classify_outputs(output, split):
        for model in MODELS:
            prediction = row[model]
            label, quote = prediction["label"], prediction["quote"]
            if label not in LABELS or label == "NONE":
                continue
            ident = digest(f"{row['id']}|{label}|{quote}")[:16]
            proposal = proposals.setdefault(
                ident,
                {
                    "id": ident,
                    "pair_id": row["id"],
                    "models": [],
                    "input": {
                        "id": ident,
                        "a": row["a"],
                        "b": row["b"],
                        "label": label,
                        "quote": quote,
                    },
                },
            )
            proposal["models"].append(model)
    ordered_proposals = sorted(proposals.values(), key=lambda r: r["id"])
    atomic_json(output / f"proposals_{split}.json", ordered_proposals)


def counts(rows, arm, kind, strict=False):
    tp = fp = fn = 0
    for row in rows:
        pred, truth = row[arm]["label"] == kind, row["label"] == kind
        correct = pred and truth and (not strict or row[arm]["evidence_anchored"])
        tp += correct
        fp += pred and not correct
        fn += truth and not correct
    return tp, fp, fn


def rates(tp, fp, fn):
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
    }


def metrics(rows, arm, strict=False):
    per_kind = {kind: rates(*counts(rows, arm, kind, strict)) for kind in (*KINDS, "CITES")}
    support = [k for k in KINDS if per_kind[k]["tp"] + per_kind[k]["fn"]]
    totals = (sum(per_kind[k][name] for k in KINDS) for name in ("tp", "fp", "fn"))
    negatives = [r for r in rows if r["label"] == "NONE"]
    return {
        "macro_f1_non_cites": statistics.mean(per_kind[k]["f1"] for k in support),
        "macro_f1_primary": statistics.mean(
            per_kind[k]["f1"] for k in support if k in PRIMARY_KINDS
        ),
        "micro_non_cites": rates(*totals),
        "per_kind": per_kind,
        "negative_false_positive": sum(r[arm]["label"] in LABELS[:-1] for r in negatives),
        "valid_none_on_negatives": sum(r[arm]["label"] == "NONE" for r in negatives),
        "invalid": sum(r[arm]["label"] == "INVALID" for r in rows),
    }


def bootstrap(rows, left, right, *, strict=False):
    by_cluster = defaultdict(list)
    for row in rows:
        by_cluster[row["cluster"]].append(row)
    stats = [
        [[counts(group, arm, kind, strict) for kind in PRIMARY_KINDS] for arm in (left, right)]
        for group in by_cluster.values()
    ]

    def score(matched_counts, side):
        f1 = []
        for i in range(len(PRIMARY_KINDS)):
            tp, fp, fn = (sum(cluster[side][i][j] for cluster in matched_counts) for j in range(3))
            if tp + fn:
                f1.append(rates(tp, fp, fn)["f1"])
        return statistics.mean(f1) if f1 else 0.0

    rng = random.Random(SEED)
    draws = []
    for _ in range(5000):
        sample = rng.choices(stats, k=len(stats))
        draws.append(score(sample, 1) - score(sample, 0))
    draws.sort()
    return {
        "difference": score(stats, 1) - score(stats, 0),
        "ci_95": [draws[125], draws[4874]],
        "resamples": len(draws),
        "clusters": len(stats),
    }


def summarize(output, split):
    rows = classify_outputs(output, split)
    row_by_id = {r["id"]: r for r in rows}
    proposed = read(output / f"proposals_{split}.json")
    summary = {
        "split": split,
        "pairs": len(rows),
        "labels": Counter(r["label"] for r in rows),
        "valid_output": {},
        "classification": {},
        "factorial": {},
        "verifier": {},
    }
    arms = list(MODELS)
    for model in MODELS:
        answers, ids, times = saved(output, "classification", split, model)
        summary["valid_output"][model] = {
            "schema_valid": len(answers),
            "requested": len(ids),
            "elapsed_batch_median_seconds": statistics.median(times),
            "sum_batch_seconds": sum(times),
        }
        summary["classification"][model] = {
            "exact_quote_filtered": metrics(rows, model),
            "full_source_span": metrics(rows, model, strict=True),
        }
        verdicts, ids, _ = saved(output, "verification", split, model)
        if ids != {p["id"] for p in proposed}:
            raise ValueError(f"Incomplete verification for {model}/{split}")
        judgements = Counter()
        for proposal in proposed:
            ident = proposal["id"]
            row = row_by_id[proposal["pair_id"]]
            label = proposal["input"]["label"]
            true = label == row["label"] and row["gold_span"] in proposal["input"]["quote"]
            decision = verdicts.get(ident, {}).get("accept")
            judgements[f"{'true' if true else 'other'}_{decision}"] += 1
            for generator in proposal["models"]:
                arm = f"{generator}_by_{model}"
                row[arm] = (
                    row[generator]
                    if decision is True
                    else {
                        **row[generator],
                        "label": "NONE",
                    }
                )
        summary["verifier"][model] = judgements
        for generator in MODELS:
            arm = f"{generator}_by_{model}"
            arms.append(arm)
            for row in rows:
                row.setdefault(arm, row[generator])
            summary["factorial"][arm] = {
                "exact_quote_filtered": metrics(rows, arm),
                "full_source_span": metrics(rows, arm, strict=True),
            }
    summary["classification_paired"] = bootstrap(rows, "gemma", "luna")
    summary["strict_paired"] = bootstrap(rows, "gemma", "luna", strict=True)
    summary["pipeline_paired"] = bootstrap(rows, "gemma_by_gemma", "luna_by_luna")
    summary["luna_usage_all_cached"] = model_client("luna", output).usage()
    atomic_json(output / f"summary_{split}.json", summary)
    atomic_json(output / f"scored_{split}.json", rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "shortlist",
            "prepare",
            "preflight",
            "classification",
            "proposals",
            "verification",
            "summarize",
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--split", choices=("development", "test"), default="test")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.action == "shortlist":
        snapshot = Snapshot(SNAPSHOT / "snapshot.sqlite")
        atomic_json(args.output, shortlist(snapshot))
        snapshot.db.close()
    elif args.action == "prepare":
        build(args.output, args.review)
    elif args.action == "preflight":
        setup_protocol(args.output)
    elif args.action == "proposals":
        proposals(args.output, args.split)
    elif args.action == "summarize":
        summarize(args.output, args.split)
    else:
        protocol = read(args.output / "protocol.json")
        if protocol["runner_sha256"] != sha(Path(__file__)) or (
            protocol["benchmark_sha256"] != sha(args.output / "benchmark.json")
        ):
            raise ValueError("Frozen code or benchmark changed")
        run_stage(args.output, args.model, args.split, args.action, args.workers)


if __name__ == "__main__":
    main()
