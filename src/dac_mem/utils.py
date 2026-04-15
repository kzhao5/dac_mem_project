from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from dateutil import parser as dtparser


MONTHS = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
DAY_WORDS = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|tomorrow|yesterday|tonight|morning|afternoon|evening|weekend|next week|last week|this week|next month|last month)"
TIME_PATTERN = re.compile(rf"\b(?:{MONTHS}|{DAY_WORDS}|\d{{1,2}}[:.]\d{{2}}(?:\s?[ap]m)?|\d{{1,2}}\s?[ap]m|\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?)\b", re.I)
PATH_PATTERN = re.compile(r"(?:[\w.-]+/)+[\w.-]+|\b(?:src|app|lib|docs|tests|config|data)/[\w./-]+\b")
URL_PATTERN = re.compile(r"https?://\S+")
VERSION_PATTERN = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b")
HEDGE_PATTERN = re.compile(r"\b(?:maybe|might|probably|i think|i guess|seems|perhaps|not sure)\b", re.I)
PREFERENCE_PATTERN = re.compile(r"\b(?:i like|i love|i prefer|i dislike|i hate|my favorite|i want|i don't like|i usually prefer)\b", re.I)
PROFILE_PATTERN = re.compile(r"\b(?:i am|i'm|i work as|i live in|my name is|i study|i'm from|i work at|i work in)\b", re.I)
RATIONALE_PATTERN = re.compile(r"\b(?:because|since|so that|the reason|we decided|i chose|i rejected|i prefer not to)\b", re.I)
PLAN_PATTERN = re.compile(r"\b(?:i will|we will|let's|plan to|going to|next step|todo|to-do|follow up|should)\b", re.I)
DEBUG_PATTERN = re.compile(r"\b(?:error|exception|stack trace|debug|bug|temporary|flaky|failed test|traceback|warning)\b", re.I)
NAMEY_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p



def load_json(path: str | Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)



def dump_json(obj, path: str | Path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)



def dump_jsonl(records: Iterable[dict], path: str | Path):
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')



def normalize_text(text: str) -> str:
    text = text.strip().replace('\u2019', "'")
    text = re.sub(r'\s+', ' ', text)
    return text



def simple_sentence_split(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    chunks = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])', text)
    out = [c.strip() for c in chunks if c.strip()]
    return out if out else [text]



def parse_timestamp(ts: str | None) -> datetime:
    if not ts:
        return datetime(1970, 1, 1)
    try:
        return dtparser.parse(ts)
    except Exception:
        return datetime(1970, 1, 1)



def hours_between(ts1: str | None, ts2: str | None) -> float:
    dt1 = parse_timestamp(ts1)
    dt2 = parse_timestamp(ts2)
    return abs((dt2 - dt1).total_seconds()) / 3600.0



def exp_recency(item_ts: str | None, ref_ts: str | None, tau_hours: float = 24.0 * 30.0) -> float:
    age = hours_between(item_ts, ref_ts)
    return math.exp(-age / max(tau_hours, 1e-6))



def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_./'-]+", text.lower())



def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)



def token_f1(pred: str, gold: str) -> float:
    p = tokenize(pred)
    g = tokenize(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p)
    recall = num_same / len(g)
    return 2 * precision * recall / (precision + recall)



def exact_match(pred: str, gold: str) -> float:
    return float(normalize_text(pred).lower() == normalize_text(gold).lower())



def contains_temporal_cue(text: str) -> bool:
    return bool(TIME_PATTERN.search(text))



def has_path_or_artifact(text: str) -> bool:
    return bool(PATH_PATTERN.search(text) or URL_PATTERN.search(text) or VERSION_PATTERN.search(text))



def confidence_from_text(text: str) -> float:
    return 0.65 if HEDGE_PATTERN.search(text) else 1.0



def specificity_score(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    digits = sum(ch.isdigit() for ch in text)
    proper = len(NAMEY_PATTERN.findall(text))
    artifact = 1 if has_path_or_artifact(text) else 0
    temp = 1 if contains_temporal_cue(text) else 0
    raw = min(1.0, 0.04 * len(tokens) + 0.05 * digits + 0.15 * proper + 0.15 * artifact + 0.1 * temp)
    return raw



def stable_hash(text: str) -> str:
    import hashlib
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:12]
