from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests

from .schema import Example, Turn
from .utils import load_json, normalize_text

LOCOMO_URL = 'https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json'
LONGMEMEVAL_URLS = {
    's_cleaned': 'https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json',
    'm_cleaned': 'https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json',
    'oracle': 'https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json',
}


def maybe_download_dataset(dataset: str, out_path: str, variant: str = 's_cleaned') -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return out
    if dataset == 'locomo':
        url = LOCOMO_URL
    elif dataset == 'longmemeval':
        url = LONGMEMEVAL_URLS[variant]
    else:
        raise ValueError(f'Unsupported dataset: {dataset}')
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    out.write_bytes(r.content)
    return out



def _flatten_locomo_conversation(conversation: Dict) -> List[Turn]:
    turns: List[Turn] = []
    speaker_a = conversation.get('speaker_a', 'speaker_a')
    speaker_b = conversation.get('speaker_b', 'speaker_b')
    session_keys = [k for k in conversation.keys() if k.startswith('session_') and not k.endswith('_date_time')]
    session_keys = sorted(session_keys, key=lambda x: int(x.split('_')[1]))
    for sk in session_keys:
        sid = sk
        ts = conversation.get(f'{sk}_date_time', '')
        for t in conversation.get(sk, []):
            speaker = t.get('speaker', '')
            if speaker == speaker_a:
                speaker = 'speaker_a'
            elif speaker == speaker_b:
                speaker = 'speaker_b'
            turn = Turn(
                turn_id=str(t.get('dia_id', f'{sid}_{len(turns)}')),
                session_id=sid,
                speaker=speaker,
                text=normalize_text(t.get('text', '')),
                timestamp=str(ts),
                metadata={k: v for k, v in t.items() if k not in {'text', 'speaker', 'dia_id'}},
            )
            turns.append(turn)
    return turns



def load_locomo(path: str | Path) -> List[Example]:
    raw = load_json(path)
    examples: List[Example] = []
    for conv in raw:
        turns = _flatten_locomo_conversation(conv['conversation'])
        turn_session_map = {t.turn_id: t.session_id for t in turns}
        qas = conv.get('qa', [])
        for idx, qa in enumerate(qas):
            evidence_turns = [str(x) for x in qa.get('evidence', [])]
            evidence_sessions = sorted({turn_session_map[e] for e in evidence_turns if e in turn_session_map})
            examples.append(Example(
                example_id=f"{conv.get('sample_id', 'locomo')}_{idx}",
                dataset='locomo',
                question=normalize_text(qa['question']),
                answer=normalize_text(qa['answer']),
                question_type=qa.get('category', 'qa'),
                turns=turns,
                question_timestamp=turns[-1].timestamp if turns else None,
                evidence_turn_ids=evidence_turns,
                evidence_session_ids=evidence_sessions,
                metadata={'sample_id': conv.get('sample_id'), 'category': qa.get('category', '')},
            ))
    return examples



def _flatten_longmemeval_sessions(sessions: List[List[Dict]], session_ids: List[str], session_dates: List[str]):
    turns: List[Turn] = []
    for sid, ts, sess in zip(session_ids, session_dates, sessions):
        for idx, turn in enumerate(sess):
            turn_id = f'{sid}_{idx}'
            if 'has_answer' in turn:
                turn_id = f'{turn_id}_answer'
            turns.append(Turn(
                turn_id=turn_id,
                session_id=str(sid),
                speaker=turn.get('role', 'user'),
                text=normalize_text(turn.get('content', '')),
                timestamp=str(ts),
                metadata={k: v for k, v in turn.items() if k not in {'role', 'content'}},
            ))
    return turns



def load_longmemeval(path: str | Path) -> List[Example]:
    raw = load_json(path)
    examples: List[Example] = []
    for inst in raw:
        session_ids = [str(x) for x in inst['haystack_session_ids']]
        session_dates = [str(x) for x in inst['haystack_dates']]
        turns = _flatten_longmemeval_sessions(inst['haystack_sessions'], session_ids, session_dates)
        evidence_turns = [t.turn_id for t in turns if t.metadata.get('has_answer')]
        evidence_sessions = [str(x) for x in inst.get('answer_session_ids', [])]
        examples.append(Example(
            example_id=str(inst['question_id']),
            dataset='longmemeval',
            question=normalize_text(inst['question']),
            answer=normalize_text(inst['answer']),
            question_type=inst.get('question_type', 'qa'),
            turns=turns,
            question_timestamp=str(inst.get('question_date', session_dates[-1] if session_dates else '')),
            evidence_turn_ids=evidence_turns,
            evidence_session_ids=evidence_sessions,
            metadata={
                'question_id': inst['question_id'],
                'question_type': inst.get('question_type', ''),
                'is_abstention': str(inst['question_id']).endswith('_abs'),
            },
        ))
    return examples



def load_synthetic(path: str | Path) -> List[Example]:
    raw = load_json(path)
    examples: List[Example] = []
    for inst in raw:
        turns = [Turn(**turn) for turn in inst['turns']]
        examples.append(Example(
            example_id=inst['example_id'],
            dataset=inst.get('dataset', 'synthetic'),
            question=inst['question'],
            answer=inst['answer'],
            question_type=inst.get('question_type', 'qa'),
            turns=turns,
            question_timestamp=inst.get('question_timestamp'),
            evidence_turn_ids=inst.get('evidence_turn_ids', []),
            evidence_session_ids=inst.get('evidence_session_ids', []),
            metadata=inst.get('metadata', {}),
        ))
    return examples
