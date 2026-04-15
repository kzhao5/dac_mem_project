"""
Dataset loaders for DAC-Mem evaluation.

Supported benchmarks:
  - LoCoMo        — multi-session conversational memory (必选)
  - LongMemEval   — 5 long-term memory abilities (必选)
  - PERMA         — event-driven preference personalisation
  - MemoryBench   — memory + continual learning
  - Synthetic     — tiny examples for smoke testing
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import requests

from .schema import Example, Turn
from .utils import load_json, normalize_text

# ── download URLs ──────────────────────────────────────────────────────────

LOCOMO_URL = ('https://raw.githubusercontent.com/snap-research/locomo/'
              'main/data/locomo10.json')

LONGMEMEVAL_URLS = {
    's_cleaned': ('https://huggingface.co/datasets/xiaowu0162/'
                  'longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json'),
    'm_cleaned': ('https://huggingface.co/datasets/xiaowu0162/'
                  'longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json'),
    'oracle':    ('https://huggingface.co/datasets/xiaowu0162/'
                  'longmemeval-cleaned/resolve/main/longmemeval_oracle.json'),
}

PERMA_URL = ('https://huggingface.co/datasets/xu1998hz/'
             'PERMA/resolve/main/perma.json')

MEMORYBENCH_URL = ('https://huggingface.co/datasets/THU-KEG/'
                   'MemoryBench/resolve/main/memorybench.json')


def maybe_download(url: str, out_path: str, timeout: int = 180) -> Path:
    out = Path(out_path)
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    out.write_bytes(r.content)
    return out


def maybe_download_dataset(dataset: str, out_path: str,
                           variant: str = 's_cleaned') -> Path:
    if dataset == 'locomo':
        return maybe_download(LOCOMO_URL, out_path)
    if dataset == 'longmemeval':
        return maybe_download(LONGMEMEVAL_URLS[variant], out_path)
    if dataset == 'perma':
        return maybe_download(PERMA_URL, out_path)
    if dataset == 'memorybench':
        return maybe_download(MEMORYBENCH_URL, out_path)
    raise ValueError(f'Unsupported dataset: {dataset}')


# ═══════════════════════════════════════════════════════════════════════════
# LoCoMo
# ═══════════════════════════════════════════════════════════════════════════

def _flatten_locomo_conversation(conv: Dict) -> List[Turn]:
    turns: List[Turn] = []
    speaker_a = conv.get('speaker_a', 'speaker_a')
    speaker_b = conv.get('speaker_b', 'speaker_b')
    sess_keys = sorted(
        [k for k in conv if k.startswith('session_')
         and not k.endswith('_date_time')],
        key=lambda x: int(x.split('_')[1]))
    for sk in sess_keys:
        ts = conv.get(f'{sk}_date_time', '')
        for t in conv.get(sk, []):
            sp = t.get('speaker', '')
            if sp == speaker_a:
                sp = 'speaker_a'
            elif sp == speaker_b:
                sp = 'speaker_b'
            turns.append(Turn(
                turn_id=str(t.get('dia_id', f'{sk}_{len(turns)}')),
                session_id=sk, speaker=sp,
                text=normalize_text(t.get('text', '')),
                timestamp=str(ts),
                metadata={k: v for k, v in t.items()
                          if k not in ('text', 'speaker', 'dia_id')},
            ))
    return turns


def load_locomo(path: str | Path) -> List[Example]:
    raw = load_json(path)
    examples: List[Example] = []
    for conv in raw:
        turns = _flatten_locomo_conversation(conv['conversation'])
        tsmap = {t.turn_id: t.session_id for t in turns}
        for idx, qa in enumerate(conv.get('qa', [])):
            ev_turns = [str(x) for x in qa.get('evidence', [])]
            ev_sessions = sorted({tsmap[e] for e in ev_turns if e in tsmap})
            examples.append(Example(
                example_id=f"{conv.get('sample_id', 'locomo')}_{idx}",
                dataset='locomo',
                question=normalize_text(qa['question']),
                answer=normalize_text(qa['answer']),
                question_type=qa.get('category', 'qa'),
                turns=turns,
                question_timestamp=turns[-1].timestamp if turns else None,
                evidence_turn_ids=ev_turns,
                evidence_session_ids=ev_sessions,
                metadata={'sample_id': conv.get('sample_id'),
                          'category': qa.get('category', '')},
            ))
    return examples


# ═══════════════════════════════════════════════════════════════════════════
# LongMemEval
# ═══════════════════════════════════════════════════════════════════════════

def _flatten_longmemeval_sessions(sessions, session_ids, session_dates):
    turns: List[Turn] = []
    for sid, ts, sess in zip(session_ids, session_dates, sessions):
        for idx, turn in enumerate(sess):
            tid = f'{sid}_{idx}'
            if 'has_answer' in turn:
                tid = f'{tid}_answer'
            turns.append(Turn(
                turn_id=tid, session_id=str(sid),
                speaker=turn.get('role', 'user'),
                text=normalize_text(turn.get('content', '')),
                timestamp=str(ts),
                metadata={k: v for k, v in turn.items()
                          if k not in ('role', 'content')},
            ))
    return turns


def load_longmemeval(path: str | Path) -> List[Example]:
    raw = load_json(path)
    examples: List[Example] = []
    for inst in raw:
        sids = [str(x) for x in inst['haystack_session_ids']]
        sdates = [str(x) for x in inst['haystack_dates']]
        turns = _flatten_longmemeval_sessions(
            inst['haystack_sessions'], sids, sdates)
        ev_turns = [t.turn_id for t in turns if t.metadata.get('has_answer')]
        ev_sessions = [str(x) for x in inst.get('answer_session_ids', [])]
        examples.append(Example(
            example_id=str(inst['question_id']),
            dataset='longmemeval',
            question=normalize_text(inst['question']),
            answer=normalize_text(inst['answer']),
            question_type=inst.get('question_type', 'qa'),
            turns=turns,
            question_timestamp=str(inst.get(
                'question_date', sdates[-1] if sdates else '')),
            evidence_turn_ids=ev_turns,
            evidence_session_ids=ev_sessions,
            metadata={
                'question_id': inst['question_id'],
                'question_type': inst.get('question_type', ''),
                'is_abstention': str(inst['question_id']).endswith('_abs'),
            },
        ))
    return examples


# ═══════════════════════════════════════════════════════════════════════════
# PERMA — event-driven preference personalisation
# ═══════════════════════════════════════════════════════════════════════════

def load_perma(path: str | Path) -> List[Example]:
    """Load PERMA benchmark.

    PERMA focuses on preference formation through events. Each instance has
    a dialogue history and questions about user preferences.
    """
    raw = load_json(path)
    examples: List[Example] = []
    for inst in raw:
        turns: List[Turn] = []
        # PERMA format: list of dialogues with turns
        for si, session in enumerate(inst.get('dialogues', inst.get('sessions', []))):
            sid = f'session_{si}'
            ts = session.get('date', session.get('timestamp', f'2024-01-{si+1:02d}'))
            for ti, turn in enumerate(session.get('turns', session.get('utterances', []))):
                text = turn.get('text', turn.get('content', turn.get('utterance', '')))
                speaker = turn.get('speaker', turn.get('role', 'user'))
                turns.append(Turn(
                    turn_id=f'{sid}_t{ti}', session_id=sid,
                    speaker=speaker, text=normalize_text(text),
                    timestamp=str(ts),
                    metadata={k: v for k, v in turn.items()
                              if k not in ('text', 'content', 'utterance',
                                           'speaker', 'role')},
                ))

        # questions
        for qi, qa in enumerate(inst.get('questions', inst.get('qa', []))):
            q = qa.get('question', qa.get('query', ''))
            a = qa.get('answer', qa.get('response', ''))
            ev_turns = [str(x) for x in qa.get('evidence_turns', [])]
            ev_sessions = [str(x) for x in qa.get('evidence_sessions', [])]
            examples.append(Example(
                example_id=f"perma_{inst.get('id', len(examples))}_{qi}",
                dataset='perma',
                question=normalize_text(q),
                answer=normalize_text(a),
                question_type=qa.get('type', 'preference'),
                turns=turns,
                evidence_turn_ids=ev_turns,
                evidence_session_ids=ev_sessions,
                metadata={'source': 'perma'},
            ))
    return examples


# ═══════════════════════════════════════════════════════════════════════════
# MemoryBench — memory + continual learning
# ═══════════════════════════════════════════════════════════════════════════

def load_memorybench(path: str | Path) -> List[Example]:
    """Load MemoryBench benchmark.

    MemoryBench emphasises memory plus continual learning from user feedback.
    """
    raw = load_json(path)
    examples: List[Example] = []
    for inst in raw:
        turns: List[Turn] = []
        for si, session in enumerate(inst.get('sessions', [])):
            sid = f'session_{si}'
            ts = session.get('timestamp', f'2024-01-{si+1:02d}')
            for ti, turn in enumerate(session.get('turns', [])):
                text = turn.get('content', turn.get('text', ''))
                speaker = turn.get('role', turn.get('speaker', 'user'))
                turns.append(Turn(
                    turn_id=f'{sid}_t{ti}', session_id=sid,
                    speaker=speaker, text=normalize_text(text),
                    timestamp=str(ts),
                ))
        q = inst.get('question', inst.get('query', ''))
        a = inst.get('answer', inst.get('response', ''))
        ev = inst.get('evidence_sessions', [])
        examples.append(Example(
            example_id=f"memorybench_{inst.get('id', len(examples))}",
            dataset='memorybench',
            question=normalize_text(q),
            answer=normalize_text(a),
            question_type=inst.get('type', 'qa'),
            turns=turns,
            evidence_session_ids=[str(x) for x in ev],
            metadata={'source': 'memorybench'},
        ))
    return examples


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic (smoke test)
# ═══════════════════════════════════════════════════════════════════════════

def load_synthetic(path: str | Path) -> List[Example]:
    raw = load_json(path)
    examples: List[Example] = []
    for inst in raw:
        turns = [Turn(**t) for t in inst['turns']]
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


# ═══════════════════════════════════════════════════════════════════════════
# Unified loader
# ═══════════════════════════════════════════════════════════════════════════

def load_dataset(name: str, path: str | Path) -> List[Example]:
    loaders = {
        'locomo': load_locomo,
        'longmemeval': load_longmemeval,
        'perma': load_perma,
        'memorybench': load_memorybench,
        'synthetic': load_synthetic,
    }
    loader = loaders.get(name)
    if loader is None:
        raise ValueError(f'Unknown dataset: {name}. '
                         f'Available: {list(loaders.keys())}')
    return loader(path)
