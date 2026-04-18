#!/usr/bin/env python3
"""
Download models and datasets on the login node (has internet).
Run this BEFORE submitting GPU jobs.

Usage:
    module load miniconda3 && conda activate dac_mem
    python scripts/download.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))


def download_models(cache_dir: str):
    """Pre-download HuggingFace models for offline use."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sentence_transformers import SentenceTransformer

    models = [
        'Qwen/Qwen2.5-7B-Instruct',
        'meta-llama/Llama-3.1-8B-Instruct',
        'mistralai/Mistral-7B-Instruct-v0.3',
    ]

    print('=== Downloading embedding model ===')
    SentenceTransformer('BAAI/bge-large-en-v1.5', cache_folder=cache_dir)
    print('  bge-large-en-v1.5 done')

    for model_name in models:
        print(f'\n=== Downloading {model_name} ===')
        try:
            AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir,
                                          trust_remote_code=True)
            print(f'  tokenizer done')
            AutoModelForCausalLM.from_pretrained(
                model_name, cache_dir=cache_dir, trust_remote_code=True,
                torch_dtype='auto')
            print(f'  model done')
        except Exception as e:
            print(f'  FAILED: {e}')
            print(f'  (You may need to accept the license on huggingface.co)')


def download_datasets(data_dir: str):
    """Pre-download benchmark datasets."""
    from dac_mem.datasets import maybe_download_dataset

    datasets = [
        ('locomo', f'{data_dir}/locomo/locomo10.json'),
        ('longmemeval', f'{data_dir}/longmemeval/longmemeval_s_cleaned.json'),
    ]

    for name, path in datasets:
        print(f'\n=== Downloading {name} ===')
        try:
            maybe_download_dataset(name, path)
            print(f'  saved to {path}')
        except Exception as e:
            print(f'  FAILED: {e}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache_dir', default='~/.cache/huggingface/hub',
                   help='HuggingFace cache directory')
    p.add_argument('--data_dir', default='data',
                   help='Dataset directory')
    p.add_argument('--models_only', action='store_true')
    p.add_argument('--data_only', action='store_true')
    args = p.parse_args()

    cache_dir = str(Path(args.cache_dir).expanduser())

    if not args.models_only:
        download_datasets(args.data_dir)

    if not args.data_only:
        download_models(cache_dir)

    print('\n=== All downloads complete ===')
    print(f'Models cached in: {cache_dir}')
    print(f'Datasets in: {args.data_dir}/')
    print('\nYou can now submit GPU jobs with:')
    print('  sbatch scripts/submit_job.sh')


if __name__ == '__main__':
    main()
