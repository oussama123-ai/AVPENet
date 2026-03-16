"""
Preprocess raw audio files into mel-spectrograms.

Usage:
    python scripts/preprocess_audio.py \
        --input_dir data/raw_audio/ \
        --output_dir data/audio/ \
        --n_workers 8
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from avpenet.data.audio_preprocessing import AudioPreprocessor

EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def process_file(args):
    audio_path, output_path, preprocessor = args
    try:
        mel = preprocessor(audio_path)          # (1, 128, 300)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(mel, output_path)
        return str(audio_path), None
    except Exception as e:
        return str(audio_path), str(e)


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio files")
    parser.add_argument("--input_dir",  type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_workers",  type=int, default=4)
    parser.add_argument("--overwrite",  action="store_true")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = [
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in EXTENSIONS
    ]
    print(f"Found {len(audio_files)} audio files")

    preprocessor = AudioPreprocessor()

    tasks = []
    for audio_path in audio_files:
        rel_path    = audio_path.relative_to(input_dir)
        output_path = output_dir / rel_path.with_suffix(".pt")
        if not args.overwrite and output_path.exists():
            continue
        tasks.append((audio_path, output_path, preprocessor))

    print(f"Processing {len(tasks)} files...")
    errors = []

    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        futures = {pool.submit(process_file, t): t[0] for t in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks)):
            path, err = future.result()
            if err:
                errors.append((path, err))

    print(f"\nDone. {len(tasks) - len(errors)} succeeded, {len(errors)} failed.")
    if errors:
        print("Errors:")
        for path, err in errors[:10]:
            print(f"  {path}: {err}")


if __name__ == "__main__":
    main()
