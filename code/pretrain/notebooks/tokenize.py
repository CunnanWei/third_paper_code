import argparse
import os
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


def resolve_path(raw_path, datasets_root):
    path_obj = Path(raw_path)
    if path_obj.is_absolute():
        return path_obj
    if raw_path.startswith("datasets/"):
        return Path(datasets_root) / Path(raw_path).relative_to("datasets")
    return Path(datasets_root) / path_obj


def build_input_ids_path(ecg_abs_path, ids_dir_name):
    ecg_dir = ecg_abs_path.parent
    target_dir = ecg_dir.parent / ids_dir_name
    target_dir.mkdir(parents=True, exist_ok=True)
    return (target_dir / ecg_abs_path.name).with_suffix(".npz")


def format_csv_path(ids_abs_path, sample_path_str, datasets_root):
    if sample_path_str.startswith("datasets/"):
        rel = ids_abs_path.relative_to(datasets_root)
        return str(PurePosixPath("datasets") / PurePosixPath(rel.as_posix()))
    if os.path.isabs(sample_path_str):
        return ids_abs_path.as_posix()
    rel = ids_abs_path.relative_to(datasets_root)
    return rel.as_posix()


def tokenize_csv(
    csv_path,
    datasets_root,
    text_model,
    text_column,
    path_column,
    output_column,
    ids_dir_name,
    max_length,
    overwrite,
):
    csv_path = Path(csv_path)
    datasets_root = Path(datasets_root)
    df = pd.read_csv(csv_path)
    tokenizer = AutoTokenizer.from_pretrained(text_model)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing"):
        raw_ecg_path = row[path_column]
        text = row[text_column] if pd.notna(row[text_column]) else ""
        ecg_abs_path = resolve_path(raw_ecg_path, datasets_root)
        ids_abs_path = build_input_ids_path(ecg_abs_path, ids_dir_name)

        if not overwrite and ids_abs_path.exists():
            df.at[idx, output_column] = format_csv_path(ids_abs_path, raw_ecg_path, datasets_root)
            continue

        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded.input_ids.squeeze(0).numpy()
        np.savez_compressed(ids_abs_path, input_ids=input_ids)
        df.at[idx, output_column] = format_csv_path(ids_abs_path, raw_ecg_path, datasets_root)

    df.to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="预处理 ECG 报告 input_ids")
    parser.add_argument("--csv", required=True, help="train/val CSV 路径")
    parser.add_argument("--datasets-root", required=True, help="datasets 根目录")
    parser.add_argument("--text-model", default="ncbi/MedCPT-Query-Encoder")
    parser.add_argument("--text-column", default="total_report")
    parser.add_argument("--path-column", default="path")
    parser.add_argument("--output-column", default="input_ids_path")
    parser.add_argument("--ids-dir", default="input_ids", help="与 ecg_npy 同级的存储目录")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    tokenize_csv(
        csv_path=args.csv,
        datasets_root=args.datasets_root,
        text_model=args.text_model,
        text_column=args.text_column,
        path_column=args.path_column,
        output_column=args.output_column,
        ids_dir_name=args.ids_dir,
        max_length=args.max_length,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
