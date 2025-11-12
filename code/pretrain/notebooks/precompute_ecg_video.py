from pathlib import Path

import numpy as np
import pandas as pd
from pyts.image import GramianAngularField, RecurrencePlot, MarkovTransitionField
from tqdm import tqdm


PROJECT_ROOT = Path.cwd().resolve()
while not (PROJECT_ROOT / "datasets").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

DATASET_DIR = PROJECT_ROOT / "datasets" / "pretrain"
TARGET_DIR = DATASET_DIR / "ecg_video"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

POINTS_PER_SEGMENT = 100
STRIDE = 100
SIGNAL_LENGTH = 5000
SEGMENTS_PER_LEAD = (SIGNAL_LENGTH - POINTS_PER_SEGMENT) // STRIDE + 1

segment_indices = (
    np.arange(SEGMENTS_PER_LEAD)[:, None] *
    STRIDE + np.arange(POINTS_PER_SEGMENT)
)

gadf = GramianAngularField(method="difference")
rp = RecurrencePlot(dimension=1, time_delay=1,
                    threshold="point", percentage=10)
mtf = MarkovTransitionField(n_bins=8, strategy="quantile")


def build_target_path(raw_path: str) -> Path:
    raw_stem = Path(raw_path).with_suffix("").name
    return TARGET_DIR / f"{raw_stem}.npy"


def transform_ecg(ecg: np.ndarray) -> np.ndarray:
    segments = ecg[:, segment_indices]
    batch = segments.reshape(-1, POINTS_PER_SEGMENT)

    gadf_images = gadf.fit_transform(batch).reshape(
        12, SEGMENTS_PER_LEAD, POINTS_PER_SEGMENT, POINTS_PER_SEGMENT
    )
    rp_images = rp.fit_transform(batch).reshape(
        12, SEGMENTS_PER_LEAD, POINTS_PER_SEGMENT, POINTS_PER_SEGMENT
    )
    mtf_images = mtf.fit_transform(batch).reshape(
        12, SEGMENTS_PER_LEAD, POINTS_PER_SEGMENT, POINTS_PER_SEGMENT
    )

    stacked = np.stack([gadf_images, rp_images, mtf_images], axis=2).astype(
        np.float32, copy=False
    )
    transposed = np.transpose(stacked, (1, 0, 2, 3, 4))
    merged = transposed.reshape(
        SEGMENTS_PER_LEAD, 12 * 3, POINTS_PER_SEGMENT, POINTS_PER_SEGMENT
    ).astype(np.float32, copy=False)
    return merged


def process_split(csv_path: Path, split_name: str) -> None:
    df = pd.read_csv(csv_path)
    video_paths = []
    for raw_path in tqdm(df["path"].tolist(), desc=f"处理 {split_name}"):
        input_path = PROJECT_ROOT / raw_path
        target_path = build_target_path(raw_path)
        if not target_path.exists():
            ecg = np.load(input_path)
            processed = transform_ecg(ecg)
            np.save(target_path, processed)
        video_paths.append(str(target_path.relative_to(PROJECT_ROOT)))

    df["video_path"] = video_paths
    df.to_csv(csv_path, index=False)


def main() -> None:
    for split in ("train", "val"):
        process_split(DATASET_DIR / f"{split}.csv", split)
    print("预处理完成")


if __name__ == "__main__":
    main()
