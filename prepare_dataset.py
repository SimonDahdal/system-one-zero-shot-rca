"""Create a reusable dataset and train/test split without running any model."""
import argparse
from pathlib import Path
from rca_experiments import config
from rca_experiments.dataset import prepare_dataset, save_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--output", type=Path, default=config.PREPARED_DATA_PATH)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    dataset = prepare_dataset(args.data_dir, download=not args.skip_download)
    save_dataset(dataset, args.output)
    print(f"Saved {args.output}: {len(dataset.train_idx)} train, {len(dataset.test_idx)} test")


if __name__ == "__main__":
    main()
