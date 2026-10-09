import argparse
import os
from .drift_detector import DriftDetector
from .label_studio_exporter import LabelStudioExporter
from .lexicon_miner import LexiconMiner
from training.train import train  # Import training function


def run_pipeline(stage="detect"):
    """
    Stages:
    - detect: Find uncertain samples and export for labeling.
    - retrain: (Assumes labeling done) Merge data and retrain.
    """

    if stage == "detect":
        print("--- Stage 1: Drift Detection & Export ---")
        detector = DriftDetector()
        samples = detector.get_uncertain_samples(limit=200)

        if not samples:
            print("No uncertain samples found.")
            return

        # Mine Lexicon
        print("--- Lexicon Mining ---")
        miner = LexiconMiner()
        texts = [s["text"] for s in samples]
        new_terms = miner.mine_unknowns(texts)
        print("Potential new terms:", new_terms)

        # Export
        print("--- Exporting to Label Studio ---")
        exporter = LabelStudioExporter()
        file_path = exporter.export(samples)
        print(f"Please upload {file_path} to Label Studio.")

    elif stage == "retrain":
        print("--- Stage 2: Retraining ---")
        # Placeholder for loading new labeled data
        # new_data_path = "data/labeled_new.csv"
        # current_version = "v1.0.0"
        # next_version = "v1.1.0"

        print("Merging new data...")
        # merge_datasets(original, new_data)

        print("Triggering training...")
        # train(data_path=merged_path, output_dir=f"models/sauti-sense-afriberta-{next_version}")

        print("Retraining complete (Mock).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["detect", "retrain"], default="detect")
    args = parser.parse_args()

    run_pipeline(args.stage)
