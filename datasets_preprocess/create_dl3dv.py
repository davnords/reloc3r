import os
import random
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Generate train/test splits for DL3DV dataset.")
    parser.add_argument("--dl3dv_dir", type=str, required=True,
                        help="Root directory of the DL3DV dataset (e.g., /data/dataset/DL3DV-10K/)")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Fraction of sequences to use for testing (default: 0.1 = 10%)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    random.seed(args.seed)

    all_sequences = []
    for scene_id in os.listdir(args.dl3dv_dir):
        scene_path = os.path.join(args.dl3dv_dir, scene_id)
        if not os.path.isdir(scene_path):
            continue
        for seq_name in os.listdir(scene_path):
            seq_path = os.path.join(scene_path, seq_name)
            # Only include valid sequences (those that have transforms.json)
            if os.path.isdir(seq_path) and os.path.exists(os.path.join(seq_path, "transforms.json")):
                rel_path = f"{scene_id}/{seq_name}"
                all_sequences.append(rel_path)

    if not all_sequences:
        print("❌ No valid sequences found (missing transforms.json). Check dataset path.")
        return

    print(f"✅ Found {len(all_sequences)} total sequences.")

    # Shuffle and split
    random.shuffle(all_sequences)
    split_idx = int(len(all_sequences) * (1 - args.test_ratio))
    train_sequences = all_sequences[:split_idx]
    test_sequences = all_sequences[split_idx:]

    # Write splits
    train_path = os.path.join(args.dl3dv_dir, "train.txt")
    test_path = os.path.join(args.dl3dv_dir, "test.txt")

    with open(train_path, "w") as f:
        f.write("\n".join(train_sequences))
    with open(test_path, "w") as f:
        f.write("\n".join(test_sequences))

    print(f"📝 Wrote {len(train_sequences)} train sequences to {train_path}")
    print(f"🧪 Wrote {len(test_sequences)} test sequences to {test_path}")

if __name__ == "__main__":
    main()
