import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

def generate_splits(data_path, save_path, name="mnist_splits_hyperparam_tuning.json"):    
    data_path = Path(data_path)
    out_file = Path(save_path) / name

    # 1) Collect all paths + labels
    data = defaultdict(lambda: defaultdict(list))
    for p in data_path.glob("mnist_png_*/**/*.pth"):
        split = "train" if "train" in p.as_posix() else "test"
        label = p.parent.parent.stem.split("_")[-2]
        data[split]["path"].append(p.as_posix())
        data[split]["label"].append(label)

    # 2) Stratified sample 20% of original train
    train_paths = data["train"]["path"]
    train_labels = data["train"]["label"]
    idx = list(range(len(train_paths)))
    train_idx, rem_idx = train_test_split(
        idx, train_size=0.20, stratify=train_labels, random_state=42, shuffle=True
    )
    train_paths_final = [train_paths[i] for i in train_idx]
    train_labels_final = [train_labels[i] for i in train_idx]

    # 3) From the remaining 80%, stratified sample 5% of original -> 6.25% of rem
    rem_paths = [train_paths[i] for i in rem_idx]
    rem_labels = [train_labels[i] for i in rem_idx]
    val_frac = 0.05 / 0.80  # =0.0625
    val_idx_in_rem, _ = train_test_split(
        list(range(len(rem_paths))), train_size=val_frac,
        stratify=rem_labels, random_state=42, shuffle=True
    )
    val_paths_final = [rem_paths[i] for i in val_idx_in_rem]
    val_labels_final = [rem_labels[i] for i in val_idx_in_rem]

    # 4) Stratified sample 10% of original test
    test_paths = data["test"]["path"]
    test_labels = data["test"]["label"]
    idx_test = list(range(len(test_paths)))
    test_idx_final, _ = train_test_split(
        idx_test, train_size=0.10, stratify=test_labels, random_state=42, shuffle=True
    )
    test_paths_final = [test_paths[i] for i in test_idx_final]
    test_labels_final = [test_labels[i] for i in test_idx_final]

    # 5) Assemble output
    out = {
        "train": {"path": train_paths_final, "label": train_labels_final},
        "val":   {"path": val_paths_final,   "label": val_labels_final},
        "test":  {"path": test_paths_final,  "label": test_labels_final},
    }

    # Save JSON
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    parser = ArgumentParser("MNIST - generate balanced data splits")
    parser.add_argument('--data_path', type=str, default='data/mnist-inrs')
    parser.add_argument('--save_path', type=str, default='data/mnist-inrs')
    parser.add_argument(
        "--name", type=str, default="mnist_splits_hyperparam_tuning.json", help="output json file name"
    )
    args = parser.parse_args()

    generate_splits(
        data_path=args.data_path,
        save_path=args.save_path,
        name=args.name,
    )  
