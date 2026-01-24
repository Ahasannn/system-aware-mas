import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _select_device(device: str) -> str:
    device = (device or "auto").strip().lower()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _load_role_profiles(domain: str) -> List[Tuple[str, str]]:
    roles_dir = Path(__file__).resolve().parents[1] / "MAR" / "Roles" / domain
    if not roles_dir.is_dir():
        raise RuntimeError(f"Role domain not found: {roles_dir}")

    roles: List[Tuple[str, str]] = []
    for path in sorted(roles_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            profile = json.load(f)
        role_name = profile.get("Name") or path.stem
        roles.append((str(role_name), json.dumps(profile)))
    if not roles:
        raise RuntimeError(f"No role profiles found under {roles_dir}")
    return roles


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute role profile embeddings into a CSV file.")
    parser.add_argument("--domain", type=str, default="Code", help="Role domain under MAR/Roles/.")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("Datasets", "embeddings", "role_embeddings.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for embedding model: auto/cpu/cuda/cuda:0/cuda:1.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists.")
    args = parser.parse_args()

    output_path = args.output
    if os.path.exists(output_path) and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {output_path} (use --overwrite)")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    roles = _load_role_profiles(args.domain)
    roles.sort(key=lambda item: item[0])

    device = _select_device(args.device)
    model = SentenceTransformer(args.model, device=device)

    fieldnames = ("role_name", "embedding", "dataset_name")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        batch_size = max(1, int(args.batch_size))
        dataset_name = f"roles_{args.domain.lower()}"
        total = len(roles)
        for start in range(0, total, batch_size):
            batch = roles[start : start + batch_size]
            names = [item[0] for item in batch]
            texts = [item[1] for item in batch]
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,
            )
            for name, emb in zip(names, embeddings):
                writer.writerow(
                    {
                        "role_name": name,
                        "embedding": json.dumps(emb.tolist()),
                        "dataset_name": dataset_name,
                    }
                )
            print(f"wrote {min(start + batch_size, total)}/{total}", flush=True)

    print(f"done: {output_path}")


if __name__ == "__main__":
    main()

