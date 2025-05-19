import argparse
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


def load_log(path: str) -> List[Tuple[int, int]]:
    """Load training log as (step, map_id) tuples."""
    entries: List[Tuple[int, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            step_str, map_id_str = line.strip().split("\t")
            entries.append((int(step_str), int(map_id_str)))
    return entries


def load_coords(path: str) -> Dict[int, Tuple[float, float]]:
    """Load map coordinates from JSON mapping of map ID to [x, y]."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): tuple(v) for k, v in data.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize training progress on Pokemon map")
    parser.add_argument("--log-path", required=True, help="Path to map ID log produced during training")
    parser.add_argument("--coords", required=True, help="JSON file mapping map IDs to coordinates")
    parser.add_argument("--output", default="training_map.png", help="Output image path")
    args = parser.parse_args()

    entries = load_log(args.log_path)
    coords = load_coords(args.coords)

    xs = []
    ys = []
    colors = []
    for step, map_id in entries:
        if map_id in coords:
            x, y = coords[map_id]
            xs.append(x)
            ys.append(y)
            colors.append(step)

    if not xs:
        raise SystemExit("No coordinates found for logged map IDs")

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(xs, ys, c=colors, cmap="viridis", s=10, alpha=0.7)
    plt.colorbar(scatter, label="Training step")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Training Progress by Map")
    plt.tight_layout()
    plt.savefig(args.output)
    plt.show(block=False)


if __name__ == "__main__":
    main()
