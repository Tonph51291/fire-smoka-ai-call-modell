
from pathlib import Path
from contextlib import redirect_stdout
from ultralytics import YOLO
import json
import traceback

BASE_DIR = Path(__file__).resolve().parent
REPORT = BASE_DIR / "model_report.txt"
MODEL_PATHS = [
    BASE_DIR / "models" / "fire_yolo.pt",
    BASE_DIR / "models" / "best.pt",
]

def inspect_model(path):
    print(f"\n{'=' * 65}\nMODEL: {path.name}\n{'=' * 65}")

    if not path.exists():
        print(f"ERROR: Không tìm thấy file {path}")
        return

    model = YOLO(str(path))
    net = model.model
    ckpt = getattr(model, "ckpt", None) or {}

    print("\n[THONG TIN CHUNG]")
    print("Task:", model.task)
    print("Classes:", json.dumps(model.names, ensure_ascii=False))
    print("Ultralytics version:", ckpt.get("version"))
    print("Checkpoint keys:", list(ckpt.keys()))

    print("\n[CAU HINH KIEN TRUC]")
    print(json.dumps(
        getattr(net, "yaml", {}),
        ensure_ascii=False,
        indent=2,
        default=str
    ))

    print("\n[THONG TIN HUAN LUYEN]")
    for key in ("epoch", "best_fitness", "train_args", "train_metrics", "date"):
        if ckpt.get(key) is not None:
            print(f"{key}: {ckpt[key]}")

    print("\n[THONG KE LAYER]")
    total_params = 0

    for i, layer in enumerate(net.model):
        params = sum(p.numel() for p in layer.parameters())
        total_params += params
        print(
            f"Layer {i}: {layer.__class__.__name__}, "
            f"params={params:,}, from={getattr(layer, 'f', None)}"
        )

    print(f"\nTong parameters: {total_params:,}")

    print("\n[MODEL SUMMARY]")
    net.info(verbose=True)

def main():
    print("Dang kiem tra 2 model...")
    errors = []

    with REPORT.open("w", encoding="utf-8") as file:
        with redirect_stdout(file):
            print("BAO CAO KIEM TRA MODEL FIRE & SMOKE AI")
            print("Report file:", REPORT)
            print("Luu y: checkpoint co the khong luu day du lich su huan luyen.")

            for path in MODEL_PATHS:
                try:
                    inspect_model(path)
                except Exception:
                    errors.append(path.name)
                    print(f"\nLOI KHI KIEM TRA {path.name}")
                    traceback.print_exc()

    print(f"Da xuat bao cao: {REPORT}")
    print("Mo file model_report.txt de xem ket qua.")

    if errors:
        print("Co model gap loi:", ", ".join(errors))
        print("Xem chi tiet trong file bao cao.")

if __name__ == "__main__":
    main()
