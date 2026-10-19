"""Script tải trọng số model từ Hugging Face Hub."""

import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files


def download_fire_yolo_model(
    repo_id: str = "seawsurf/fire_yolo",
    output_filename: str = "fire_yolo.pt"
) -> Path:
    """
    Tự động tìm và tải file .pt từ Hugging Face repo.
    Ưu tiên file best.pt. Lưu vào thư mục models/.
    """
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    target_path = model_dir / output_filename

    print(f"[*] Đang kiểm tra repo: {repo_id}")
    files = list_repo_files(repo_id)
    pt_files = [f for f in files if f.endswith(".pt")]
    print(f"[*] Tìm thấy các file .pt: {pt_files}")

    # Ưu tiên best.pt nếu có
    selected_file = None
    for candidate in ["best.pt", "weights/best.pt", "fire_yolo.pt"]:
        if candidate in pt_files:
            selected_file = candidate
            break

    if not selected_file and pt_files:
        selected_file = pt_files[0]

    if not selected_file:
        raise FileNotFoundError(f"Không tìm thấy file .pt nào trong repo {repo_id}")

    print(f"[*] Đang tải: {selected_file} từ {repo_id}...")
    downloaded_path = hf_hub_download(repo_id=repo_id, filename=selected_file)
    shutil.copy2(downloaded_path, target_path)
    print(f"[+] Đã tải và lưu thành công model: {target_path.resolve()} ({target_path.stat().st_size:,} bytes)")
    return target_path


if __name__ == "__main__":
    download_fire_yolo_model()
