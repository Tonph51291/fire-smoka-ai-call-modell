
from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil

# Tạo thư mục lưu model
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

# Tải trọng số đã huấn luyện
downloaded_path = hf_hub_download(
    repo_id="rabahdev/fire-smoke-yolov8n",
    filename="best.pt"
)

# Sao chép model về thư mục dự án
target_path = model_dir / "fire_smoke_best.pt"
shutil.copy2(downloaded_path, target_path)

print(f"Da tai model: {target_path.resolve()}")
