"""Các hàm và lớp hỗ trợ xử lý hình học ảnh và bounding box.
Bao gồm:
- LetterboxTransform: Chuẩn hóa tỉ lệ ảnh và chuyển đổi tọa độ bounding box 2 chiều.
- compute_iou: Tính tỉ lệ Intersection over Union giữa hai hộp.
- compute_containment: Tính tỉ lệ hộp con nằm lọt trong hộp mẹ.
"""

from typing import Tuple, Optional, Union
import numpy as np
import cv2


class LetterboxTransform:
    """
    Biến đổi ảnh bằng cách giữ nguyên tỉ lệ khung hình (aspect ratio),
    thêm padding (viền đen/xám) đều hai bên hoặc trên dưới, và ánh xạ
    tọa độ bounding box ngược lại ảnh gốc một cách chuẩn xác.
    """

    @staticmethod
    def apply(
        image: np.ndarray,
        target_shape: Tuple[int, int] = (640, 640),
        fill_color: int = 114,
        stride: int = 32,
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Thực hiện letterbox trên ảnh.

        Args:
            image: Ảnh đầu vào BGR (H, W, C).
            target_shape: (target_h, target_w) mong muốn.
            fill_color: Giá trị pixel dùng để padding (mặc định 114 như Ultralytics).
            stride: Bội số kích thước (thường là 32 cho YOLO).

        Returns:
            padded_image: Ảnh đã letterbox.
            scale: Tỉ lệ co dãn (new / old).
            (pad_w, pad_h): Số pixel đệm ở mỗi phía (trái/phải, trên/dưới).
        """
        shape = image.shape[:2]  # (h, w)
        target_h, target_w = target_shape

        # Tính scale ratio
        r = min(target_h / shape[0], target_w / shape[1])

        # Kích thước ảnh mới sau khi scale nhưng chưa pad
        new_unpad_w = int(round(shape[1] * r))
        new_unpad_h = int(round(shape[0] * r))

        # Tính khoảng đệm còn thiếu
        dw = target_w - new_unpad_w
        dh = target_h - new_unpad_h

        # Chia đều 2 bên
        dw /= 2.0
        dh /= 2.0

        # Resize ảnh nếu cần
        if (shape[1], shape[0]) != (new_unpad_w, new_unpad_h):
            resized = cv2.resize(image, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = image

        # Tính padding nguyên
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(fill_color, fill_color, fill_color)
        )

        return padded, r, (left, top)

    @staticmethod
    def inverse_bbox(
        bbox: Tuple[float, float, float, float],
        scale: float,
        pad: Tuple[int, int],
        orig_shape: Optional[Tuple[int, int]] = None
    ) -> Tuple[int, int, int, int]:
        """
        Ánh xạ tọa độ box từ không gian letterbox ngược về không gian ảnh gốc.

        Args:
            bbox: (x1, y1, x2, y2) trong ảnh đã letterbox.
            scale: Tỉ lệ scale đã dùng trong apply.
            pad: (pad_w, pad_h) độ lệch đệm ở góc trái/trên.
            orig_shape: (orig_h, orig_w) để cắt giới hạn box không vượt ra ngoài ảnh.

        Returns:
            (x1, y1, x2, y2) tọa độ nguyên trên ảnh gốc.
        """
        pad_w, pad_h = pad
        x1 = (bbox[0] - pad_w) / scale
        y1 = (bbox[1] - pad_h) / scale
        x2 = (bbox[2] - pad_w) / scale
        y2 = (bbox[3] - pad_h) / scale

        if orig_shape is not None:
            orig_h, orig_w = orig_shape[:2]
            x1 = max(0, min(orig_w - 1, x1))
            y1 = max(0, min(orig_h - 1, y1))
            x2 = max(0, min(orig_w, x2))
            y2 = max(0, min(orig_h, y2))

        return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def scale_bbox(
    bbox: Tuple[Union[int, float], Union[int, float], Union[int, float], Union[int, float]],
    scale: float,
    pad_x: int,
    pad_y: int
) -> Tuple[int, int, int, int]:
    """
    Chuyển đổi tọa độ bounding box từ ảnh gốc sang tọa độ trên canvas letterbox.
    Returns (x1, y1, x2, y2) dạng int.
    """
    x1, y1, x2, y2 = bbox
    return (
        int(round(x1 * scale + pad_x)),
        int(round(y1 * scale + pad_y)),
        int(round(x2 * scale + pad_x)),
        int(round(y2 * scale + pad_y)),
    )


def compute_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float]
) -> float:
    """
    Tính Intersection over Union (IoU) giữa 2 bounding box dạng (x1, y1, x2, y2).
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    if inter_area <= 0.0:
        return 0.0

    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0

    return float(inter_area / union_area)


def compute_containment(
    inner_box: Tuple[float, float, float, float],
    outer_box: Tuple[float, float, float, float]
) -> float:
    """
    Tính tỉ lệ diện tích phần giao nhau so với diện tích hộp inner_box.
    Giá trị = 1.0 nghĩa là inner_box nằm lọt hoàn toàn trong outer_box.
    """
    x1 = max(inner_box[0], outer_box[0])
    y1 = max(inner_box[1], outer_box[1])
    x2 = min(inner_box[2], outer_box[2])
    y2 = min(inner_box[3], outer_box[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    inner_area = max(0.0, (inner_box[2] - inner_box[0]) * (inner_box[3] - inner_box[1]))
    if inner_area <= 0.0:
        return 0.0

    return float(inter_area / inner_area)


def is_reflective_traffic_sign(
    gray_img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    sat_thresh: int = 235,
    sat_ratio_min: float = 0.20,
    solidity_min: float = 0.88
) -> bool:
    """
    Kiểm tra vùng phát hiện có phải biển báo giao thông phản quang hoặc chao đèn hình học cứng hay không.
    Nguyên lý:
    - Biển phản quang / chao đèn có lõi bão hòa sáng rất cao (pixel >= sat_thresh).
    - Có đường bao hình học phẳng lồi (Solidity >= solidity_min).
    - Đa giác xấp xỉ có 3 đến 5 đỉnh (tam giác, hình chữ nhật, hình thang, ngũ giác).
    - Ngược lại, ngọn lửa thật có viền răng cưa nham nhở (solidity thấp < 0.75 hoặc đa giác > 6-8 đỉnh).
    """
    if gray_img is None or gray_img.size == 0:
        return False
    x1, y1, x2, y2 = bbox
    h_img, w_img = gray_img.shape[:2]
    cx1, cy1 = max(0, min(w_img - 1, x1)), max(0, min(h_img - 1, y1))
    cx2, cy2 = max(0, min(w_img, x2)), max(0, min(h_img, y2))
    if cx2 <= cx1 or cy2 <= cy1:
        return False

    crop = gray_img[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return False

    # Lõi sáng bão hòa cao (>= 235)
    sat_ratio = float(np.sum(crop >= sat_thresh) / crop.size)
    if sat_ratio < sat_ratio_min:
        return False

    # Phân tích hình học đường bao (contour)
    _, thresh = cv2.threshold(crop, 215, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)

    frame_area = max(1, h_img * w_img)
    box_w = cx2 - cx1
    box_h = cy2 - cy1
    box_area = box_w * box_h
    # Giới hạn kích thước theo tỷ lệ khung hình (chuẩn hóa đa độ phân giải từ 360p đến 4K):
    # Biển báo / chao đèn chiếm tối đa 3.5% diện tích khung hình
    if area < 30 or (area / frame_area) > 0.035:
        return False
    if (box_area / frame_area) > 0.05:
        return False

    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = (area / hull_area) if hull_area > 0 else 0.0

    peri = cv2.arcLength(c, True)
    hull_peri = cv2.arcLength(hull, True)
    roughness = (peri / hull_peri) if hull_peri > 0 else 1.0
    approx = cv2.approxPolyDP(c, 0.035 * peri, True)

    # Biển báo tam giác (3 đỉnh), hình chữ nhật/máng đèn trần (4-5 đỉnh)
    # có độ lồi cao (>= solidity_min) và quầng viền phẳng hình học (roughness < 1.15)
    if solidity >= solidity_min and len(approx) in (3, 4, 5) and roughness < 1.15:
        return True
    return False
