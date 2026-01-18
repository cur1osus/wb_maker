from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

ASSETS_DIR: Final[Path] = Path(__file__).resolve().parent / "assets"
ON_REVIEW_TEMPLATE_NAME: Final[str] = "on_check.jpg"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        try:
            return int(float(raw.replace(",", ".")))
        except ValueError:
            return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw.replace(",", "."))
    except ValueError:
        return default


# Параметры обработки «НА ПРОВЕРКЕ».
ON_REVIEW_THRESHOLD: Final[int] = 150  # Базовый порог бинаризации для поиска текста
ON_REVIEW_THRESHOLDS: Final[list[int]] = [150, 140, 170, 190, 120, 200]
ON_REVIEW_PADDING_X: Final[int] = _env_int("ON_REVIEW_PADDING_X", 3)
ON_REVIEW_PADDING_TOP: Final[int] = _env_int("ON_REVIEW_PADDING_TOP", 6)
ON_REVIEW_PADDING_BOTTOM: Final[int] = _env_int("ON_REVIEW_PADDING_BOTTOM", 18)
ON_REVIEW_ROI_WIDTH_MULTIPLIER: Final[float] = _env_float(
    "ON_REVIEW_ROI_WIDTH_MULTIPLIER", 1.6
)
ON_REVIEW_ROI_EXTRA_WIDTH: Final[int] = _env_int("ON_REVIEW_ROI_EXTRA_WIDTH", 600)
ON_REVIEW_ROI_BELOW_MULTIPLIER: Final[int] = _env_int(
    "ON_REVIEW_ROI_BELOW_MULTIPLIER", 2
)
ON_REVIEW_ROI_BELOW_MIN: Final[int] = _env_int("ON_REVIEW_ROI_BELOW_MIN", 65)
ON_REVIEW_LEFT_CLAMP_RATIO: Final[float] = _env_float("ON_REVIEW_LEFT_CLAMP_RATIO", 0.0)
ON_REVIEW_LEFT_CLAMP_MIN_PX: Final[int] = _env_int("ON_REVIEW_LEFT_CLAMP_MIN_PX", 0)
ON_REVIEW_GUARD_LEFT_RATIO: Final[float] = _env_float(
    "ON_REVIEW_GUARD_LEFT_RATIO", 0.14
)

# Ветка v1 — параметры из последнего коммита (оригинал без цветовой маски).
ON_REVIEW_THRESHOLD_V1: Final[int] = 233
ON_REVIEW_PADDING_X_V1: Final[int] = 37
ON_REVIEW_PADDING_TOP_V1: Final[int] = 16
ON_REVIEW_PADDING_BOTTOM_V1: Final[int] = 12
ON_REVIEW_ROI_WIDTH_MULTIPLIER_V1: Final[float] = 4.0
ON_REVIEW_ROI_EXTRA_WIDTH_V1: Final[int] = 239  # -1 — до правого края
ON_REVIEW_ROI_BELOW_MULTIPLIER_V1: Final[int] = 0
ON_REVIEW_ROI_BELOW_MIN_V1: Final[int] = 151

# Цветовое окно для поиска оранжевой плашки «НА ПРОВЕРКЕ».
ON_REVIEW_HSV_LOWER: Final[np.ndarray] = np.array([5, 140, 170], dtype=np.uint8)
ON_REVIEW_HSV_UPPER: Final[np.ndarray] = np.array([22, 255, 255], dtype=np.uint8)
ON_REVIEW_COLOR_MIN_AREA_RATIO: Final[float] = 0.0025  # от площади кадра
ON_REVIEW_COLOR_MIN_RATIO: Final[float] = 3.5  # ширина к высоте
ON_REVIEW_COLOR_MAX_Y_RATIO: Final[float] = 0.55  # плашка находится в верхней половине
ON_REVIEW_COLOR_PADDING: Final[int] = 6
ON_REVIEW_TEMPLATE_THRESHOLD: Final[float] = _env_float(
    "ON_REVIEW_TEMPLATE_THRESHOLD", 0.45
)
ON_REVIEW_TEMPLATE_MAX_Y_RATIO: Final[float] = 0.6
ON_REVIEW_TEMPLATE_MIN_WIDTH: Final[int] = 80
ON_REVIEW_TEMPLATE_MIN_HEIGHT: Final[int] = 20
ON_REVIEW_TEMPLATE_SCALE_MULTIPLIERS: Final[tuple[float, ...]] = (
    0.5,
    0.65,
    0.8,
    0.95,
    1.1,
    1.25,
    1.4,
)
ON_REVIEW_CANNY_LOW: Final[int] = 40
ON_REVIEW_CANNY_HIGH: Final[int] = 120


def _resolve_asset(name: str) -> Path:
    candidates = [
        ASSETS_DIR / name,
        Path.cwd() / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return ASSETS_DIR / name


@lru_cache(maxsize=1)
def _load_on_review_template() -> np.ndarray | None:
    template_path = _resolve_asset(ON_REVIEW_TEMPLATE_NAME)
    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if template is None:
        logger.error(
            "Не удалось загрузить шаблон '%s': %s",
            ON_REVIEW_TEMPLATE_NAME,
            template_path,
        )
        return None
    return template


def _candidate_template_scales(img_w: int, template_w: int) -> tuple[float, ...]:
    if template_w <= 0:
        return (1.0,)
    base = img_w / template_w
    scales = [base * multiplier for multiplier in ON_REVIEW_TEMPLATE_SCALE_MULTIPLIERS]
    filtered = [s for s in scales if 0.2 <= s <= 1.8]
    if not filtered:
        filtered = [min(1.0, base)]
    return tuple(sorted({round(s, 3) for s in filtered}))


def _find_on_review_box_by_color(img: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Пытается найти плашку по цвету (оранжевый фон), что устойчивее к OCR-шуму.
    Возвращает bbox (x, y, w, h) или None.
    """

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ON_REVIEW_HSV_LOWER, ON_REVIEW_HSV_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h_img, w_img = img.shape[:2]
    min_area = int(w_img * h_img * ON_REVIEW_COLOR_MIN_AREA_RATIO)

    best: tuple[int, int, int, int, int] | None = None  # area, x, y, w, h
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area:
            continue
        if w <= 0 or h <= 0:
            continue
        if w / h < ON_REVIEW_COLOR_MIN_RATIO:
            continue
        if y > h_img * ON_REVIEW_COLOR_MAX_Y_RATIO:
            continue
        if best is None or area > best[0]:
            best = (area, x, y, w, h)

    if best is None:
        return None

    _, x, y, w, h = best
    x = max(0, x - ON_REVIEW_COLOR_PADDING)
    y = max(0, y - ON_REVIEW_COLOR_PADDING)
    w = min(w_img - x, w + ON_REVIEW_COLOR_PADDING * 2)
    h = min(h_img - y, h + ON_REVIEW_COLOR_PADDING * 2)
    return x, y, w, h


def _find_on_review_box_by_template(
    gray: np.ndarray,
) -> tuple[int, int, int, int] | None:
    template = _load_on_review_template()
    if template is None:
        return None

    img_h, img_w = gray.shape[:2]
    template_h, template_w = template.shape[:2]
    if img_h <= 0 or img_w <= 0 or template_h <= 0 or template_w <= 0:
        return None

    edges_img = cv2.Canny(gray, ON_REVIEW_CANNY_LOW, ON_REVIEW_CANNY_HIGH)

    best_score = -1.0
    best_loc = (0, 0)
    best_size = (0, 0)

    for scale in _candidate_template_scales(img_w, template_w):
        scaled_w = max(1, int(template_w * scale))
        scaled_h = max(1, int(template_h * scale))
        if (
            scaled_w < ON_REVIEW_TEMPLATE_MIN_WIDTH
            or scaled_h < ON_REVIEW_TEMPLATE_MIN_HEIGHT
        ):
            continue
        if scaled_w >= img_w or scaled_h >= img_h:
            continue

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        scaled_template = cv2.resize(
            template, (scaled_w, scaled_h), interpolation=interp
        )

        res_gray = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res_gray)
        if max_val > best_score:
            best_score = max_val
            best_loc = max_loc
            best_size = (scaled_w, scaled_h)

        tmpl_edges = cv2.Canny(
            scaled_template, ON_REVIEW_CANNY_LOW, ON_REVIEW_CANNY_HIGH
        )
        res_edges = cv2.matchTemplate(edges_img, tmpl_edges, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res_edges)
        if max_val > best_score:
            best_score = max_val
            best_loc = max_loc
            best_size = (scaled_w, scaled_h)

    if best_score < ON_REVIEW_TEMPLATE_THRESHOLD:
        return None

    x, y = best_loc
    w, h = best_size
    if y > int(img_h * ON_REVIEW_TEMPLATE_MAX_Y_RATIO):
        return None
    return x, y, w, h


def _find_on_review_box(
    gray: np.ndarray, bgr: np.ndarray, force_threshold: int | None = None
) -> tuple[int, int, int, int] | None:
    """
    Ищет «НА ПРОВЕРКЕ», возвращает bbox (x, y, w, h) или None.
    Приоритет: цветовая маска -> совпадение по шаблону.
    """

    color_bbox = _find_on_review_box_by_color(bgr)
    if color_bbox:
        return color_bbox

    return _find_on_review_box_by_template(gray)


def _find_on_review_box_v1(
    gray: np.ndarray, force_threshold: int | None = None
) -> tuple[int, int, int, int] | None:
    """Версия 1: поиск плашки по шаблону без OCR."""

    return _find_on_review_box_by_template(gray)


def _remove_on_review_badge_v1(
    input_path: str, output_dir: Path, *, threshold: int | None = None
) -> bool:
    """Версия 1 — исходный алгоритм (старые константы, шаблонный поиск)."""

    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(input_path)
    if img is None:
        logger.info("Не удалось загрузить: %s", input_path)
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bbox = _find_on_review_box_v1(gray, force_threshold=threshold)
    if bbox is None:
        output_path = output_dir / Path(input_path).name.lower()
        cv2.imwrite(str(output_path), img)
        return False

    x, y, w, h = bbox

    padding_x = 0 if ON_REVIEW_ROI_EXTRA_WIDTH_V1 < 0 else ON_REVIEW_PADDING_X_V1
    padding_top = ON_REVIEW_PADDING_TOP_V1
    padding_bottom = ON_REVIEW_PADDING_BOTTOM_V1

    x1 = max(0, x - padding_x)
    roi_width = max(
        int(w * ON_REVIEW_ROI_WIDTH_MULTIPLIER_V1),
        img.shape[1] - x1
        if ON_REVIEW_ROI_EXTRA_WIDTH_V1 < 0
        else w + ON_REVIEW_ROI_EXTRA_WIDTH_V1,
    )
    x2 = min(img.shape[1], x1 + roi_width)
    y1 = max(0, y - padding_top)
    y2 = min(img.shape[0], y + h + padding_bottom)

    strip_height = y2 - y1
    if strip_height > 0:
        roi_y2 = min(
            img.shape[0],
            y2
            + max(
                strip_height * ON_REVIEW_ROI_BELOW_MULTIPLIER_V1,
                ON_REVIEW_ROI_BELOW_MIN_V1,
            ),
        )
        roi = img[y1:roi_y2, x1:x2, :]

        if strip_height < roi.shape[0]:
            roi[0 : roi.shape[0] - strip_height, :] = roi[strip_height:, :]
            fill_row = roi[
                roi.shape[0] - strip_height - 1 : roi.shape[0] - strip_height, :, :
            ]
            roi[roi.shape[0] - strip_height :, :] = fill_row

    output_path = output_dir / Path(input_path).name.lower()
    cv2.imwrite(str(output_path), img)
    return True


def _remove_on_review_badge_v2(
    input_path: str, output_dir: Path, *, threshold: int | None = None
) -> bool:
    """Версия 2 — с цветовой маской и шаблонным поиском."""

    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(input_path)
    if img is None:
        logger.info("Не удалось загрузить: %s", input_path)
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bbox = _find_on_review_box(gray, img, force_threshold=threshold)
    if bbox is None:
        output_path = output_dir / Path(input_path).name.lower()
        cv2.imwrite(str(output_path), img)
        return False

    x, y, w, h = bbox

    padding_x = ON_REVIEW_PADDING_X
    padding_top = ON_REVIEW_PADDING_TOP
    padding_bottom = ON_REVIEW_PADDING_BOTTOM

    guard_left = int(img.shape[1] * ON_REVIEW_GUARD_LEFT_RATIO)
    left_clip = max(
        ON_REVIEW_LEFT_CLAMP_MIN_PX,
        int(w * ON_REVIEW_LEFT_CLAMP_RATIO),
    )
    x1 = max(guard_left, x - padding_x + left_clip)

    max_x1 = x + int(w * 0.4)
    x1 = min(x1, max_x1)

    roi_width = max(
        int(w * ON_REVIEW_ROI_WIDTH_MULTIPLIER),
        img.shape[1] - x1
        if ON_REVIEW_ROI_EXTRA_WIDTH < 0
        else w + ON_REVIEW_ROI_EXTRA_WIDTH,
    )
    x2 = min(img.shape[1], x1 + roi_width)
    y1 = max(0, y - padding_top)
    y2 = min(img.shape[0], y + h + padding_bottom)

    strip_height = y2 - y1
    if strip_height > 0:
        roi_y2 = min(
            img.shape[0],
            y2
            + max(
                strip_height * ON_REVIEW_ROI_BELOW_MULTIPLIER,
                ON_REVIEW_ROI_BELOW_MIN,
            ),
        )
        roi = img[y1:roi_y2, x1:x2, :]

        if strip_height < roi.shape[0]:
            roi[0 : roi.shape[0] - strip_height, :] = roi[strip_height:, :]
            fill_row = roi[
                roi.shape[0] - strip_height - 1 : roi.shape[0] - strip_height, :, :
            ]
            roi[roi.shape[0] - strip_height :, :] = fill_row

    output_path = output_dir / Path(input_path).name.lower()
    cv2.imwrite(str(output_path), img)
    return True


def remove_on_review_badge(
    input_path: str,
    output_dir: Path,
    *,
    threshold: int | None = None,
    version: str = "v2",
) -> bool:
    """Прокси: позволяет выбрать версию алгоритма."""

    if version == "v1":
        return _remove_on_review_badge_v1(input_path, output_dir, threshold=threshold)
    return _remove_on_review_badge_v2(input_path, output_dir, threshold=threshold)
