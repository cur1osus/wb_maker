from __future__ import annotations

import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

SCALE: Final[float] = 3.0
PSM_MODES: Final[list[int]] = [6, 7, 8, 13]
ON_REVIEW_RE: Final[re.Pattern[str]] = re.compile(r"НА\s*ПРОВЕРКЕ", re.IGNORECASE)


# Параметры обработки «НА ПРОВЕРКЕ».
ON_REVIEW_THRESHOLD: Final[int] = 150  # Базовый порог бинаризации для поиска текста
ON_REVIEW_THRESHOLDS: Final[list[int]] = [150, 140, 170, 190, 120, 200]
ON_REVIEW_PADDING_X: Final[int] = 3  # Горизонтальный отступ слева только от текста
ON_REVIEW_PADDING_TOP: Final[int] = 12  # Отступ сверху относительно текста
ON_REVIEW_PADDING_BOTTOM: Final[int] = 30  # Отступ снизу относительно текста
ON_REVIEW_ROI_WIDTH_MULTIPLIER: Final[float] = (
    1.6  # Во сколько раз расширить ROI от ширины текста
)
ON_REVIEW_ROI_EXTRA_WIDTH: Final[int] = 600  # Доп. ширина вправо
ON_REVIEW_ROI_BELOW_MULTIPLIER: Final[int] = (
    2  # Насколько глубоко вниз захватывать контент (множитель высоты строки)
)
ON_REVIEW_ROI_BELOW_MIN: Final[int] = (
    65  # Минимальная глубина вниз, если множителя не хватает
)
ON_REVIEW_LEFT_CLAMP_RATIO: Final[float] = 0.4  # Отсекаем слева, но не забираем фото
ON_REVIEW_LEFT_CLAMP_MIN_PX: Final[int] = (
    12  # Минимум пикселей слева, чтобы не задеть картинку
)
ON_REVIEW_GUARD_LEFT_RATIO: Final[float] = (
    0.14  # Доля ширины кадра, которую гарантированно не трогаем слева
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


def _configure_tesseract() -> None:
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd:
        pytesseract.pytesseract.tesseract_cmd = env_cmd
        return

    path_cmd = shutil.which("tesseract")
    if path_cmd:
        pytesseract.pytesseract.tesseract_cmd = path_cmd
        return

    if sys.platform.startswith("win"):
        default_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if Path(default_cmd).exists():
            pytesseract.pytesseract.tesseract_cmd = default_cmd


_configure_tesseract()


def _run_ocr(pil_img: Image.Image) -> dict:
    cfg_base = "-c tessedit_char_whitelist=АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ --oem 3"
    for psm in PSM_MODES:
        cfg = f"{cfg_base} --psm {psm}"
        try:
            return pytesseract.image_to_data(
                pil_img,
                config=cfg,
                lang="rus",
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("OCR error (psm=%s): %s", psm, exc)
            continue
    return {"text": [], "left": [], "top": [], "width": [], "height": []}


def _refine_bbox_from_binary(
    binary: np.ndarray, x: int, y: int, w: int, h: int
) -> tuple[int, int, int, int]:
    """
    Уточняет bbox по бинарной маске, чтобы не тянуть левый блок с фото.
    Возвращает bbox в координатах `binary`.
    """

    if w <= 0 or h <= 0:
        return x, y, w, h

    pad_x = max(4, w // 6)
    pad_y = max(2, h // 3)

    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(binary.shape[1], x + w + pad_x)
    y1 = min(binary.shape[0], y + h + pad_y)

    crop = binary[y0:y1, x0:x1]
    if crop.size == 0:
        return x, y, w, h

    white = int(np.count_nonzero(crop))
    black = crop.size - white
    text_mask = crop if white <= black else cv2.bitwise_not(crop)
    mask_bool = text_mask > 0

    col_hits = mask_bool.sum(axis=0)
    row_hits = mask_bool.sum(axis=1)
    active_cols = np.flatnonzero(col_hits > mask_bool.shape[0] * 0.1)
    active_rows = np.flatnonzero(row_hits > mask_bool.shape[1] * 0.1)

    if active_cols.size == 0 or active_rows.size == 0:
        return x, y, w, h

    left = x0 + int(active_cols[0])
    right = x0 + int(active_cols[-1])
    top = y0 + int(active_rows[0])
    bottom = y0 + int(active_rows[-1])
    return left, top, max(1, right - left), max(1, bottom - top)


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


def _find_on_review_box(
    gray: np.ndarray, bgr: np.ndarray, force_threshold: int | None = None
) -> tuple[int, int, int, int] | None:
    """
    Ищет «НА ПРОВЕРКЕ», возвращает bbox (x, y, w, h) или None.
    Пробуем несколько порогов и инверсий для высокой устойчивости.
    """

    color_bbox = _find_on_review_box_by_color(bgr)
    if color_bbox:
        return color_bbox

    resized = cv2.resize(gray, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    thresholds: list[int] = []
    if force_threshold is not None:
        thresholds.append(force_threshold)
    thresholds.extend(t for t in ON_REVIEW_THRESHOLDS if t not in thresholds)

    tried: set[tuple[int, bool]] = set()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    for thr in thresholds:
        for invert in (True, False):
            key = (thr, invert)
            if key in tried:
                continue
            tried.add(key)

            flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
            _, binary = cv2.threshold(resized, thr, 255, flag)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

            pil_img = Image.fromarray(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB))
            ocr = _run_ocr(pil_img)

            texts = ocr.get("text", [])
            for i, raw_text in enumerate(texts):
                text = (raw_text or "").strip()
                if not text or not ON_REVIEW_RE.search(text):
                    continue

                x_raw = int(ocr["left"][i])
                y_raw = int(ocr["top"][i])
                w_raw = int(ocr["width"][i])
                h_raw = int(ocr["height"][i])

                refined = _refine_bbox_from_binary(binary, x_raw, y_raw, w_raw, h_raw)
                x_r, y_r, w_r, h_r = refined

                x = int(x_r / SCALE)
                y = int(y_r / SCALE)
                w = int(w_r / SCALE)
                h = int(h_r / SCALE)
                return x, y, w, h

    return None


def _find_on_review_box_v1(
    gray: np.ndarray, force_threshold: int | None = None
) -> tuple[int, int, int, int] | None:
    """Версия 1: ровно как в последнем коммите — один порог, без уточнений и цвета."""

    resized = cv2.resize(gray, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    thr = force_threshold if force_threshold is not None else ON_REVIEW_THRESHOLD_V1
    _, thresh_inv = cv2.threshold(resized, thr, 255, cv2.THRESH_BINARY_INV)
    pil_for_ocr = Image.fromarray(cv2.cvtColor(thresh_inv, cv2.COLOR_GRAY2RGB))

    for psm in PSM_MODES:
        cfg = (
            f"--oem 3 --psm {psm} "
            "-c tessedit_char_whitelist=АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя "
        )
        try:
            ocr = pytesseract.image_to_data(
                pil_for_ocr,
                config=cfg,
                lang="rus",
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR error (psm=%s): %s", psm, exc)
            continue

        texts = ocr.get("text", [])
        for i, raw_text in enumerate(texts):
            text = (raw_text or "").strip()
            if not text or not ON_REVIEW_RE.search(text):
                continue

            x = int(ocr["left"][i] / SCALE)
            y = int(ocr["top"][i] / SCALE)
            w = int(ocr["width"][i] / SCALE)
            h = int(ocr["height"][i] / SCALE)
            return x, y, w, h

    return None


def _remove_on_review_badge_v1(
    input_path: str, output_dir: Path, *, threshold: int | None = None
) -> bool:
    """Версия 1 — исходный алгоритм (OCR-only, старые константы)."""

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
    """Версия 2 — с цветовой маской, уточнением bbox и защитой слева."""

    """
    Убирает надпись «НА ПРОВЕРКЕ» и сдвигает содержимое под ней вверх.
    Результат сохраняется в `output_dir/<имя_файла>`.
    """

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

    # Не уходим слишком далеко вправо — оставляем до 40% ширины текста.
    max_x1 = x + int(w * 0.4)
    x1 = min(x1, max_x1)

    roi_width = max(
        int(w * ON_REVIEW_ROI_WIDTH_MULTIPLIER),
        w + ON_REVIEW_ROI_EXTRA_WIDTH,
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
