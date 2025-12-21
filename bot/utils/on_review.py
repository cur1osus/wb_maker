from __future__ import annotations

import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Final

import cv2
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

SCALE: Final[float] = 3.0
PSM_MODES: Final[list[int]] = [6, 7, 8, 13]
ON_REVIEW_RE: Final[re.Pattern[str]] = re.compile(r"НА\s*ПРОВЕРКЕ", re.IGNORECASE)

# Параметры обработки «НА ПРОВЕРКЕ».
ON_REVIEW_THRESHOLD: Final[int] = 233  # Порог бинаризации для поиска текста
ON_REVIEW_PADDING_X: Final[int] = (
    37  # Горизонтальный отступ слева (если EXTRA_WIDTH >= 0)
)
ON_REVIEW_PADDING_TOP: Final[int] = 16  # Отступ сверху относительно текста
ON_REVIEW_PADDING_BOTTOM: Final[int] = 12  # Отступ снизу относительно текста
ON_REVIEW_ROI_WIDTH_MULTIPLIER: Final[float] = (
    4.0  # Во сколько раз расширить ROI от ширины текста
)
ON_REVIEW_ROI_EXTRA_WIDTH: Final[int] = (
    239  # Доп. ширина вправо; -1 — тянем ROI до правого края
)
ON_REVIEW_ROI_BELOW_MULTIPLIER: Final[int] = (
    0  # Насколько глубоко вниз захватывать контент (множитель высоты строки)
)
ON_REVIEW_ROI_BELOW_MIN: Final[int] = (
    151  # Минимальная глубина вниз, если множителя не хватает
)


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


def remove_on_review_badge(input_path: str, output_dir: Path) -> bool:
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
    resized = cv2.resize(
        gray,
        None,
        fx=SCALE,
        fy=SCALE,
        interpolation=cv2.INTER_CUBIC,
    )
    _, thresh_inv = cv2.threshold(
        resized, ON_REVIEW_THRESHOLD, 255, cv2.THRESH_BINARY_INV
    )
    pil_for_ocr = Image.fromarray(cv2.cvtColor(thresh_inv, cv2.COLOR_GRAY2RGB))

    found_any = False

    for psm in PSM_MODES:
        custom_config = (
            f"--oem 3 --psm {psm} "
            "-c tessedit_char_whitelist="
            "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
            "абвгдеёжзийклмнопрстуфхцчшщъыьэюя "
        )
        try:
            ocr_data = pytesseract.image_to_data(
                pil_for_ocr,
                config=custom_config,
                lang="rus",
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ошибка OCR: %s", exc)
            break

        for i, raw_text in enumerate(ocr_data.get("text", [])):
            text = (raw_text or "").strip()
            if not text or not ON_REVIEW_RE.search(text):
                continue

            x = int(ocr_data["left"][i] / SCALE)
            y = int(ocr_data["top"][i] / SCALE)
            w = int(ocr_data["width"][i] / SCALE)
            h = int(ocr_data["height"][i] / SCALE)

            # Если тянем ROI до правого края (EXTRA_WIDTH < 0), не расширяем влево.
            padding_x = 0 if ON_REVIEW_ROI_EXTRA_WIDTH < 0 else ON_REVIEW_PADDING_X
            padding_top = ON_REVIEW_PADDING_TOP
            padding_bottom = ON_REVIEW_PADDING_BOTTOM

            x1 = max(0, x - padding_x)
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
            if strip_height <= 0:
                continue

            roi_y2 = min(
                img.shape[0],
                y2
                + max(
                    strip_height * ON_REVIEW_ROI_BELOW_MULTIPLIER,
                    ON_REVIEW_ROI_BELOW_MIN,
                ),
            )
            roi = img[y1:roi_y2, x1:x2, :]

            if strip_height >= roi.shape[0]:
                continue

            roi[0 : roi.shape[0] - strip_height, :] = roi[strip_height:, :]
            fill_row = roi[
                roi.shape[0] - strip_height - 1 : roi.shape[0] - strip_height, :, :
            ]
            roi[roi.shape[0] - strip_height :, :] = fill_row

            found_any = True
            break

        if found_any:
            break

    output_path = output_dir / Path(input_path).name.lower()
    cv2.imwrite(str(output_path), img)
    return found_any
