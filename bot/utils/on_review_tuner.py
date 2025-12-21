from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytesseract

logger = logging.getLogger(__name__)

try:
    # Берем актуальные константы из общего модуля обработки «НА ПРОВЕРКЕ».
    from bot.utils.on_review import (
        ON_REVIEW_PADDING_BOTTOM as DEFAULT_PADDING_BOTTOM,
    )
    from bot.utils.on_review import (
        ON_REVIEW_PADDING_TOP as DEFAULT_PADDING_TOP,
    )
    from bot.utils.on_review import (
        ON_REVIEW_PADDING_X as DEFAULT_PADDING_X,
    )
    from bot.utils.on_review import (
        ON_REVIEW_ROI_BELOW_MIN as DEFAULT_ROI_BELOW_MIN,
    )
    from bot.utils.on_review import (
        ON_REVIEW_ROI_BELOW_MULTIPLIER as DEFAULT_ROI_BELOW_MULT,
    )
    from bot.utils.on_review import (
        ON_REVIEW_ROI_EXTRA_WIDTH as DEFAULT_ROI_EXTRA_WIDTH,
    )
    from bot.utils.on_review import (
        ON_REVIEW_ROI_WIDTH_MULTIPLIER as DEFAULT_ROI_WIDTH_MULT,
    )
    from bot.utils.on_review import (
        ON_REVIEW_THRESHOLD as DEFAULT_THRESHOLD,
    )
except Exception as exc:  # noqa: BLE001
    logger.warning("Не удалось загрузить актуальные константы: %s", exc)
    DEFAULT_THRESHOLD = 170
    DEFAULT_PADDING_X = 12
    DEFAULT_PADDING_TOP = 4
    DEFAULT_PADDING_BOTTOM = 12
    DEFAULT_ROI_WIDTH_MULT = 1.9
    DEFAULT_ROI_EXTRA_WIDTH = 180
    DEFAULT_ROI_BELOW_MULT = 6
    DEFAULT_ROI_BELOW_MIN = 260

PSM_MODES = [6, 7, 8, 13]
ON_REVIEW_RE = "НА\\s*ПРОВЕРКЕ"
WINDOW_NAME = "on_review_tuner"


@dataclass
class Params:
    threshold: int = DEFAULT_THRESHOLD
    padding_x: int = DEFAULT_PADDING_X
    padding_top: int = DEFAULT_PADDING_TOP
    padding_bottom: int = DEFAULT_PADDING_BOTTOM
    roi_width_mult: float = DEFAULT_ROI_WIDTH_MULT
    roi_extra_width: int = DEFAULT_ROI_EXTRA_WIDTH
    roi_below_mult: int = DEFAULT_ROI_BELOW_MULT
    roi_below_min: int = DEFAULT_ROI_BELOW_MIN


@dataclass
class _OcrCache:
    threshold: int
    thresh_img: np.ndarray
    ocr: dict


def _run_ocr(thresh_img: np.ndarray) -> dict:
    cfg_base = (
        "-c tessedit_char_whitelist="
        "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ "
        "--oem 3"
    )
    for psm in PSM_MODES:
        cfg = f"{cfg_base} --psm {psm}"
        try:
            return pytesseract.image_to_data(
                thresh_img,
                config=cfg,
                lang="rus",
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR error (psm=%s): %s", psm, exc)
            continue
    return {"text": [], "left": [], "top": [], "width": [], "height": []}


def _detect_and_strip(
    img: np.ndarray, params: Params, cache: _OcrCache | None
) -> tuple[np.ndarray, bool, tuple[int, int, int, int] | None, _OcrCache]:
    """Возвращает (картинка, найдено, bbox, cache)."""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    if cache is None or cache.threshold != params.threshold:
        _, thresh_inv = cv2.threshold(resized, params.threshold, 255, cv2.THRESH_BINARY_INV)
        thresh_rgb = cv2.cvtColor(thresh_inv, cv2.COLOR_GRAY2RGB)
        ocr = _run_ocr(thresh_rgb)
        cache = _OcrCache(threshold=params.threshold, thresh_img=thresh_rgb, ocr=ocr)

    result = img.copy()
    ocr = cache.ocr

    for i, raw_text in enumerate(ocr.get("text", [])):
        text = (raw_text or "").strip()
        if not text:
            continue
        if not re.search(ON_REVIEW_RE, text, re.IGNORECASE):
            continue

        x = int(ocr["left"][i] / 3.0)
        y = int(ocr["top"][i] / 3.0)
        w = int(ocr["width"][i] / 3.0)
        h = int(ocr["height"][i] / 3.0)

        padding_x = 0 if params.roi_extra_width < 0 else params.padding_x
        x1 = max(0, x - padding_x)
        roi_width = max(
            int(w * params.roi_width_mult),
            img.shape[1] - x1 if params.roi_extra_width < 0 else w + params.roi_extra_width,
        )
        x2 = min(img.shape[1], x1 + roi_width)
        y1 = max(0, y - params.padding_top)
        y2 = min(img.shape[0], y + h + params.padding_bottom)

        strip_height = y2 - y1
        if strip_height <= 0:
            continue

        roi_y2 = min(
            img.shape[0],
            y2 + max(strip_height * params.roi_below_mult, params.roi_below_min),
        )
        roi = result[y1:roi_y2, x1:x2, :]
        if strip_height >= roi.shape[0]:
            continue

        roi[0 : roi.shape[0] - strip_height, :] = roi[strip_height:, :]
        fill_row = roi[
            roi.shape[0] - strip_height - 1 : roi.shape[0] - strip_height, :, :
        ]
        roi[roi.shape[0] - strip_height :, :] = fill_row

        return result, True, (x1, y1, x2, y2), cache

    return img, False, None, cache


def _setup_trackbars(win: str, params: Params) -> None:
    cv2.createTrackbar("threshold", win, params.threshold, 250, lambda _=None: None)
    cv2.createTrackbar("padding_x", win, params.padding_x, 80, lambda _=None: None)
    cv2.createTrackbar("padding_top", win, params.padding_top, 40, lambda _=None: None)
    cv2.createTrackbar(
        "padding_bottom", win, params.padding_bottom, 60, lambda _=None: None
    )
    cv2.createTrackbar(
        "roi_width_mult_x100",
        win,
        int(params.roi_width_mult * 100),
        400,
        lambda _=None: None,
    )
    # 0 = до правого края, иначе пиксели
    extra_init = 0 if params.roi_extra_width < 0 else params.roi_extra_width
    cv2.createTrackbar("roi_extra_width", win, extra_init, 600, lambda _=None: None)
    cv2.createTrackbar(
        "roi_below_mult", win, params.roi_below_mult, 20, lambda _=None: None
    )
    cv2.createTrackbar(
        "roi_below_min", win, params.roi_below_min, 600, lambda _=None: None
    )


def _read_params(win: str) -> Params:
    extra_width_raw = cv2.getTrackbarPos("roi_extra_width", win)
    return Params(
        threshold=cv2.getTrackbarPos("threshold", win),
        padding_x=cv2.getTrackbarPos("padding_x", win),
        padding_top=cv2.getTrackbarPos("padding_top", win),
        padding_bottom=cv2.getTrackbarPos("padding_bottom", win),
        roi_width_mult=max(0.1, cv2.getTrackbarPos("roi_width_mult_x100", win) / 100.0),
        roi_extra_width=-1 if extra_width_raw == 0 else extra_width_raw,
        roi_below_mult=max(0, cv2.getTrackbarPos("roi_below_mult", win)),
        roi_below_min=cv2.getTrackbarPos("roi_below_min", win),
    )


def _print_params(params: Params) -> None:
    print(
        "threshold={threshold} padding_x={padding_x} padding_top={padding_top} "
        "padding_bottom={padding_bottom} roi_width_mult={roi_width_mult} "
        "roi_extra_width={roi_extra_width} roi_below_mult={roi_below_mult} "
        "roi_below_min={roi_below_min}".format(**params.__dict__)
    )
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tuner for removing «НА ПРОВЕРКЕ» badge."
    )
    parser.add_argument("image", type=Path, help="Путь к скриншоту.")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Файл не найден: {args.image}")
        sys.exit(1)

    img = cv2.imread(str(args.image))
    if img is None:
        print(f"Не удалось открыть: {args.image}")
        sys.exit(1)

    win = WINDOW_NAME
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.waitKey(1)  # дать Qt/GTK создать окно перед созданием трекбаров
    _setup_trackbars(win, Params())

    preview = img.copy()
    last_params: Params | None = None
    cache: _OcrCache | None = None
    while True:
        params = _read_params(win)
        if params != last_params:
            processed, found, roi, cache = _detect_and_strip(img, params, cache)
            preview = processed
            overlay = preview.copy()
            if roi:
                x1, y1, x2, y2 = roi
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 180, 255), 2)
            alpha = 0.35 if roi else 0.0
            preview = cv2.addWeighted(overlay, alpha, preview, 1 - alpha, 0)
            status = "FOUND" if found else "NOT FOUND"
            info_lines = [
                f"{status}  thresh={params.threshold} extra={'right' if params.roi_extra_width < 0 else params.roi_extra_width}px",
                f"pad x={params.padding_x} y={params.padding_top}/{params.padding_bottom}",
                f"roi mult={params.roi_width_mult:.2f} down mult={params.roi_below_mult} min={params.roi_below_min}",
                "Controls: p=print, w=save, q/esc=exit, 0 in extra→right edge",
            ]
            y_cursor = 30
            for line in info_lines:
                cv2.putText(
                    preview,
                    line,
                    (10, y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 200, 0) if found else (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                y_cursor += 24
            last_params = params

        cv2.imshow(win, preview)
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("p"):
            _print_params(params)
        if key == ord("w"):
            out_path = args.image.with_name(f"{args.image.stem}_preview.png")
            cv2.imwrite(str(out_path), preview)
            print(f"Сохранено: {out_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
