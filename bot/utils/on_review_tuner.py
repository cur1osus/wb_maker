from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pytesseract
from PIL import Image

_PLUGIN_ROOT: Path | None = None


def _ensure_qt_env() -> None:
    """Настраиваем QT путь к плагинам до импорта Qt, чтобы не подцепить cv2/qt/plugins."""

    plugin_root: Path | None = None
    try:
        import importlib.util

        spec = importlib.util.find_spec("PyQt5")
        if spec and spec.origin:
            candidate = Path(spec.origin).parent / "Qt5" / "plugins"
            if candidate.exists():
                plugin_root = candidate
    except Exception:
        plugin_root = None

    if plugin_root:
        global _PLUGIN_ROOT
        _PLUGIN_ROOT = plugin_root
        os.environ["QT_PLUGIN_PATH"] = str(plugin_root)
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugin_root / "platforms")

    if (
        not os.environ.get("QT_QPA_PLATFORM")
        and sys.platform.startswith("linux")
        and not os.environ.get("DISPLAY")
    ):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"


_ensure_qt_env()

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets

if _PLUGIN_ROOT:
    QtCore.QCoreApplication.setLibraryPaths([str(_PLUGIN_ROOT)])

logger = logging.getLogger(__name__)

# Берем актуальные константы из общего модуля, если он доступен.
try:
    from on_review import (
        ON_REVIEW_LEFT_CLAMP_MIN_PX as DEFAULT_LEFT_CLAMP_MIN_PX,
    )
    from on_review import (
        ON_REVIEW_LEFT_CLAMP_RATIO as DEFAULT_LEFT_CLAMP_RATIO,
    )
    from on_review import (
        ON_REVIEW_PADDING_BOTTOM as DEFAULT_PADDING_BOTTOM,
    )
    from on_review import (
        ON_REVIEW_PADDING_TOP as DEFAULT_PADDING_TOP,
    )
    from on_review import (
        ON_REVIEW_PADDING_X as DEFAULT_PADDING_X,
    )
    from on_review import (
        ON_REVIEW_RE as ON_REVIEW_RE,
    )
    from on_review import (
        ON_REVIEW_ROI_BELOW_MIN as DEFAULT_ROI_BELOW_MIN,
    )
    from on_review import (
        ON_REVIEW_ROI_BELOW_MULTIPLIER as DEFAULT_ROI_BELOW_MULT,
    )
    from on_review import (
        ON_REVIEW_ROI_EXTRA_WIDTH as DEFAULT_ROI_EXTRA_WIDTH,
    )
    from on_review import (
        ON_REVIEW_ROI_WIDTH_MULTIPLIER as DEFAULT_ROI_WIDTH_MULT,
    )
    from on_review import (
        ON_REVIEW_THRESHOLD as DEFAULT_THRESHOLD,
    )
    from on_review import (
        ON_REVIEW_THRESHOLDS as DEFAULT_THRESHOLDS,
    )
    from on_review import (
        ON_REVIEW_COLOR_MAX_Y_RATIO,
        ON_REVIEW_COLOR_MIN_AREA_RATIO,
        ON_REVIEW_COLOR_MIN_RATIO,
        ON_REVIEW_COLOR_PADDING,
        ON_REVIEW_GUARD_LEFT_RATIO as DEFAULT_LEFT_GUARD_RATIO,
        ON_REVIEW_HSV_LOWER,
        ON_REVIEW_HSV_UPPER,
    )
except Exception as exc:  # noqa: BLE001
    logger.warning("Не удалось загрузить актуальные константы: %s", exc)
    DEFAULT_THRESHOLD = 150
    DEFAULT_THRESHOLDS = [150, 140, 170, 190, 120, 200]
    DEFAULT_PADDING_X = 2
    DEFAULT_PADDING_TOP = 9
    DEFAULT_PADDING_BOTTOM = 15
    DEFAULT_ROI_WIDTH_MULT = 1.6
    DEFAULT_ROI_EXTRA_WIDTH = 260
    DEFAULT_ROI_BELOW_MULT = 2
    DEFAULT_ROI_BELOW_MIN = 65
    DEFAULT_LEFT_CLAMP_RATIO = 0.15
    DEFAULT_LEFT_CLAMP_MIN_PX = 12
    DEFAULT_LEFT_GUARD_RATIO = 0.14
    ON_REVIEW_RE: Final[re.Pattern[str]] = re.compile(r"НА\s*ПРОВЕРКЕ", re.IGNORECASE)
    ON_REVIEW_HSV_LOWER: Final[np.ndarray] = np.array([5, 140, 170], dtype=np.uint8)
    ON_REVIEW_HSV_UPPER: Final[np.ndarray] = np.array([22, 255, 255], dtype=np.uint8)
    ON_REVIEW_COLOR_MIN_AREA_RATIO: Final[float] = 0.0025
    ON_REVIEW_COLOR_MIN_RATIO: Final[float] = 3.5
    ON_REVIEW_COLOR_MAX_Y_RATIO: Final[float] = 0.55
    ON_REVIEW_COLOR_PADDING: Final[int] = 6

SCALE: Final[float] = 3.0
PSM_MODES: Final[list[int]] = [6, 7, 8, 13]


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
    left_clamp_ratio: float = DEFAULT_LEFT_CLAMP_RATIO
    left_clamp_min_px: int = DEFAULT_LEFT_CLAMP_MIN_PX
    guard_left_ratio: float = DEFAULT_LEFT_GUARD_RATIO


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
        default_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        if Path(default_cmd).exists():
            pytesseract.pytesseract.tesseract_cmd = default_cmd


_configure_tesseract()


class _OcrCache:
    def __init__(self, gray: np.ndarray):
        self.resized = cv2.resize(gray, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        self.cache: dict[tuple[int, bool], tuple[np.ndarray, dict]] = {}

    def get(self, thr: int, invert: bool) -> tuple[np.ndarray, dict]:
        key = (thr, invert)
        if key in self.cache:
            return self.cache[key]

        flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, binary = cv2.threshold(self.resized, thr, 255, flag)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel, iterations=1)

        pil_img = Image.fromarray(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB))
        ocr = _run_ocr(pil_img)
        self.cache[key] = (binary, ocr)
        return self.cache[key]


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
    gray: np.ndarray, bgr: np.ndarray, params: Params, cache: _OcrCache
) -> tuple[int, int, int, int] | None:
    color_bbox = _find_on_review_box_by_color(bgr)
    if color_bbox:
        return color_bbox

    thresholds: list[int] = []
    thresholds.append(params.threshold)
    thresholds.extend(t for t in DEFAULT_THRESHOLDS if t not in thresholds)

    tried: set[tuple[int, bool]] = set()

    for thr in thresholds:
        for invert in (True, False):
            key = (thr, invert)
            if key in tried:
                continue
            tried.add(key)

            binary, ocr = cache.get(thr, invert)
            texts = ocr.get("text", [])
            for i, raw_text in enumerate(texts):
                text = (raw_text or "").strip()
                if not text or not ON_REVIEW_RE.search(text):
                    continue

                x_raw = int(ocr["left"][i])
                y_raw = int(ocr["top"][i])
                w_raw = int(ocr["width"][i])
                h_raw = int(ocr["height"][i])

                x_r, y_r, w_r, h_r = _refine_bbox_from_binary(binary, x_raw, y_raw, w_raw, h_raw)

                x = int(x_r / SCALE)
                y = int(y_r / SCALE)
                w = int(w_r / SCALE)
                h = int(h_r / SCALE)
                return x, y, w, h

    return None


def _strip_badge(img: np.ndarray, bbox: tuple[int, int, int, int], params: Params) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x, y, w, h = bbox

    padding_x = params.padding_x
    padding_top = params.padding_top
    padding_bottom = params.padding_bottom

    guard_left = int(img.shape[1] * params.guard_left_ratio)
    left_clip = max(params.left_clamp_min_px, int(w * params.left_clamp_ratio))
    x1 = max(guard_left, x - padding_x + left_clip)
    max_x1 = x + int(w * 0.4)
    x1 = min(x1, max_x1)

    roi_width = max(
        int(w * params.roi_width_mult),
        img.shape[1] - x1 if params.roi_extra_width < 0 else w + params.roi_extra_width,
    )
    x2 = min(img.shape[1], x1 + roi_width)
    y1 = max(0, y - padding_top)
    y2 = min(img.shape[0], y + h + padding_bottom)

    result = img.copy()
    strip_height = y2 - y1
    if strip_height > 0:
        roi_y2 = min(
            img.shape[0],
            y2 + max(strip_height * params.roi_below_mult, params.roi_below_min),
        )
        roi = result[y1:roi_y2, x1:x2, :]
        if strip_height < roi.shape[0]:
            roi[0 : roi.shape[0] - strip_height, :] = roi[strip_height:, :]
            fill_row = roi[
                roi.shape[0] - strip_height - 1 : roi.shape[0] - strip_height, :, :
            ]
            roi[roi.shape[0] - strip_height :, :] = fill_row

    return result, (x1, y1, x2, y2)


class _ParamControl(QtWidgets.QWidget):
    changed = QtCore.pyqtSignal()

    def __init__(self, label: str, *, is_float: bool, min_v: float, max_v: float, step: float, value: float, factor: int = 1, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.factor = factor
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(int(min_v * factor), int(max_v * factor))
        self.slider.setValue(int(value * factor))

        if is_float:
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setSingleStep(step)
        else:
            spin = QtWidgets.QSpinBox()
            spin.setSingleStep(int(step))

        spin.setRange(min_v, max_v)
        spin.setValue(value)
        self.spin = spin

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel(label))
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(spin)

        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)

    def _on_slider(self, raw: int) -> None:
        val = raw / self.factor
        if isinstance(self.spin, QtWidgets.QSpinBox):
            val_to_set = int(round(val))
        else:
            val_to_set = val

        self.spin.blockSignals(True)
        self.spin.setValue(val_to_set)
        self.spin.blockSignals(False)
        self.changed.emit()

    def _on_spin(self, val: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(int(val * self.factor))
        self.slider.blockSignals(False)
        self.changed.emit()

    def value(self) -> float:
        return float(self.spin.value())

    def set_value(self, val: float) -> None:
        if isinstance(self.spin, QtWidgets.QSpinBox):
            val = int(round(val))
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setValue(int(val * self.factor))
        self.spin.setValue(val)
        self.slider.blockSignals(False)
        self.spin.blockSignals(False)


class TunerWindow(QtWidgets.QWidget):
    def __init__(self, image_path: Path, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("NA PROVERKE tuner (PyQt5)")
        self.resize(1100, 800)

        self.img_path = image_path
        self.img_bgr = cv2.imread(str(image_path))
        if self.img_bgr is None:
            raise RuntimeError(f"Не удалось открыть: {image_path}")
        self.gray = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)
        self.cache = _OcrCache(self.gray)

        self.params = Params()

        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(480, 480)
        self.status_label = QtWidgets.QLabel("...")
        self.status_label.setStyleSheet("font-weight: bold; color: #444;")

        self.controls: dict[str, _ParamControl] = {}
        controls_widget = self._build_controls()

        buttons = QtWidgets.QHBoxLayout()
        btn_print = QtWidgets.QPushButton("Print params")
        btn_print.clicked.connect(self._print_params)
        btn_save = QtWidgets.QPushButton("Save preview")
        btn_save.clicked.connect(self._save_preview)
        btn_set = QtWidgets.QPushButton("Set params")
        btn_set.clicked.connect(self._set_params_dialog)
        buttons.addWidget(btn_print)
        buttons.addWidget(btn_set)
        buttons.addWidget(btn_save)
        buttons.addStretch(1)

        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(self.status_label)
        right_layout.addWidget(controls_widget)
        right_layout.addLayout(buttons)
        right_layout.addStretch(1)

        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.addWidget(self.image_label, stretch=2)
        main_layout.addLayout(right_layout, stretch=1)

        self._render_timer = QtCore.QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render_preview)

        self._last_rgb: np.ndarray | None = None
        self._render_preview()

    def _build_controls(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        def add_control(key: str, label: str, *, is_float: bool, min_v: float, max_v: float, step: float, factor: int = 1):
            value = getattr(self.params, key)
            ctrl = _ParamControl(label, is_float=is_float, min_v=min_v, max_v=max_v, step=step, value=value, factor=factor)
            ctrl.changed.connect(lambda _=None: self._on_param_changed(key))
            self.controls[key] = ctrl
            layout.addWidget(ctrl)

        add_control("threshold", "threshold", is_float=False, min_v=50, max_v=240, step=1, factor=1)
        add_control("padding_x", "padding_x", is_float=False, min_v=0, max_v=30, step=1, factor=1)
        add_control("padding_top", "padding_top", is_float=False, min_v=0, max_v=50, step=1, factor=1)
        add_control("padding_bottom", "padding_bottom", is_float=False, min_v=0, max_v=80, step=1, factor=1)
        add_control("roi_width_mult", "roi_width_mult", is_float=True, min_v=0.5, max_v=3.5, step=0.05, factor=100)
        add_control("roi_extra_width", "roi_extra_width", is_float=False, min_v=-1, max_v=600, step=1, factor=1)
        add_control("roi_below_mult", "roi_below_mult", is_float=False, min_v=0, max_v=20, step=1, factor=1)
        add_control("roi_below_min", "roi_below_min", is_float=False, min_v=0, max_v=800, step=5, factor=1)
        add_control("left_clamp_ratio", "left_clamp_ratio", is_float=True, min_v=0.0, max_v=0.8, step=0.01, factor=100)
        add_control("left_clamp_min_px", "left_clamp_min_px", is_float=False, min_v=0, max_v=120, step=1, factor=1)
        add_control("guard_left_ratio", "guard_left_ratio", is_float=True, min_v=0.0, max_v=0.35, step=0.01, factor=100)

        layout.addStretch(1)
        return widget

    def _on_param_changed(self, key: str) -> None:
        ctrl = self.controls[key]
        value = ctrl.value()
        if isinstance(getattr(self.params, key), int):
            value = int(round(value))
        setattr(self.params, key, value)
        self.status_label.setText("Пересчет...")
        self._render_timer.start(120)

    def _render_preview(self) -> None:
        bbox = _find_on_review_box(self.gray, self.img_bgr, self.params, self.cache)
        status = "NOT FOUND"
        preview = self.img_bgr.copy()
        roi_rect: tuple[int, int, int, int] | None = None

        if bbox:
            status = f"FOUND @ {bbox}"
            preview, roi_rect = _strip_badge(preview, bbox, self.params)
        else:
            status = "NOT FOUND"

        self._last_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        h, w, _ = self._last_rgb.shape
        qimg = QtGui.QImage(
            self._last_rgb.data, w, h, self._last_rgb.strides[0], QtGui.QImage.Format_RGB888
        )
        pix = QtGui.QPixmap.fromImage(qimg)
        scaled = pix.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)

        color = "#0c8b00" if bbox else "#c0392b"
        self.status_label.setStyleSheet(f"font-weight:bold; color:{color};")
        self.status_label.setText(status)

    def _print_params(self) -> None:
        print(
            "threshold={threshold} padding_x={padding_x} padding_top={padding_top} "
            "padding_bottom={padding_bottom} roi_width_mult={roi_width_mult} "
            "roi_extra_width={roi_extra_width} roi_below_mult={roi_below_mult} "
            "roi_below_min={roi_below_min} left_clamp_ratio={left_clamp_ratio} "
            "left_clamp_min_px={left_clamp_min_px} guard_left_ratio={guard_left_ratio}".format(
                **self.params.__dict__
            )
        )
        sys.stdout.flush()
        self.status_label.setText("Printed to stdout")

    def _save_preview(self) -> None:
        target = self.img_path.with_name(f"{self.img_path.stem}_preview.png")
        bbox = _find_on_review_box(self.gray, self.img_bgr, self.params, self.cache)
        preview = self.img_bgr.copy()
        if bbox:
            preview, roi_rect = _strip_badge(preview, bbox, self.params)
        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(target), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.status_label.setText(f"Saved → {target.name}")

    def _set_params_dialog(self) -> None:
        text, ok = QtWidgets.QInputDialog.getMultiLineText(
            self,
            "Set params",
            "Вставь строку вида:\nthreshold=... padding_x=... ... guard_left_ratio=...",
            self._params_as_str(),
        )
        if not ok:
            return
        applied = self._apply_params_from_string(text)
        if applied:
            self.status_label.setText("Params applied")
            self._render_preview()
        else:
            self.status_label.setText("No params applied (check format)")

    def _params_as_str(self) -> str:
        return (
            "threshold={threshold} padding_x={padding_x} padding_top={padding_top} "
            "padding_bottom={padding_bottom} roi_width_mult={roi_width_mult} "
            "roi_extra_width={roi_extra_width} roi_below_mult={roi_below_mult} "
            "roi_below_min={roi_below_min} left_clamp_ratio={left_clamp_ratio} "
            "left_clamp_min_px={left_clamp_min_px} guard_left_ratio={guard_left_ratio}"
        ).format(**self.params.__dict__)

    def _apply_params_from_string(self, raw: str) -> bool:
        pairs = re.findall(r"(\\w+)=([^\\s]+)", raw)
        if not pairs:
            return False

        updated = False
        for key, val_str in pairs:
            if not hasattr(self.params, key):
                continue
            current = getattr(self.params, key)
            try:
                if isinstance(current, int):
                    val = int(round(float(val_str)))
                else:
                    val = float(val_str)
            except ValueError:
                continue

            setattr(self.params, key, val)
            ctrl = self.controls.get(key)
            if ctrl:
                ctrl.set_value(val)
            updated = True

        return updated


def _main() -> None:
    parser = argparse.ArgumentParser(description="PyQt5 tuner for removing «НА ПРОВЕРКЕ».")
    parser.add_argument("image", type=Path, nargs="?", default=Path("test2.jpg"), help="Путь к скриншоту (по умолчанию test2.jpg).")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Файл не найден: {args.image}")
        sys.exit(1)

    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    win = TunerWindow(args.image)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    _main()
