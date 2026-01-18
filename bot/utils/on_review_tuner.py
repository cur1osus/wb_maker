from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

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

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets

if _PLUGIN_ROOT:
    QtCore.QCoreApplication.setLibraryPaths([str(_PLUGIN_ROOT)])

logger = logging.getLogger(__name__)

from on_review import OnReviewParams as Params
from on_review import find_on_review_box, strip_badge
from process_bought_out import apply_vykupili_overlay


def _format_env_value(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    rendered = f"{value:.6f}".rstrip("0").rstrip(".")
    return rendered if rendered else "0"


def _update_env_file(path: Path, updates: dict[str, str]) -> None:
    lines: list[str] = []
    used: set[str] = set()
    pattern = re.compile(r"^\\s*([A-Za-z_][A-Za-z0-9_]*)\\s*=.*$")

    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()

    next_lines: list[str] = []
    for line in lines:
        match = pattern.match(line)
        if not match:
            next_lines.append(line)
            continue
        key = match.group(1)
        if key in updates:
            next_lines.append(f"{key}={updates[key]}")
            used.add(key)
        else:
            next_lines.append(line)

    for key, value in updates.items():
        if key in used:
            continue
        next_lines.append(f"{key}={value}")

    content = "\n".join(next_lines).rstrip("\n") + "\n"
    path.write_text(content, encoding="utf-8")


class _ParamControl(QtWidgets.QWidget):
    changed = QtCore.pyqtSignal()

    def __init__(
        self,
        label: str,
        *,
        is_float: bool,
        min_v: float,
        max_v: float,
        step: float,
        value: float,
        factor: int = 1,
        parent: QtWidgets.QWidget | None = None,
    ):
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

        self.params = Params()

        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(480, 480)
        self.status_label = QtWidgets.QLabel("...")
        self.status_label.setStyleSheet("font-weight: bold; color: #444;")
        self.apply_vykupili_checkbox = QtWidgets.QCheckBox("Apply «ВЫКУПИЛИ»")
        self.apply_vykupili_checkbox.setChecked(True)
        self.apply_vykupili_checkbox.stateChanged.connect(
            lambda _=None: self._render_preview()
        )

        self.controls: dict[str, _ParamControl] = {}
        controls_widget = self._build_controls()

        buttons = QtWidgets.QHBoxLayout()
        btn_print = QtWidgets.QPushButton("Print params")
        btn_print.clicked.connect(self._print_params)
        btn_save_env = QtWidgets.QPushButton("Save to .env")
        btn_save_env.clicked.connect(self._save_to_env)
        btn_save = QtWidgets.QPushButton("Save preview")
        btn_save.clicked.connect(self._save_preview)
        btn_set = QtWidgets.QPushButton("Set params")
        btn_set.clicked.connect(self._set_params_dialog)
        buttons.addWidget(btn_print)
        buttons.addWidget(btn_set)
        buttons.addWidget(btn_save_env)
        buttons.addWidget(btn_save)
        buttons.addStretch(1)

        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(self.status_label)
        right_layout.addWidget(self.apply_vykupili_checkbox)
        right_layout.addWidget(controls_widget)
        right_layout.addLayout(buttons)
        right_layout.addStretch(1)

        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.addWidget(self.image_label, stretch=2)
        main_layout.addLayout(right_layout, stretch=1)

        self._render_timer = QtCore.QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render_preview)

        self._last_rgb: object | None = None
        self._render_preview()

    def _build_controls(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        def add_control(
            key: str,
            label: str,
            *,
            is_float: bool,
            min_v: float,
            max_v: float,
            step: float,
            factor: int = 1,
        ):
            value = getattr(self.params, key)
            ctrl = _ParamControl(
                label,
                is_float=is_float,
                min_v=min_v,
                max_v=max_v,
                step=step,
                value=value,
                factor=factor,
            )
            ctrl.changed.connect(lambda _=None: self._on_param_changed(key))
            self.controls[key] = ctrl
            layout.addWidget(ctrl)

        add_control(
            "template_threshold",
            "template_threshold",
            is_float=True,
            min_v=0.01,
            max_v=0.9,
            step=0.01,
            factor=100,
        )
        add_control(
            "padding_x",
            "padding_x",
            is_float=False,
            min_v=-30,
            max_v=30,
            step=1,
            factor=1,
        )
        add_control(
            "padding_top",
            "padding_top",
            is_float=False,
            min_v=-30,
            max_v=50,
            step=1,
            factor=1,
        )
        add_control(
            "padding_bottom",
            "padding_bottom",
            is_float=False,
            min_v=-30,
            max_v=80,
            step=1,
            factor=1,
        )
        add_control(
            "roi_width_mult",
            "roi_width_mult",
            is_float=True,
            min_v=0.05,
            max_v=3.5,
            step=0.05,
            factor=100,
        )
        add_control(
            "roi_extra_width",
            "roi_extra_width",
            is_float=False,
            min_v=-800,
            max_v=600,
            step=1,
            factor=1,
        )
        add_control(
            "roi_below_mult",
            "roi_below_mult",
            is_float=False,
            min_v=-5,
            max_v=20,
            step=1,
            factor=1,
        )
        add_control(
            "roi_below_min",
            "roi_below_min",
            is_float=False,
            min_v=-200,
            max_v=800,
            step=5,
            factor=1,
        )
        add_control(
            "left_clamp_ratio",
            "left_clamp_ratio",
            is_float=True,
            min_v=-0.5,
            max_v=0.8,
            step=0.01,
            factor=100,
        )
        add_control(
            "left_clamp_min_px",
            "left_clamp_min_px",
            is_float=False,
            min_v=-100,
            max_v=120,
            step=1,
            factor=1,
        )
        add_control(
            "guard_left_ratio",
            "guard_left_ratio",
            is_float=True,
            min_v=-0.2,
            max_v=0.35,
            step=0.01,
            factor=100,
        )

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
        work_img = self.img_bgr.copy()
        if self.apply_vykupili_checkbox.isChecked():
            work_img, _ = apply_vykupili_overlay(work_img)
        gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
        bbox = find_on_review_box(gray, work_img, self.params)

        status = "NOT FOUND"
        preview = work_img.copy()

        if bbox:
            status = f"FOUND @ {bbox}"
            preview = strip_badge(preview, bbox, self.params)

        self._last_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        h, w, _ = self._last_rgb.shape
        qimg = QtGui.QImage(
            self._last_rgb.data,
            w,
            h,
            self._last_rgb.strides[0],
            QtGui.QImage.Format_RGB888,
        )
        pix = QtGui.QPixmap.fromImage(qimg)
        scaled = pix.scaled(
            self.image_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

        color = "#0c8b00" if bbox else "#c0392b"
        self.status_label.setStyleSheet(f"font-weight:bold; color:{color};")
        self.status_label.setText(status)

    def _print_params(self) -> None:
        print(
            "template_threshold={template_threshold} padding_x={padding_x} padding_top={padding_top} "
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
        work_img = self.img_bgr.copy()
        if self.apply_vykupili_checkbox.isChecked():
            work_img, _ = apply_vykupili_overlay(work_img)
        gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
        bbox = find_on_review_box(gray, work_img, self.params)
        preview = work_img.copy()
        if bbox:
            preview = strip_badge(preview, bbox, self.params)
        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(target), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.status_label.setText(f"Saved → {target.name}")

    def _save_to_env(self) -> None:
        env_map = {
            "template_threshold": "ON_REVIEW_TEMPLATE_THRESHOLD",
            "padding_x": "ON_REVIEW_PADDING_X",
            "padding_top": "ON_REVIEW_PADDING_TOP",
            "padding_bottom": "ON_REVIEW_PADDING_BOTTOM",
            "roi_width_mult": "ON_REVIEW_ROI_WIDTH_MULTIPLIER",
            "roi_extra_width": "ON_REVIEW_ROI_EXTRA_WIDTH",
            "roi_below_mult": "ON_REVIEW_ROI_BELOW_MULTIPLIER",
            "roi_below_min": "ON_REVIEW_ROI_BELOW_MIN",
            "left_clamp_ratio": "ON_REVIEW_LEFT_CLAMP_RATIO",
            "left_clamp_min_px": "ON_REVIEW_LEFT_CLAMP_MIN_PX",
            "guard_left_ratio": "ON_REVIEW_GUARD_LEFT_RATIO",
        }

        updates = {
            env_key: _format_env_value(getattr(self.params, param_key))
            for param_key, env_key in env_map.items()
        }
        _update_env_file(ENV_PATH, updates)
        self.status_label.setText(f"Saved to {ENV_PATH.name} (restart bot to apply)")

    def _set_params_dialog(self) -> None:
        text, ok = QtWidgets.QInputDialog.getMultiLineText(
            self,
            "Set params",
            "Вставь строку вида:\ntemplate_threshold=... padding_x=... ... guard_left_ratio=...",
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
            "template_threshold={template_threshold} padding_x={padding_x} padding_top={padding_top} "
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
    parser = argparse.ArgumentParser(
        description="PyQt5 tuner for removing «НА ПРОВЕРКЕ»."
    )
    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        default=Path("test2.jpg"),
        help="Путь к скриншоту (по умолчанию test2.jpg).",
    )
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
