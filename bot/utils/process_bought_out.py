from __future__ import annotations

import logging
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Final, NamedTuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ASSETS_DIR: Final[Path] = Path(__file__).resolve().parent / "assets"
INPUT_ROOT: Final[Path] = Path("images_v")
OUTPUT_ROOT: Final[Path] = Path("result_images_v")
SUPPORTED_EXTENSIONS: Final[set[str]] = {".png", ".jpg", ".jpeg"}
TEMPLATE_THRESHOLD: Final[float] = 0.8

# Backward-compatible aliases (some code may rely on these names).
input_folder: Final[str] = str(INPUT_ROOT)
output_dir: Final[str] = str(OUTPUT_ROOT)


def _ensure_dirs() -> None:
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


_ensure_dirs()


def _user_dirs(user_id: int | str) -> tuple[Path, Path]:
    uid = str(user_id)
    input_dir = INPUT_ROOT / uid
    output_dir = OUTPUT_ROOT / uid
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir


def ensure_user_dirs(user_id: int | str) -> tuple[Path, Path]:
    return _user_dirs(user_id)


def get_paths(user_id: int | str) -> list[str]:
    input_dir, _ = _user_dirs(user_id)
    return sorted(
        str(p)
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


class _Assets(NamedTuple):
    vykupili: np.ndarray
    otkazalis_template: np.ndarray
    template_h: int
    template_w: int


def _resolve_asset(name: str) -> Path:
    candidates = [
        ASSETS_DIR / name,
        Path.cwd() / name,  # legacy location
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return ASSETS_DIR / name


@lru_cache(maxsize=1)
def _load_assets() -> _Assets | None:
    vykupili_path = _resolve_asset("vu.png")
    template_path = _resolve_asset("ot.png")

    vykupili = cv2.imread(str(vykupili_path), cv2.IMREAD_COLOR)
    if vykupili is None:
        logger.error("Не удалось загрузить ассет 'vu.png': %s", vykupili_path)
        return None

    otkazalis_template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if otkazalis_template is None:
        logger.error("Не удалось загрузить ассет 'ot.png': %s", template_path)
        return None

    template_h, template_w = otkazalis_template.shape[:2]
    return _Assets(
        vykupili=vykupili,
        otkazalis_template=otkazalis_template,
        template_h=template_h,
        template_w=template_w,
    )


def _copy_to_output(img_path: str, output_dir: Path) -> None:
    output_path = output_dir / Path(img_path).name
    try:
        shutil.copy2(img_path, output_path)
    except OSError as exc:
        logger.warning("Не удалось сохранить результат %s: %s", output_path, exc)


def init_source_bought_out(scale: float = 1.1) -> tuple[np.ndarray, int, int]:
    assets = _load_assets()
    if assets is None:
        logger.error("Не удалось инициализировать плашку: отсутствуют ассеты.")
        return np.empty((0, 0, 3), dtype=np.uint8), 0, 0

    if scale <= 0:
        logger.warning("Некорректный scale=%s, использую 1.0", scale)
        scale = 1.0

    h_orig, w_orig = assets.vykupili.shape[:2]
    new_w = max(1, int(w_orig * scale))
    new_h = max(1, int(h_orig * scale))

    if scale == 1.0:
        resized_vykupili = assets.vykupili.copy()
    else:
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        resized_vykupili = cv2.resize(
            assets.vykupili,
            (new_w, new_h),
            interpolation=interp,
        )

    return resized_vykupili, new_h, new_w


def process_image_v(
    resized_vykupili: np.ndarray,
    new_h: int,
    new_w: int,
    img_path: str,
    output_dir: Path,
    y_offset: int = -5,
) -> bool:
    """Заменяет найденную плашку «ОТКАЗАЛИСЬ» на «ВЫКУПИЛИ» и сохраняет файл в `result_images_v/`."""

    assets = _load_assets()
    if assets is None or new_h <= 0 or new_w <= 0 or resized_vykupili.size == 0:
        _copy_to_output(img_path, output_dir)
        return False

    img = cv2.imread(img_path)
    if img is None:
        logger.info("Не удалось загрузить: %s", img_path)
        return False

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_gray.shape[0] < assets.template_h or img_gray.shape[1] < assets.template_w:
        _copy_to_output(img_path, output_dir)
        return False

    res = cv2.matchTemplate(img_gray, assets.otkazalis_template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= TEMPLATE_THRESHOLD)

    for x, y in zip(*loc[::-1]):
        cv2.rectangle(
            img,
            (x, y),
            (x + assets.template_w, y + assets.template_h),
            (255, 255, 255),
            -1,
        )

        vx1 = x
        vx2 = x + new_w

        vy_center = y + (assets.template_h - new_h) // 2
        vy1 = vy_center + y_offset
        vy2 = vy1 + new_h

        if vx2 > img.shape[1] or vy2 > img.shape[0] or vx1 < 0 or vy1 < 0:
            logger.info("Не удалось вставить плашку (границы): %s", img_path)
            continue

        img[vy1:vy2, vx1:vx2] = resized_vykupili

    cv2.imwrite(str(output_dir / Path(img_path).name), img)
    return True


def clear_dirs_bought_out(user_id: int | str) -> None:
    input_dir, output_dir = _user_dirs(user_id)
    for folder in (output_dir, input_dir):
        if not folder.exists():
            continue
        for file_path in folder.iterdir():
            if not file_path.is_file():
                continue
            try:
                file_path.unlink()
            except OSError as exc:
                logger.debug("Не удалось удалить %s: %s", file_path, exc)
