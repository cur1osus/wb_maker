from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

INPUT_ROOT: Final[Path] = Path("images_s")
OUTPUT_ROOT: Final[Path] = Path("result_images_s")
SUPPORTED_EXTENSIONS: Final[set[str]] = {".png", ".jpg", ".jpeg"}

# Подсказки по именам файлов, чтобы угадывать «верх/низ», если пользователь подписывает файлы.
TOP_HINTS: Final[tuple[str, ...]] = ("верх", "top", "up", "topside")
BOTTOM_HINTS: Final[tuple[str, ...]] = ("низ", "bottom", "down", "bottomside")


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
    """
    Возвращает пути к загруженным файлам в порядке загрузки (по mtime).
    """

    input_dir, _ = _user_dirs(user_id)
    candidates = [
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return [str(p) for p in sorted(candidates, key=lambda p: p.stat().st_mtime)]


def clear_dirs_stitch(user_id: int | str) -> None:
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


def _role_from_name(path: Path) -> str | None:
    name = path.stem.lower()
    if any(hint in name for hint in TOP_HINTS):
        return "top"
    if any(hint in name for hint in BOTTOM_HINTS):
        return "bottom"
    return None


def order_pair(first: str, second: str) -> tuple[str, str]:
    """
    Пытается понять, какой файл верх, какой низ, по имени. Если не уверены — оставляем порядок.
    """

    p1, p2 = Path(first), Path(second)
    r1, r2 = _role_from_name(p1), _role_from_name(p2)
    if r1 == "top" and r2 == "bottom":
        return first, second
    if r1 == "bottom" and r2 == "top":
        return second, first
    return first, second


def stitch_pair(top_path: str, bottom_path: str, output_dir: Path) -> Path | None:
    """
    Склеивает верхний и нижний кадр по вертикали. Возвращает путь к результату или None.
    """

    try:
        with Image.open(top_path) as top_raw:
            top_img = ImageOps.exif_transpose(top_raw)
            top_icc = top_raw.info.get("icc_profile")
        with Image.open(bottom_path) as bottom_raw:
            bottom_img = ImageOps.exif_transpose(bottom_raw)
            bottom_icc = bottom_raw.info.get("icc_profile")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Не удалось открыть файлы для сращивания %s и %s: %s",
            top_path,
            bottom_path,
            exc,
        )
        return None

    use_alpha = "A" in top_img.getbands() or "A" in bottom_img.getbands()
    mode = "RGBA" if use_alpha else "RGB"
    if top_img.mode != mode:
        top_img = top_img.convert(mode)
    if bottom_img.mode != mode:
        bottom_img = bottom_img.convert(mode)

    target_width = max(top_img.width, bottom_img.width)
    if target_width <= 0:
        logger.warning("Некорректная ширина кадра: %s, %s", top_path, bottom_path)
        return None

    def _match_width(img: Image.Image) -> Image.Image:
        if img.width == target_width:
            return img.copy()
        ratio = target_width / max(1, img.width)
        new_height = max(1, int(img.height * ratio))
        return img.resize((target_width, new_height), Image.LANCZOS)

    top_resized = _match_width(top_img)
    bottom_resized = _match_width(bottom_img)

    bg_color = (255, 255, 255, 0) if use_alpha else (255, 255, 255)
    result = Image.new(
        mode,
        (target_width, top_resized.height + bottom_resized.height),
        color=bg_color,
    )
    result.paste(top_resized, (0, 0))
    result.paste(bottom_resized, (0, top_resized.height))

    def _slug(stem: str) -> str:
        cleaned = "".join(ch if ch.isalnum() else "_" for ch in stem.lower()).strip("_")
        return cleaned or "image"

    output_name = f"{_slug(Path(top_path).stem)}__{_slug(Path(bottom_path).stem)}.png"
    output_path = output_dir / output_name
    icc_profile = top_icc or bottom_icc
    save_kwargs = {"format": "PNG"}
    if icc_profile:
        save_kwargs["icc_profile"] = icc_profile
    try:
        result.save(output_path, **save_kwargs)
    except OSError as exc:
        logger.warning("Не удалось сохранить результат %s: %s", output_path, exc)
        return None

    return output_path


def pairs_from_queue(paths: list[str]) -> list[tuple[str, str]]:
    """
    Формирует пары (верх, низ) из очереди. В основе — порядок загрузки.
    """

    pairs: list[tuple[str, str]] = []
    for idx in range(0, len(paths), 2):
        if idx + 1 >= len(paths):
            break
        pairs.append(order_pair(paths[idx], paths[idx + 1]))
    return pairs
