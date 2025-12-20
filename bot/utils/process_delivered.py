from __future__ import annotations

import datetime
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Final

import cv2
import pytesseract
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

ASSETS_DIR: Final[Path] = Path(__file__).resolve().parent / "assets"
INPUT_ROOT: Final[Path] = Path("images_d")
OUTPUT_ROOT: Final[Path] = Path("result_images_d")
SUPPORTED_EXTENSIONS: Final[set[str]] = {".png", ".jpg", ".jpeg"}

# Backward-compatible aliases (some code may rely on these names).
input_folder: Final[str] = str(INPUT_ROOT)
output_dir: Final[str] = str(OUTPUT_ROOT)

FONT_SIZE: Final[float] = 46.5
PSM_MODES: Final[list[int]] = [6, 7, 8, 13]

SCALE_V2: Final[float] = 3.0
SCALE_V1: Final[float] = 3.0

DATE_DDMMYYYY_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<!\d)(\d{2})(\d{2})(\d{4})(?!\d)"
)
RETURNED_WORD_RE: Final[re.Pattern[str]] = re.compile(
    r"В[оа][зz3]вр[ао]щ[её]н",
    re.IGNORECASE,
)


def _ensure_dirs() -> None:
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


_ensure_dirs()


def _user_dirs(user_id: int | str) -> tuple[Path, Path]:
    """Returns user-isolated input/output dirs."""

    uid = str(user_id)
    in_dir = INPUT_ROOT / uid
    out_dir = OUTPUT_ROOT / uid
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    return in_dir, out_dir


def ensure_user_dirs(user_id: int | str) -> tuple[Path, Path]:
    return _user_dirs(user_id)


def _resolve_asset(name: str) -> Path:
    candidates = [
        ASSETS_DIR / name,
        Path.cwd() / name,  # legacy location
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return ASSETS_DIR / name


def _load_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = _resolve_asset("hr.ttf")
    try:
        return ImageFont.truetype(str(font_path), FONT_SIZE)
    except OSError as exc:
        logger.warning("Не удалось загрузить шрифт %s: %s", font_path, exc)
        return ImageFont.load_default()


FONT = _load_font()


def _configure_tesseract() -> None:
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd:
        pytesseract.pytesseract.tesseract_cmd = env_cmd
        return

    # If user has tesseract in PATH, prefer it.
    path_cmd = shutil.which("tesseract")
    if path_cmd:
        pytesseract.pytesseract.tesseract_cmd = path_cmd
        return

    # Reasonable Windows default.
    if sys.platform.startswith("win"):
        default_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if Path(default_cmd).exists():
            pytesseract.pytesseract.tesseract_cmd = default_cmd


_configure_tesseract()


def get_paths(user_id: int | str) -> list[str]:
    input_dir, _ = _user_dirs(user_id)
    return sorted(
        str(p)
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _copy_to_output(input_path: str, output_dir: Path) -> None:
    output_path = output_dir / Path(input_path).name.lower()
    try:
        shutil.copy2(input_path, output_path)
    except OSError as exc:
        logger.warning("Не удалось сохранить результат %s: %s", output_path, exc)


def _normalize_date(text: str) -> str:
    """Преобразует `ddmmyyyy` → `dd.mm.yyyy` внутри строки (если есть)."""

    match = DATE_DDMMYYYY_RE.search(text)
    if not match:
        return text

    dd, mm, yyyy = match.groups()
    try:
        dt = datetime.datetime.strptime(f"{dd}{mm}{yyyy}", "%d%m%Y")
        return text.replace(match.group(0), dt.strftime("%d.%m.%Y"))
    except ValueError:
        return text


def process_image_d_v2(input_path: str, output_dir: Path) -> bool:
    img = cv2.imread(input_path)
    if img is None:
        logger.info("Не удалось загрузить: %s", input_path)
        return False

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge((cl, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(
        gray,
        None,
        fx=SCALE_V2,
        fy=SCALE_V2,
        interpolation=cv2.INTER_CUBIC,
    )

    versions: list[tuple[str, "cv2.typing.MatLike"]] = []
    _, thresh_inv = cv2.threshold(resized, 70, 255, cv2.THRESH_BINARY_INV)
    versions.append(("dark", thresh_inv))
    _, thresh = cv2.threshold(resized, 180, 255, cv2.THRESH_BINARY)
    versions.append(("light", thresh))

    found_any = False
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    for _version_name, processed in versions:
        for psm in PSM_MODES:
            custom_config = (
                f"--oem 3 --psm {psm} "
                "-c tessedit_char_whitelist="
                "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
                "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
                "0123456789 "
            )

            try:
                ocr_data = pytesseract.image_to_data(
                    Image.fromarray(processed),
                    config=custom_config,
                    lang="rus",
                    output_type=pytesseract.Output.DICT,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Ошибка OCR: %s", exc)
                _copy_to_output(input_path, output_dir)
                return False

            for i, raw_text in enumerate(ocr_data.get("text", [])):
                text = (raw_text or "").strip()
                if not text:
                    continue

                if not RETURNED_WORD_RE.search(text):
                    continue

                x = int(ocr_data["left"][i] / SCALE_V2)
                y = int(ocr_data["top"][i] / SCALE_V2)
                w = int(ocr_data["width"][i] / SCALE_V2)
                h = int(ocr_data["height"][i] / SCALE_V2)

                padding_horiz = 8
                padding_vert_top = 3
                padding_vert_bot = 10

                x1 = max(0, x - padding_horiz)
                y1 = max(0, y - padding_vert_top)
                x2 = min(pil_img.width, x + w + padding_horiz)
                y2 = min(pil_img.height, y + h + padding_vert_bot)

                draw.rectangle((x1, y1, x2, y2), fill=(24, 24, 27))

                normalized = _normalize_date(text)
                new_text = RETURNED_WORD_RE.sub("Доставлен ", normalized)

                draw.text(
                    (max(0, x - 5), max(0, y - int(h * 0.1))),
                    new_text,
                    fill=(0, 179, 89),
                    font=FONT,
                )
                found_any = True

            if found_any:
                break
        if found_any:
            break

    output_path = output_dir / Path(input_path).name.lower()
    pil_img.save(output_path, format="PNG")
    return found_any


def process_image_d_v1(input_path: str, output_dir: Path) -> bool:
    original_image = cv2.imread(input_path)
    if original_image is None:
        logger.error("Не удалось загрузить изображение: %s", input_path)
        return False

    pil_img = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=SCALE_V1, fy=SCALE_V1)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    pil_for_ocr = Image.fromarray(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB))

    found_any = False

    for psm in PSM_MODES:
        custom_config = f"--oem 3 --psm {psm}"
        try:
            ocr_data = pytesseract.image_to_data(
                pil_for_ocr,
                config=custom_config,
                lang="rus",
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ошибка OCR: %s", exc)
            _copy_to_output(input_path, output_dir)
            return False

        texts = ocr_data.get("text", [])
        for i, raw_text in enumerate(texts):
            text = (raw_text or "").strip()
            if not text:
                continue

            if not re.search(r"Возвращ[её]н", text, re.IGNORECASE):
                continue

            x = int(ocr_data["left"][i] / SCALE_V1)
            y = int(ocr_data["top"][i] / SCALE_V1)
            w = int(ocr_data["width"][i] / SCALE_V1)
            h = int(ocr_data["height"][i] / SCALE_V1)

            # Some screenshots split date into the next token; avoid IndexError.
            next_text = ""
            if i + 1 < len(texts):
                next_text = (texts[i + 1] or "").strip()

            full_text = f"{text} {next_text}".strip()
            full_text = _normalize_date(full_text)

            draw.rectangle((x, y, x + w, y + h), fill="white")
            if next_text:
                x1 = int(ocr_data["left"][i + 1] / SCALE_V1)
                y1 = int(ocr_data["top"][i + 1] / SCALE_V1)
                w1 = int(ocr_data["width"][i + 1] / SCALE_V1)
                h1 = int(ocr_data["height"][i + 1] / SCALE_V1)
                draw.rectangle((x1, y1, x1 + w1, y1 + h1), fill="white")

            new_text = re.sub(
                r"Возвращ[её]н",
                "Доставлен",
                full_text,
                flags=re.IGNORECASE,
            )
            draw.text((max(0, x - 5), y), new_text, fill=(0, 179, 89), font=FONT)
            found_any = True

        if found_any:
            break

    output_path = output_dir / Path(input_path).name.lower()
    pil_img.save(output_path, format="PNG")
    return found_any


def process_image_d_vertical(input_path: str, output_dir: Path) -> bool:
    original_image = cv2.imread(input_path)
    if original_image is None:
        logger.error("Не удалось загрузить изображение: %s", input_path)
        return False

    pil_img = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=SCALE_V1, fy=SCALE_V1)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    pil_for_ocr = Image.fromarray(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB))

    found_any = False
    date_re = re.compile(r"\b(\d{2})[./](\d{2})[./](\d{4})\b")

    for psm in PSM_MODES:
        custom_config = f"--oem 3 --psm {psm}"
        try:
            ocr_data = pytesseract.image_to_data(
                pil_for_ocr,
                config=custom_config,
                lang="rus",
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ошибка OCR: %s", exc)
            _copy_to_output(input_path, output_dir)
            return False

        for i, raw_text in enumerate(ocr_data.get("text", [])):
            text = (raw_text or "").strip()
            if not text:
                continue

            if not (
                re.search(r"Возвращ[её]н", text, re.IGNORECASE) or date_re.search(text)
            ):
                continue

            x = int(ocr_data["left"][i] / SCALE_V1)
            y = int(ocr_data["top"][i] / SCALE_V1)
            w = int(ocr_data["width"][i] / SCALE_V1)
            h = int(ocr_data["height"][i] / SCALE_V1)

            draw.rectangle((x, y, x + w, y + h), fill="white")
            new_text = _normalize_date(text)

            new_text = re.sub(
                r"Возвращ[её]н",
                "Доставлен",
                new_text,
                flags=re.IGNORECASE,
            )
            draw.text((max(0, x - 5), y), new_text, fill=(0, 179, 89), font=FONT)
            found_any = True

        if found_any:
            break

    output_path = output_dir / Path(input_path).name.lower()
    pil_img.save(output_path, format="PNG")
    return found_any


def clear_dirs_d(user_id: int | str) -> None:
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
