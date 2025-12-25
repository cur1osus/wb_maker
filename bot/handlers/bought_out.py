from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Final

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import FSInputFile
from aiogram.utils.media_group import MediaGroupBuilder

from bot.db.models import UserManager
from bot.keyboards.reply import (
    BTN_CANCEL,
    BTN_CLEAR,
    BTN_FILES,
    BTN_MAIN_BOUGHT_OUT,
    BTN_START,
    rk_processing,
)
from bot.states import UserState
from bot.utils import fn
from bot.utils.on_review import remove_on_review_badge
from bot.utils.process_bought_out import (
    clear_dirs_bought_out,
    ensure_user_dirs,
    get_paths,
    init_source_bought_out,
    process_image_v,
    process_image_v_small_phone,
)

if TYPE_CHECKING:
    from aiogram.types import Message
    from redis.asyncio import Redis

router = Router()
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS: Final[set[str]] = {".png", ".jpg", ".jpeg"}
FILES_PREVIEW_LIMIT: Final[int] = 20
REVIEW_LABEL: Final[str] = "🔴 На проверке"
REVIEW_SELECTED_PREFIX: Final[str] = "✅ "
STATE_KEY_REVIEW: Final[str] = "bo_use_review"
STATE_KEY_REVIEW_VERSION: Final[str] = "bo_review_version"
REVIEW_VERSIONS: Final[list[str]] = ["v1", "v2"]
REVIEW_VERSION_LABELS: Final[dict[str, str]] = {"v1": "⚙️ V1", "v2": "⚙️ V2"}
DEFAULT_REVIEW_VERSION: Final[str] = "v1"
SMALL_PHONE_LABEL: Final[str] = "Маленький телефон"
STATE_KEY_SMALL_PHONE: Final[str] = "bo_small_phone"


async def _review_enabled(state: FSMContext) -> bool:
    data = await state.get_data()
    return bool(data.get(STATE_KEY_REVIEW, False))


async def _small_phone_enabled(state: FSMContext) -> bool:
    data = await state.get_data()
    return bool(data.get(STATE_KEY_SMALL_PHONE, False))


async def _current_review_version(state: FSMContext) -> str:
    data = await state.get_data()
    version = data.get(STATE_KEY_REVIEW_VERSION, DEFAULT_REVIEW_VERSION)
    if version not in REVIEW_VERSIONS:
        return DEFAULT_REVIEW_VERSION
    return version


async def _processing_keyboard(state: FSMContext):
    review_on = await _review_enabled(state)
    review_version = await _current_review_version(state)
    small_phone_on = await _small_phone_enabled(state)

    label = f"{REVIEW_SELECTED_PREFIX}{REVIEW_LABEL}" if review_on else REVIEW_LABEL
    version_label = REVIEW_VERSION_LABELS.get(
        review_version, REVIEW_VERSION_LABELS[DEFAULT_REVIEW_VERSION]
    )
    small_phone_label = (
        f"{REVIEW_SELECTED_PREFIX}{SMALL_PHONE_LABEL}"
        if small_phone_on
        else SMALL_PHONE_LABEL
    )
    return await rk_processing([label, version_label, small_phone_label])


def _user_id(user: UserManager, message: Message) -> int:
    return getattr(user, "id_user", None) or (
        message.from_user.id if message.from_user else 0
    )


def _render_queue(paths: list[str]) -> str:
    if not paths:
        return "Очередь пуста. Пришли PNG как документ."

    preview = [Path(p).name for p in paths[:FILES_PREVIEW_LIMIT]]
    body = "\n".join(f"{i + 1}. {name}" for i, name in enumerate(preview))
    tail = ""
    if len(paths) > len(preview):
        tail = f"\n... и еще {len(paths) - len(preview)} файл(ов)"
    return f"В очереди {len(paths)} файл(ов):\n{body}{tail}"


async def _send_results(message: Message, folder: str) -> None:
    if not os.path.isdir(folder):
        await message.answer("Готовых файлов нет.")
        return

    files = sorted(os.listdir(folder))
    if not files:
        await message.answer("Готовых файлов нет.")
        return

    media_group = MediaGroupBuilder()
    counter = 0

    for file in files:
        if counter < 10:
            media_group.add_document(media=FSInputFile(f"{folder}/{file}"))
            counter += 1
        else:
            await message.bot.send_media_group(
                chat_id=message.chat.id, media=media_group.build()
            )
            media_group = MediaGroupBuilder()
            media_group.add_document(media=FSInputFile(f"{folder}/{file}"))
            counter = 1
    if media_group._media:
        await message.bot.send_media_group(
            chat_id=message.chat.id, media=media_group.build()
        )


async def _start_bought_out(
    message: Message,
    state: FSMContext,
) -> None:
    await fn.state_clear(state)
    await state.set_state(UserState.send_files_bo)
    await state.update_data(
        {
            STATE_KEY_REVIEW: False,
            STATE_KEY_REVIEW_VERSION: DEFAULT_REVIEW_VERSION,
            STATE_KEY_SMALL_PHONE: False,
        }
    )
    intro = (
        "Что делать:\n"
        "1) Пришлите PNG/JPG с плашкой «ОТКАЗАЛИСЬ» как документ.\n"
        "2) Нажмите «🚀 Старт».\n"
        "Переключатели:\n"
        "• 🟠 На проверке — удалять плашку.\n"
        "• 📱 Маленький телефон — для узких скринов (например, 750×1334).\n"
        "• ⚙️ V1/V2 — версия алгоритма (V2: маска и защита слева, V1: базовая).\n"
        "Сервис:\n"
        "• 📂 Файлы — показать очередь.\n"
        "• 🧹 Очистить — удалить загруженное."
    )
    await message.answer(intro, reply_markup=await _processing_keyboard(state))


@router.message(F.text == BTN_MAIN_BOUGHT_OUT)
async def bought_out_entry(
    message: Message,
    user: UserManager,
    state: FSMContext,
    redis: Redis | None = None,
) -> None:
    await _start_bought_out(message, state)


@router.message(UserState.send_files_bo, F.text == BTN_CANCEL)
async def cancel(
    message: Message,
    user: UserManager,
    state: FSMContext,
    redis: Redis | None = None,
) -> None:
    await fn.state_clear(state)
    await message.answer("Отменено")
    await fn.show_main_menu(message, state)


@router.message(UserState.send_files_bo, F.document)
async def send_files(
    message: Message,
    user: UserManager,
    state: FSMContext,
    redis: Redis | None = None,
) -> None:
    user_id = _user_id(user, message)
    input_dir, _ = ensure_user_dirs(user_id)

    file_name = message.document.file_name or "file"
    ext = Path(file_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        await message.answer("Принимаю PNG/JPG/JPEG. Отправьте файл как документ.")
        return

    target = input_dir / Path(file_name).name
    target.parent.mkdir(parents=True, exist_ok=True)

    await message.bot.download(
        message.document.file_id,
        target,
    )
    paths = get_paths(user_id)
    await message.answer(
        f"Файл {target.name} сохранен. В очереди {len(paths)}.",
        reply_markup=await _processing_keyboard(state),
    )


@router.message(UserState.send_files_bo, F.text == BTN_FILES)
async def show_queue(
    message: Message,
    user: UserManager,
    state: FSMContext,
    redis: Redis | None = None,
) -> None:
    text = _render_queue(get_paths(_user_id(user, message)))
    await message.answer(text, reply_markup=await _processing_keyboard(state))


@router.message(UserState.send_files_bo, F.text == BTN_CLEAR)
async def clear_queue(
    message: Message,
    user: UserManager,
    state: FSMContext,
    redis: Redis | None = None,
) -> None:
    clear_dirs_bought_out(_user_id(user, message))
    await message.answer(
        "Очередь и результаты очищены.", reply_markup=await _processing_keyboard(state)
    )


@router.message(
    UserState.send_files_bo,
    F.text.func(
        lambda text: (text or "").replace(REVIEW_SELECTED_PREFIX, "").strip()
        == REVIEW_LABEL
    ),
)
async def toggle_review(
    message: Message,
    user: UserManager,
    state: FSMContext,
    redis: Redis | None = None,
) -> None:
    current = await _review_enabled(state)
    await state.update_data({STATE_KEY_REVIEW: not current})
    status = "включен" if not current else "выключен"
    await message.answer(
        f"🟠 На проверке {status}.",
        reply_markup=await _processing_keyboard(state),
    )


@router.message(
    UserState.send_files_bo,
    F.text.func(
        lambda text: (text or "").replace(REVIEW_SELECTED_PREFIX, "").strip()
        == SMALL_PHONE_LABEL
    ),
)
async def toggle_small_phone(
    message: Message,
    user: UserManager,
    state: FSMContext,
    redis: Redis | None = None,
) -> None:
    current = await _small_phone_enabled(state)
    await state.update_data({STATE_KEY_SMALL_PHONE: not current})
    status = "включен" if not current else "выключен"
    await message.answer(
        f"📱 Маленький телефон {status}.",
        reply_markup=await _processing_keyboard(state),
    )


@router.message(
    UserState.send_files_bo,
    F.text.func(
        lambda text: (text or "").strip().replace(REVIEW_SELECTED_PREFIX, "")
        in REVIEW_VERSION_LABELS.values()
    ),
)
async def switch_review_version(
    message: Message,
    user: UserManager,
    state: FSMContext,
    redis: Redis | None = None,
) -> None:
    current = await _current_review_version(state)
    try:
        idx = REVIEW_VERSIONS.index(current)
    except ValueError:
        idx = 0
    next_version = REVIEW_VERSIONS[(idx + 1) % len(REVIEW_VERSIONS)]
    await state.update_data({STATE_KEY_REVIEW_VERSION: next_version})
    next_label = REVIEW_VERSION_LABELS[next_version]
    await message.answer(
        f"Версия алгоритма: {next_label}",
        reply_markup=await _processing_keyboard(state),
    )


@router.message(UserState.send_files_bo, F.text == BTN_START)
async def vu_start_cmd(
    message: Message,
    user: UserManager,
    state: FSMContext,
    redis: Redis | None = None,
) -> None:
    user_id = _user_id(user, message)
    input_dir, output_dir = ensure_user_dirs(user_id)
    paths = get_paths(user_id)
    len_paths = len(paths)
    if not len_paths:
        await message.answer(
            "Очередь пуста. Пришлите PNG/JPG как документ.",
            reply_markup=await _processing_keyboard(state),
        )
        return

    review_on = await _review_enabled(state)
    review_version = await _current_review_version(state)
    small_phone_on = await _small_phone_enabled(state)

    resized_vykupili = new_h = new_w = None
    if not small_phone_on:
        resized_vykupili, new_h, new_w = init_source_bought_out()

    clean_dir = output_dir / "_tmp_on_review"
    if clean_dir.exists():
        shutil.rmtree(clean_dir, ignore_errors=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    version_tag = review_version.upper() if review_on else ""
    phone_tag = "📱=ON" if small_phone_on else "📱=OFF"
    msg = await message.answer(
        f"Обработка [0/{len_paths}] • 🟠={'ON' if review_on else 'OFF'} {version_tag} • {phone_tag}"
    )
    success = 0
    for i, p in enumerate(paths, start=1):
        processed = (
            process_image_v_small_phone(p, clean_dir)
            if small_phone_on
            else process_image_v(resized_vykupili, new_h, new_w, p, clean_dir)
        )
        intermediate = clean_dir / Path(p).name
        if not intermediate.exists():
            intermediate = Path(p)

        if review_on:
            removed = remove_on_review_badge(
                str(intermediate), output_dir, version=review_version
            )
            final_candidate = output_dir / Path(intermediate).name.lower()
            if not final_candidate.exists() and intermediate.exists():
                try:
                    shutil.copy2(intermediate, final_candidate)
                except OSError as exc:
                    logger.warning(
                        "Не удалось скопировать результат %s: %s", intermediate, exc
                    )
            if processed or removed:
                success += 1
        else:
            final_candidate = output_dir / Path(intermediate).name
            try:
                shutil.copy2(intermediate, final_candidate)
                success += 1
            except OSError as exc:
                logger.warning(
                    "Не удалось скопировать результат %s: %s", intermediate, exc
                )

        await msg.edit_text(
            f"Обработка [{i}/{len_paths}] • 🟠={'ON' if review_on else 'OFF'} {version_tag} • {phone_tag}"
        )

    if clean_dir.exists():
        shutil.rmtree(clean_dir, ignore_errors=True)

    await _send_results(message, str(output_dir))
    clear_dirs_bought_out(user_id)

    await message.answer(
        f"Готово: {success}/{len_paths} файлов обработаны.",
        reply_markup=await _processing_keyboard(state),
    )


@router.message(UserState.send_files_bo)
async def vu_end_cmd(
    message: Message,
    user: UserManager,
    state: FSMContext,
    redis: Redis | None = None,
) -> None:
    await message.answer(
        "Пришлите PNG/JPG как документ или используйте кнопки ниже.",
        reply_markup=await _processing_keyboard(state),
    )
