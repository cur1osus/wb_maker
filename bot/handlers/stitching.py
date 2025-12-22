from __future__ import annotations

import logging
import os
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
    BTN_MAIN_STITCH,
    BTN_START,
    rk_processing,
)
from bot.states import UserState
from bot.utils import fn
from bot.utils.process_stitching import (
    clear_dirs_stitch,
    ensure_user_dirs,
    get_paths,
    pairs_from_queue,
    stitch_pair,
)

if TYPE_CHECKING:
    from aiogram.types import Message
    from redis.asyncio import Redis

router = Router()
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS: Final[set[str]] = {".png", ".jpg", ".jpeg"}
FILES_PREVIEW_LIMIT: Final[int] = 20


def _user_id(user: UserManager, message: Message) -> int:
    return getattr(user, "id_user", None) or (
        message.from_user.id if message.from_user else 0
    )


def _render_queue(paths: list[str]) -> str:
    if not paths:
        return "Очередь пуста. Пришли 2 файла (верх и низ) как документы."

    preview = [Path(p).name for p in paths[:FILES_PREVIEW_LIMIT]]
    body = "\n".join(f"{i + 1}. {name}" for i, name in enumerate(preview))
    tail = ""
    if len(paths) > len(preview):
        tail = f"\n... и еще {len(paths) - len(preview)} файл(ов)"
    warning = "\n⚠️ Нужное количество файлов — четное." if len(paths) % 2 else ""
    return f"В очереди {len(paths)} файл(ов):\n{body}{tail}{warning}"


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


async def _start_stitching(
    message: Message,
    state: FSMContext,
) -> None:
    await fn.state_clear(state)
    await state.set_state(UserState.send_files_stitch)
    intro = (
        "Что делать:\n"
        "1) Пришлите два файла: верх, затем низ (как документы).\n"
        "2) Нажмите «🚀 Старт».\n"
        "Очередь: можно загрузить несколько пар подряд, порядок сохраняется.\n"
        "Подписи «верх»/«низ» в названии помогают угадать порядок.\n"
        "Сервис: 📂 Файлы — очередь, 🧹 Очистить — удалить загруженное."
    )
    await message.answer(intro, reply_markup=await rk_processing())


@router.message(F.text == BTN_MAIN_STITCH)
async def stitch_entry(
    message: Message,
    user: UserManager,
    state: FSMContext,
    redis: Redis | None = None,
) -> None:
    await _start_stitching(message, state)


@router.message(UserState.send_files_stitch, F.text == BTN_CANCEL)
async def cancel(
    message: Message,
    user: UserManager,
    state: FSMContext,
    redis: Redis | None = None,
) -> None:
    await fn.state_clear(state)
    await message.answer("Отменено")
    await fn.show_main_menu(message, state)


@router.message(UserState.send_files_stitch, F.document)
async def receive_file(
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
    postfix = " Добавьте еще один, чтобы собрать пару." if len(paths) % 2 else ""
    await message.answer(
        f"Файл {target.name} сохранен. В очереди {len(paths)}.{postfix}",
        reply_markup=await rk_processing(),
    )


@router.message(UserState.send_files_stitch, F.text == BTN_FILES)
async def show_queue(
    message: Message,
    user: UserManager,
    state: FSMContext,
    redis: Redis | None = None,
) -> None:
    text = _render_queue(get_paths(_user_id(user, message)))
    await message.answer(text, reply_markup=await rk_processing())


@router.message(UserState.send_files_stitch, F.text == BTN_CLEAR)
async def clear_queue(
    message: Message,
    user: UserManager,
    state: FSMContext,
    redis: Redis | None = None,
) -> None:
    clear_dirs_stitch(_user_id(user, message))
    await message.answer(
        "Очередь и результаты очищены.", reply_markup=await rk_processing()
    )


@router.message(UserState.send_files_stitch, F.text == BTN_START)
async def start_stitching(
    message: Message,
    user: UserManager,
    state: FSMContext,
    redis: Redis | None = None,
) -> None:
    user_id = _user_id(user, message)
    input_dir, output_dir = ensure_user_dirs(user_id)
    paths = get_paths(user_id)

    if len(paths) < 2:
        await message.answer(
            "Нужно минимум два файла: верх и низ.",
            reply_markup=await rk_processing(),
        )
        return
    if len(paths) % 2 != 0:
        await message.answer(
            "Количество файлов должно быть четным. Добавьте или уберите один файл.",
            reply_markup=await rk_processing(),
        )
        return

    pairs = pairs_from_queue(paths)
    if not pairs:
        await message.answer(
            "Не нашел пар для сращивания. Пришлите файлы заново.",
            reply_markup=await rk_processing(),
        )
        return

    msg = await message.answer(f"Обработка [0/{len(pairs)}]")
    success = 0
    for idx, (top_path, bottom_path) in enumerate(pairs, start=1):
        result = stitch_pair(top_path, bottom_path, output_dir)
        if result:
            success += 1
        await msg.edit_text(f"Обработка [{idx}/{len(pairs)}]")

    await _send_results(message, str(output_dir))
    clear_dirs_stitch(user_id)

    await message.answer(
        f"Готово: {success}/{len(pairs)} файл(ов) срощено.",
        reply_markup=await rk_processing(),
    )


@router.message(UserState.send_files_stitch)
async def fallback(
    message: Message,
    user: UserManager,
    state: FSMContext,
    redis: Redis | None = None,
) -> None:
    await message.answer(
        "Пришлите PNG/JPG как документ или используйте кнопки ниже.",
        reply_markup=await rk_processing(),
    )
