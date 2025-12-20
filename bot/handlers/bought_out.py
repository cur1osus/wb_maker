from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Final

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import FSInputFile, ReplyKeyboardRemove
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
from bot.utils.process_bought_out import (
    clear_dirs_bought_out,
    ensure_user_dirs,
    get_paths,
    init_source_bought_out,
    process_image_v,
)

if TYPE_CHECKING:
    from aiogram.types import Message
    from redis.asyncio import Redis

router = Router()
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS: Final[set[str]] = {".png", ".jpg", ".jpeg"}
FILES_PREVIEW_LIMIT: Final[int] = 20


async def _processing_keyboard():
    return await rk_processing()


def _user_id(user: UserManager, message: Message) -> int:
    return getattr(user, "id_user", None) or (message.from_user.id if message.from_user else 0)


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
    intro = (
        "Загрузите PNG/JPG с плашкой «ОТКАЗАЛИСЬ» как документ, затем жмите «🚀 Старт».\n"
        "📂 «Файлы» — очередь, 🧹 «Очистить» — удалить загруженное."
    )
    await message.answer(intro, reply_markup=await _processing_keyboard())


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
    await message.answer("Отменено", reply_markup=ReplyKeyboardRemove())
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
        reply_markup=await _processing_keyboard(),
    )


@router.message(UserState.send_files_bo, F.text == BTN_FILES)
async def show_queue(
    message: Message,
    user: UserManager,
    state: FSMContext,
    redis: Redis | None = None,
) -> None:
    text = _render_queue(get_paths(_user_id(user, message)))
    await message.answer(text, reply_markup=await _processing_keyboard())


@router.message(UserState.send_files_bo, F.text == BTN_CLEAR)
async def clear_queue(
    message: Message,
    user: UserManager,
    state: FSMContext,
    redis: Redis | None = None,
) -> None:
    clear_dirs_bought_out(_user_id(user, message))
    await message.answer(
        "Очередь и результаты очищены.", reply_markup=await _processing_keyboard()
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
            reply_markup=await _processing_keyboard(),
        )
        return

    resized_vykupili, new_h, new_w = init_source_bought_out()
    msg = await message.answer(f"Обработка [0/{len_paths}]")
    success = 0
    for i, p in enumerate(paths, start=1):
        if process_image_v(resized_vykupili, new_h, new_w, p, output_dir):
            success += 1
        await msg.edit_text(f"Обработка [{i}/{len_paths}]")

    await _send_results(message, str(output_dir))
    clear_dirs_bought_out(user_id)

    await message.answer(
        f"Готово: {success}/{len_paths} файлов обработаны.",
        reply_markup=await _processing_keyboard(),
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
        reply_markup=await _processing_keyboard(),
    )
