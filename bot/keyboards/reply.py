import logging
from typing import Final, Iterable

from aiogram.utils.keyboard import ReplyKeyboardBuilder

logger = logging.getLogger(__name__)

BTN_CANCEL: Final[str] = "✖️ Отмена"
BTN_MAIN_BOUGHT_OUT: Final[str] = "🛒 Выкуплен"
BTN_MAIN_DELIVERED: Final[str] = "📦 Доставлен"
BTN_MAIN_STITCH: Final[str] = "🪡 Сращивание"
BTN_START: Final[str] = "🚀 Старт"
BTN_FILES: Final[str] = "📂 Файлы"
BTN_CLEAR: Final[str] = "🧹 Очистить"


async def rk_cancel():
    builder = ReplyKeyboardBuilder()
    builder.button(text=BTN_CANCEL)
    builder.adjust(1)
    return builder.as_markup(resize_keyboard=True)


async def rk_main_menu():
    builder = ReplyKeyboardBuilder()
    builder.button(text=BTN_MAIN_BOUGHT_OUT)
    builder.button(text=BTN_MAIN_DELIVERED)
    builder.button(text=BTN_MAIN_STITCH)
    builder.adjust(1)
    return builder.as_markup(resize_keyboard=True)


async def rk_processing(mode_labels: str | Iterable[str] | None = None):
    """
    Клавиатура обработки: режим(ы) (опционально), старт, очередь, очистка, отмена.
    """

    builder = ReplyKeyboardBuilder()

    normalized_mode_labels: list[str] = []
    if mode_labels:
        if isinstance(mode_labels, str):
            normalized_mode_labels = [mode_labels]
        else:
            normalized_mode_labels = list(mode_labels)

    for label in normalized_mode_labels:
        builder.button(text=label)

    builder.button(text=BTN_START)
    builder.button(text=BTN_FILES)
    builder.button(text=BTN_CLEAR)
    builder.button(text=BTN_CANCEL)

    if normalized_mode_labels:
        builder.adjust(max(1, len(normalized_mode_labels)), 2, 2)
    else:
        builder.adjust(2, 2)

    return builder.as_markup(resize_keyboard=True)
