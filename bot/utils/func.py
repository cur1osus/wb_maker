import logging

from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from bot.keyboards.reply import rk_main_menu

logger = logging.getLogger(__name__)


class Function:
    @staticmethod
    async def show_main_menu(
        message: Message,
        state: FSMContext,
    ) -> None:
        await Function.state_clear(state)
        await message.answer(
            text="Главное меню: выберите режим обработки.",
            reply_markup=await rk_main_menu(),
        )

    @staticmethod
    async def state_clear(state: FSMContext) -> None:
        await state.clear()
