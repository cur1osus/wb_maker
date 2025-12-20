from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aiogram import Router
from aiogram.filters import CommandStart

from bot.db.models import UserManager
from bot.utils import fn

if TYPE_CHECKING:
    from aiogram.fsm.context import FSMContext
    from aiogram.types import Message
    from sqlalchemy.ext.asyncio import AsyncSession


router = Router()
logger = logging.getLogger(__name__)


@router.message(CommandStart(deep_link=False))
async def start_cmd(
    message: Message,
    user: UserManager | None,
    session: AsyncSession,
    state: FSMContext,
) -> None:
    if not user and message.from_user:
        username = message.from_user.username or "none"
        new_user = UserManager(
            id_user=message.from_user.id,
            username=username,
        )
        user = new_user
        session.add(new_user)
        await message.bot.send_message(
            chat_id=474701274,
            text=f"User created: {user.id} {user.username}",
        )
        await session.commit()

    await fn.show_main_menu(message, state)
