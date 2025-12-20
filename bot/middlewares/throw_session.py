from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from aiogram import BaseMiddleware
from aiogram.types import Message
from sqlalchemy.ext.asyncio import AsyncSession

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker


class ThrowDBSessionMiddleware(BaseMiddleware):
    async def __call__(  # pyright: ignore
        self,
        handler: Callable[[Message, dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: dict[str, Any],
    ) -> Any:
        sessionmaker: async_sessionmaker[AsyncSession] = data["sessionmaker"]
        async with sessionmaker() as session:
            data["session"] = session
            return await handler(event, data)
