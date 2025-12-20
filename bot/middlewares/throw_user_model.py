from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from aiogram import BaseMiddleware

from bot.db.func import _get_user_db_model

if TYPE_CHECKING:
    from aiogram.types import TelegramObject, Update, User
    from collections.abc import Awaitable, Callable


# 777000 is Telegram's user id of service messages
TG_SERVICE_USER_ID: Final[int] = 777000


class ThrowUserMiddleware(BaseMiddleware):
    async def __call__(  # pyright: ignore
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: Update,
        data: dict[str, Any],
    ) -> Any:
        user: User | None = data.get("event_from_user")

        if not user:
            return None

        match event.event_type:
            case "message":
                if user.is_bot is False and user.id != TG_SERVICE_USER_ID:
                    data["user"] = await _get_user_db_model(
                        session=data["session"],
                        user_id=user.id,
                    )
            case "callback_query":
                if user.is_bot is False and user.id != TG_SERVICE_USER_ID:
                    data["user"] = await _get_user_db_model(
                        session=data["session"],
                        user_id=user.id,
                    )

            case _:
                pass

        return await handler(event, data)
