from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import UserManager


async def _get_user_db_model(session: AsyncSession, user_id: int) -> UserManager | None:
    return await session.scalar(
        select(UserManager).where(UserManager.id_user == user_id)
    )
