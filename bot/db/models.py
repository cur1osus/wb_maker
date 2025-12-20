from enum import Enum

from sqlalchemy import BigInteger, ForeignKey, String
from sqlalchemy.dialects.mysql import BLOB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class BotFolder(Base):
    __tablename__ = "bot_folders"

    name: Mapped[str] = mapped_column(String(100))
    user_manager_id: Mapped[int] = mapped_column(
        ForeignKey("user_managers.id", ondelete="CASCADE"),
        nullable=False,
    )

    manager: Mapped["UserManager"] = relationship(back_populates="folders")
    bots: Mapped[list["Bot"]] = relationship(back_populates="folder")


class Bot(Base):
    __tablename__ = "bots"

    user_manager_id: Mapped[int] = mapped_column(
        ForeignKey("user_managers.id"), nullable=False
    )
    manager: Mapped["UserManager"] = relationship(back_populates="bots")
    folder_id: Mapped[int | None] = mapped_column(
        ForeignKey("bot_folders.id", ondelete="SET NULL"),
        nullable=True,
    )
    folder: Mapped["BotFolder | None"] = relationship(back_populates="bots")
    chats: Mapped[list["MonitoringChat"]] = relationship(
        back_populates="bot", lazy="selectin", cascade="all, delete-orphan"
    )
    jobs: Mapped[list["Job"]] = relationship(
        back_populates="bot",
        lazy="selectin",
        cascade="all, delete-orphan",
    )
    users_analyzed: Mapped[list["UserAnalyzed"]] = relationship(
        back_populates="bot", lazy="selectin"
    )

    name: Mapped[str] = mapped_column(String(50), nullable=True)
    phone: Mapped[str] = mapped_column(String(50))
    api_id: Mapped[int] = mapped_column(BigInteger)
    api_hash: Mapped[str] = mapped_column(String(100))
    path_session: Mapped[str] = mapped_column(String(100))
    is_connected: Mapped[bool] = mapped_column(default=False)
    is_started: Mapped[bool] = mapped_column(default=False)


class Job(Base):
    __tablename__ = "jobs"

    bot_id: Mapped[int] = mapped_column(ForeignKey("bots.id"), nullable=False)
    bot: Mapped[Bot] = relationship(back_populates="jobs")

    task: Mapped[str] = mapped_column(String(50))
    task_metadata: Mapped[int] = mapped_column(BLOB, nullable=True)
    answer: Mapped[int] = mapped_column(BLOB, nullable=True)


class JobName(Enum):
    processed_users = "processed_users"
    get_folders = "get_folders"
    get_chat_title = "get_chat_title"
    get_me_name = "get_me_name"


class MonitoringChat(Base):
    __tablename__ = "monitoring_chats"

    bot_id: Mapped[int] = mapped_column(ForeignKey("bots.id"), nullable=False)
    bot: Mapped[Bot] = relationship(back_populates="chats")

    chat_id: Mapped[str] = mapped_column(String(50))
    title: Mapped[str] = mapped_column(String(200), nullable=True)


class UserAnalyzed(Base):
    __tablename__ = "users_analyzed"

    bot_id: Mapped[int] = mapped_column(ForeignKey("bots.id"), nullable=True)
    bot: Mapped[Bot] = relationship(back_populates="users_analyzed")

    # id_user: Mapped[int] = mapped_column(BigInteger, unique=True)
    username: Mapped[str] = mapped_column(String(50), nullable=True)
    message_id: Mapped[str] = mapped_column(String(50), nullable=True)
    chat_id: Mapped[str] = mapped_column(String(50), nullable=True)
    additional_message: Mapped[str] = mapped_column(String(1000))
    sended: Mapped[bool] = mapped_column(default=False)
    accepted: Mapped[bool] = mapped_column(default=True)
    decision: Mapped[int] = mapped_column(BLOB, nullable=True)


class KeyWord(Base):
    __tablename__ = "keywords"

    user_manager_id: Mapped[int] = mapped_column(
        ForeignKey("user_managers.id"), nullable=False
    )
    manager: Mapped["UserManager"] = relationship(back_populates="keywords")

    word: Mapped[str] = mapped_column(String(500), nullable=False)


class IgnoredWord(Base):
    __tablename__ = "ignored_words"

    user_manager_id: Mapped[int] = mapped_column(
        ForeignKey("user_managers.id"), nullable=False
    )
    manager: Mapped["UserManager"] = relationship(back_populates="ignored_words")

    word: Mapped[str] = mapped_column(String(500), nullable=False)


class MessageToAnswer(Base):
    __tablename__ = "messages_to_answer"

    user_manager_id: Mapped[int] = mapped_column(
        ForeignKey("user_managers.id"), nullable=False
    )
    manager: Mapped["UserManager"] = relationship(back_populates="messages_to_answer")

    sentence: Mapped[str] = mapped_column(String(500), nullable=False)


class BannedUser(Base):
    __tablename__ = "banned_users"

    user_manager_id: Mapped[int] = mapped_column(
        ForeignKey("user_managers.id"), nullable=False
    )
    manager: Mapped["UserManager"] = relationship(back_populates="banned_users")

    id_user: Mapped[int] = mapped_column(BigInteger, nullable=True)
    username: Mapped[str] = mapped_column(String(50), nullable=True)
    is_banned: Mapped[bool] = mapped_column(default=False)


class UserManager(Base):
    __tablename__ = "user_managers"

    id_user: Mapped[int] = mapped_column(BigInteger, unique=True)
    username: Mapped[str] = mapped_column(String(50), nullable=True)
    users_per_minute: Mapped[int] = mapped_column(default=1)
    is_antiflood_mode: Mapped[bool] = mapped_column(default=False)
    limit_pack: Mapped[int] = mapped_column(default=5)

    bots: Mapped[list["Bot"]] = relationship(
        back_populates="manager",
        lazy="selectin",
        order_by=[
            Bot.is_connected.desc(),
            Bot.is_started.desc(),
        ],
        cascade="all, delete-orphan",
    )
    folders: Mapped[list["BotFolder"]] = relationship(
        back_populates="manager",
        lazy="selectin",
        cascade="all, delete-orphan",
    )
    keywords: Mapped[list["KeyWord"]] = relationship(
        back_populates="manager",
        lazy="selectin",
        cascade="all, delete-orphan",
    )
    ignored_words: Mapped[list["IgnoredWord"]] = relationship(
        back_populates="manager",
        lazy="selectin",
        cascade="all, delete-orphan",
    )
    messages_to_answer: Mapped[list["MessageToAnswer"]] = relationship(
        back_populates="manager",
        lazy="selectin",
        cascade="all, delete-orphan",
    )
    banned_users: Mapped[list["BannedUser"]] = relationship(
        back_populates="manager",
        lazy="selectin",
        cascade="all, delete-orphan",
    )

    async def get_obj_bot(self, bot_id: int) -> Bot | None:
        r: list[Bot] = [bot for bot in self.bots if bot.id == bot_id]
        return r[0] if r else None
