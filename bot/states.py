from aiogram.fsm.state import State, StatesGroup


class UserState(StatesGroup):
    send_files_bo = State()
    send_files_do = State()
