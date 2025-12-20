from aiogram import Router

from . import bought_out, cmds, delivered

router = Router()
router.include_router(cmds.router)
router.include_router(bought_out.router)
router.include_router(delivered.router)
