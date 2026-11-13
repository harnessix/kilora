# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


def _start_fused_warm_up() -> None:
    try:
        from .. import fused

        fused.start_warm_up()
    except Exception:
        pass


_start_fused_warm_up()


def _start_prepare() -> None:
    try:
        from aiohttp import web
        from server import PromptServer

        from . import startup_prepare

        server = PromptServer.instance

        @server.routes.post("/kilora/prepare-startup")
        async def prepare_setting(request):
            data = await request.json()
            if not isinstance(data.get("enabled"), bool):
                raise web.HTTPBadRequest()
            startup_prepare.set_enabled(data["enabled"])
            return web.json_response({"enabled": startup_prepare.enabled()})

        async def stop_prepare(_app):
            startup_prepare.shutdown()

        if getattr(server, "app", None) is not None:
            server.app.on_shutdown.append(stop_prepare)
        startup_prepare.start()
    except Exception:
        pass


_start_prepare()


def _install_prompt_hook() -> None:
    try:
        from .live_write import install_prompt_hook

        install_prompt_hook()
    except Exception:
        pass


_install_prompt_hook()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
