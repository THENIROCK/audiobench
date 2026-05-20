"""Local Gradio GUI for audiobench.

The GUI is optional. Install with::

    pip install -e ".[gui]"

then launch with::

    audiobench --gui
"""

from __future__ import annotations

import os


def _disable_proxy_for_localhost() -> None:
    """Gradio 4.x + Starlette 1.0 refuses to launch when a system proxy hides
    localhost. This is the same workaround the spaces/leaderboard app applies.
    """
    no_proxy = "localhost,127.0.0.1,0.0.0.0,::1"
    os.environ.setdefault("NO_PROXY", no_proxy)
    os.environ.setdefault("no_proxy", no_proxy)


def _patch_gradio_client_schema() -> None:
    """Gradio 4.44's bundled ``gradio_client`` crashes when a component's
    JSON schema uses ``additionalProperties: true/false`` (a bool, not a
    dict). The crash makes ``GET /`` 500 during launch, which then makes
    Gradio think localhost is unreachable. Wrap the offending helpers so
    bool schemas degrade to ``Any``.
    """
    try:
        from gradio_client import utils as _gc_utils
    except ImportError:  # pragma: no cover - GUI not installed
        return
    if getattr(_gc_utils, "_audiobench_patched", False):
        return

    _orig_get_type = _gc_utils.get_type
    _orig_json_to_py = _gc_utils._json_schema_to_python_type

    def get_type(schema):  # type: ignore[no-redef]
        if isinstance(schema, bool):
            return "Any"
        return _orig_get_type(schema)

    def _json_schema_to_python_type(schema, defs):  # type: ignore[no-redef]
        if isinstance(schema, bool):
            return "Any"
        return _orig_json_to_py(schema, defs)

    _gc_utils.get_type = get_type
    _gc_utils._json_schema_to_python_type = _json_schema_to_python_type
    _gc_utils._audiobench_patched = True


def launch(
    *,
    host: str = "127.0.0.1",
    port: int = 7861,
    open_browser: bool = True,
    share: bool = False,
) -> None:
    """Build and serve the audiobench Gradio app."""
    _disable_proxy_for_localhost()

    try:
        import gradio  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - install guard
        import sys

        raise SystemExit(
            "gradio is not installed in this Python environment.\n"
            f"    python:    {sys.executable}\n"
            f"    error:     {exc}\n"
            "Install the GUI extra with:\n"
            "    pip install -e \".[gui]\"\n"
            "or:\n"
            "    pip install 'gradio>=4.40,<5.0'"
        ) from exc

    _patch_gradio_client_schema()
    from audiobench.gui.app import build_app

    app = build_app()
    app.queue().launch(
        server_name=host,
        server_port=port,
        inbrowser=open_browser,
        share=share,
        show_api=False,
    )


__all__ = ["launch"]
