"""Backend installation helpers for RVClaw."""

from .manager import (
    BackendSpec,
    build_install_script,
    get_backend_spec,
    list_backend_specs,
    plan_backend_install,
    run_backend_install,
)

__all__ = [
    "BackendSpec",
    "build_install_script",
    "get_backend_spec",
    "list_backend_specs",
    "plan_backend_install",
    "run_backend_install",
]
