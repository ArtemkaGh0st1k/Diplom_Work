from __future__ import annotations

import sys
from pathlib import Path


def _ensure_project_root_on_syspath() -> None:
    """
    Позволяет запускать `gui/__main__.py` как файл:
        python gui/__main__.py
    В этом режиме sys.path[0] = ".../gui", и пакет `gui` не находится.
    """
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_ensure_project_root_on_syspath()

from gui.app import run_app  # noqa: E402


if __name__ == "__main__":
    run_app()