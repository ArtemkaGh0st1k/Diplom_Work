"""
Чтение систем в формате программы Harmony (C++): мастер-файл со списком
.hl файлов поверхностей и таблицы показателя преломления .nnn.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from optimizers.tracing.surface import Surface
from optimizers.tracing.system import Medium, OpticalSystem, constant_medium, table_medium

DEFAULT_LW0 = 0.0007   # мм, значение по умолчанию в таблице Form4 (700 нм)


@dataclass
class Registrar:
    """Параметры плоскости регистрации из мастер-файла."""

    x: float
    y: float
    z: float
    size: float


def _num(text: str) -> float:
    # Файлы писались в русской локали: десятичная запятая
    return float(text.strip().replace(',', '.'))


def read_nnn(path: Path) -> np.ndarray:
    """Таблица .nnn: 1000 значений показателя преломления с шагом 10 нм."""
    values = [_num(line) for line in Path(path).read_text().split() if line.strip()]
    return np.array(values[:1000])


def find_medium(n: float, directory: Path) -> Medium:
    """
    Перенос n1read/n2read: ищет таблицу '<n * 1e6>.nnn' (или тот же файл без
    расширения) в папке directory. Если таблицы нет - среда без дисперсии.
    """
    for name in (f"{round(n * 1e6)}.nnn", f"{round(n * 1e6)}"):
        path = Path(directory) / name
        if path.exists():
            return table_medium(read_nnn(path))
    return constant_medium(n)


def read_hl(path: Path) -> Surface:
    """
    Заголовок .hl файла (перенос ReadFile). Профиль из файла не читается:
    при трассировке C++ всё равно пересчитывает его по параметрам.
    Старые файлы не содержат Lw0 - тогда берётся значение по умолчанию.
    """
    lines = Path(path).read_text().splitlines()
    head = [line.strip() for line in lines[:11]]
    lw0 = _num(head[10]) if len(head) > 10 and _num(head[10]) < 0.01 else DEFAULT_LW0
    return Surface(num=int(_num(head[0])),
                   type=int(_num(head[1])),
                   P=_num(head[2]),
                   diameter=_num(head[3]),
                   thickness=_num(head[4]),
                   n1=_num(head[5]),
                   n2=_num(head[6]),
                   z_lens=_num(head[7]),
                   harm=int(_num(head[8])),
                   hr=_num(head[9]),
                   lw0=lw0)


def read_system(master_path: Path, **system_kwargs) -> tuple[OpticalSystem, Registrar]:
    """
    Мастер-файл: число поверхностей, имена .hl файлов, XRes, YRes, ZRes, SizeRes.
    """
    master_path = Path(master_path)
    directory = master_path.parent
    tokens = master_path.read_text().split()
    n_lens = int(tokens[0])
    names = tokens[1:1 + n_lens]
    rest = [_num(t) for t in tokens[1 + n_lens:1 + n_lens + 4]]

    surfaces = [read_hl(directory / name) for name in names]
    media = [(find_medium(s.n1, directory), find_medium(s.n2, directory)) for s in surfaces]
    system = OpticalSystem(surfaces, media, **system_kwargs)
    return system, Registrar(*rest) if len(rest) == 4 else None


def default_system(n_lens: int = 8, directory: Path = None, **system_kwargs) -> OpticalSystem:
    """
    Система «по умолчанию» из N2Click: P = +-20, D = 10, T = 2, Z = 5 * j,
    чередование сред 1.000002 / 1.546704, Hr = 0.1, Lw0 = 700 нм, Harm = 0.
    """
    surfaces = []
    for j in range(1, n_lens + 1):
        odd = j % 2 == 1
        surfaces.append(Surface(num=j, type=2, P=20.0 if odd else -20.0,
                                diameter=10.0, thickness=2.0,
                                n1=1.000002 if odd else 1.546704,
                                n2=1.546704 if odd else 1.000002,
                                z_lens=5.0 * j, harm=0, hr=0.1, lw0=DEFAULT_LW0))
    if directory is None:
        media = None
    else:
        media = [(find_medium(s.n1, directory), find_medium(s.n2, directory)) for s in surfaces]
    return OpticalSystem(surfaces, media, **system_kwargs)
