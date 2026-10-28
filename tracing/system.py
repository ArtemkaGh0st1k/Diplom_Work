"""
Оптическая система из поверхностей Surface и расчёты из главной формы C++
(Unit1.cpp): трассировка пучка (N7Click/AHN1Click), положение фокуса
(N12Click) и фокальный сдвиг по спектру (N15Click).

Длины в мм, длины волн в мм (как в оригинале).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from tracing.surface import MODE_CPP, Surface, beam_axe
from utils.materials import n_bk7

# Показатель преломления среды как функция длины волны (в мм)
Medium = Callable[[float], float]


def constant_medium(n: float) -> Medium:
    """Среда без дисперсии."""
    return lambda lw: n


def bk7_medium() -> Medium:
    """Стекло BK7 (формула Селлмейера)."""
    return lambda lw: n_bk7(lw * 1e3)


def table_medium(values: np.ndarray) -> Medium:
    """
    Таблица из .nnn файла: 1000 значений с шагом 10 нм,
    индекс как в C++: n[(int)(Lw * 100000)], Lw в мм.

    Отбрасывание дробной части оставлено намеренно, как в Harmony.exe:
    при Lw = 550 * 0.000001 индекс равен 54, то есть берётся n(540 нм).
    """
    values = np.asarray(values, dtype=float)
    return lambda lw: float(values[int(lw * 100000)])


@dataclass
class TraceResult:
    """Результат трассировки пучка через систему."""

    ok: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    ax: np.ndarray
    ay: np.ndarray
    az: np.ndarray
    x_start: np.ndarray       # высота луча на входе (для анализа аберраций)


class OpticalSystem:
    """
    Система поверхностей (массив AHL в C++).

    :param surfaces: поверхности в порядке хода луча
    :param media: пары (n1(λ), n2(λ)) для каждой поверхности; None - константы
        из полей Surface.n1, Surface.n2
    :param mode: модель дифракции на гармонических поверхностях
        (MODE_CPP - как в C++, MODE_GRATING - согласованная с base.py)
    :param skip_zone_jumps: не считать нормаль по разности через скачок зоны
        (в C++ такой луч получает огромный наклон)
    """

    def __init__(self, surfaces: list[Surface],
                 media: Optional[list[tuple[Medium, Medium]]] = None,
                 mode: str = MODE_CPP,
                 skip_zone_jumps: bool = False):
        self.surfaces = surfaces
        if media is None:
            media = [(constant_medium(s.n1), constant_medium(s.n2)) for s in surfaces]
        self.media = media
        self.mode = mode
        self.skip_zone_jumps = skip_zone_jumps

    # ------------------------------------------------------------------

    def _slope_factor(self, j: int, lw: float) -> float:
        surface = self.surfaces[j]
        if surface.harm == 0:
            return 1.0
        m1, m2 = self.media[j]
        # Стеклом считаем среду с большим показателем на расчётной длине волны
        glass = m2 if m2(surface.lw0) >= m1(surface.lw0) else m1
        return surface.slope_factor(lw, glass(lw), glass(surface.lw0), self.mode)

    def trace(self, lw: float, x0, y0, z0, ax, ay) -> TraceResult:
        """
        Трассировка лучей через все поверхности на длине волны lw (мм).
        Луч считается потерянным, если промахнулся хотя бы по одной поверхности.
        """
        x, y, z, a_x, a_y = (np.atleast_1d(np.asarray(v, dtype=float))
                             for v in np.broadcast_arrays(x0, y0, z0, ax, ay))
        x_start = x.copy()
        ok = np.ones(x.shape, dtype=bool)
        a_z = np.sqrt(np.maximum(1.0 - a_x ** 2 - a_y ** 2, 0.0))

        for j, surface in enumerate(self.surfaces):
            n1, n2 = (medium(lw) for medium in self.media[j])
            hit, x, y, z, a_x, a_y, a_z = surface.beam_lens(
                x, y, z, a_x, a_y, n1, n2,
                m_factor=self._slope_factor(j, lw),
                skip_zone_jumps=self.skip_zone_jumps)
            ok &= hit

        return TraceResult(ok, x, y, z, a_x, a_y, a_z, x_start)

    def meridional_fan(self, n_rays: int = 10,
                       source: Optional[tuple[float, float, float]] = (0.0, 0.0, -1000.0)):
        """
        Лучи как в N12Click: x = -D/2 + mx * D / n_rays, mx = 1..n_rays/2 - 1, y = 0.
        source=None - параллельный пучок вместо точечного источника.
        """
        first = self.surfaces[0]
        dr = first.diameter / n_rays
        xt = -first.diameter / 2.0 + dr * np.arange(1, n_rays // 2)
        yt = np.zeros_like(xt)

        if source is None:
            z0 = first.z_lens - first.section.max() - 1.0
            return xt, yt, np.full_like(xt, z0), np.zeros_like(xt), np.zeros_like(xt)

        sx, sy, sz = source
        dis = np.sqrt((xt - sx) ** 2 + (yt - sy) ** 2 + (first.z_lens - sz) ** 2)
        ax = (xt - sx) / dis
        ay = (yt - sy) / dis
        return (np.full_like(xt, sx), np.full_like(xt, sy), np.full_like(xt, sz), ax, ay)

    def axis_crossings(self, lw: float, n_rays: int = 10,
                       source: Optional[tuple[float, float, float]] = (0.0, 0.0, -1000.0)):
        """z пересечения с осью для каждого луча веера (BeamAxe), nan - луч потерян."""
        result = self.trace(lw, *self.meridional_fan(n_rays, source))
        z_axe = beam_axe(result.x, result.z, result.ax, result.az)
        valid = result.ok & (z_axe != 0)
        return result.x_start, np.where(valid, z_axe, np.nan)

    def focal_position(self, lw: float, n_rays: int = 10,
                       source: Optional[tuple[float, float, float]] = (0.0, 0.0, -1000.0)) -> float:
        """Перенос N12Click: среднее по вееру лучей положение пересечения с осью, мм."""
        _, z_axe = self.axis_crossings(lw, n_rays, source)
        if np.all(np.isnan(z_axe)):
            return float('nan')
        return float(np.nanmean(z_axe))

    def focal_shift(self, lw_begin: float, lw_end: float, n_lw: int = 10,
                    n_rays: int = 10,
                    source: Optional[tuple[float, float, float]] = (0.0, 0.0, -1000.0)):
        """
        Перенос N15Click: положение фокуса на сетке lw_b + (lw_e - lw_b)/N * hs,
        hs = 0..N-1 (конец диапазона не включается, как в C++).

        :return: (длины волн, положения фокуса, сдвиг относительно минимума), мм
        """
        lws = lw_begin + (lw_end - lw_begin) / n_lw * np.arange(n_lw)
        positions = np.array([self.focal_position(lw, n_rays, source) for lw in lws])
        return lws, positions, positions - np.nanmin(positions)
