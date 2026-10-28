"""
Связка трассировщика с параксиальной моделью проекта (optimizers/base.py).

Каждая гармоническая линза датасета превращается в две поверхности C++ модели:
1) гармонизированный параболоид (воздух -> стекло), родительская линза
   с фокусом f0 на расчётной длине волны: P = f0 * (n(λ0) - 1);
2) плоская грань подложки (стекло -> воздух).

Расстояние distance['i-(i+1)'] отсчитывается между первыми поверхностями
соседних линз, как между тонкими линзами в base.py.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from tracing.surface import MODE_GRATING, PARABOLA, PLANE, Surface
from tracing.system import OpticalSystem, bk7_medium, constant_medium
from utils.keys import DataSetKeys
from utils.unit import UnitType

MM = 1e-3   # метров в миллиметре


def build_system(dataset: dict[DataSetKeys, Any],
                 heights: Optional[dict[UnitType, list[float]]] = None,
                 diameter_mm: float = 10.0,
                 substrate_mm: float = 1.0,
                 mode: str = MODE_GRATING,
                 skip_zone_jumps: bool = True) -> OpticalSystem:
    """
    Строит OpticalSystem по датасету проекта.

    :param heights: высоты микрорельефа как в lmbd_focus_dict; None - согласованные
        с зонами H = m * λ0 / (n(λ0) - 1)
    :param substrate_mm: толщина подложки линзы (расстояние до плоской грани)
    :param mode: MODE_GRATING (согласовано с base.py) или MODE_CPP (как в C++)
    """
    units = dataset['unit']
    f0_unit = units['focus_0'].value[0] / MM
    lmbd0_unit = units['lambda_0'].value[0] / MM
    dist_unit = units['distance'].value[0] / MM

    glass = bk7_medium()
    air = constant_medium(1.0)

    surfaces, media = [], []
    z = 0.0
    for i in range(1, dataset['count_linse'] + 1):
        m = dataset['harmonica'][i]
        lw0 = dataset['lambda_0'][i] * lmbd0_unit
        f0 = dataset['focus_0'][i] * f0_unit
        n0 = glass(lw0)

        if heights is not None:
            h_unit = list(heights.keys())[0].value[0] / MM
            hr = list(heights.values())[0][i - 1] * h_unit
        else:
            hr = m * lw0 / (n0 - 1.0)

        p = f0 * (n0 - 1.0)
        sag = (diameter_mm / 2.0) ** 2 / (2.0 * p)
        surfaces.append(Surface(num=2 * i - 1, type=PARABOLA, P=p,
                                diameter=diameter_mm, thickness=sag + hr,
                                n1=1.0, n2=n0, z_lens=z, harm=m, hr=hr, lw0=lw0))
        media.append((air, glass))

        # Плоская грань: Section = 0, поверхность лежит ровно в z_lens
        surfaces.append(Surface(num=2 * i, type=PLANE, P=1.0,
                                diameter=diameter_mm, thickness=0.0,
                                n1=n0, n2=1.0, z_lens=z + substrate_mm,
                                harm=0, hr=hr, lw0=lw0))
        media.append((glass, air))

        if i != dataset['count_linse']:
            z += dataset['distance'][f'{i}-{i + 1}'] * dist_unit / \
                dataset['refractive_area'][f'{i}-{i + 1}']

    return OpticalSystem(surfaces, media, mode=mode, skip_zone_jumps=skip_zone_jumps)


def trace_focus_dict(dataset: dict[DataSetKeys, Any],
                     heights: Optional[dict[UnitType, list[float]]] = None,
                     lambda_massive: Optional[list[float]] = None,
                     n_rays: int = 10,
                     aperture: str = 'mean',
                     **build_kwargs) -> dict[float, float]:
    """
    Аналог BaseLensOptimizer.lmbd_focus_dict, но по трассировке лучей.

    Возвращает {λ, м: положение фокуса за плоской гранью последней линзы, м}.

    :param aperture: 'mean' - среднее по вееру лучей (как N12Click),
        'paraxial' - экстраполяция z(x^2) к оси по лучам с |x| >= 0.1 R
        (у самой оси нормаль по конечной разности C++ даёт смещение f * dr / 2r)
    """
    system = build_system(dataset, heights, **build_kwargs)
    if lambda_massive is None:
        l_unit = dataset['unit']['lower_lambda'].value[0]
        u_unit = dataset['unit']['upper_lambda'].value[0]
        lambda_massive = np.linspace(dataset['lower_lambda'] * l_unit,
                                     dataset['upper_lambda'] * u_unit, 601)

    z_last = system.surfaces[-1].z_lens
    result = {}
    for lmbd in lambda_massive:
        lw = lmbd / MM
        x_start, z_axe = system.axis_crossings(lw, n_rays, source=None)
        if aperture == 'paraxial':
            radius = system.surfaces[0].diameter / 2.0
            use = ~np.isnan(z_axe) & (np.abs(x_start) >= 0.1 * radius)
            z_focus = np.polyfit(x_start[use] ** 2, z_axe[use], 1)[1]
        else:
            z_focus = np.nanmean(z_axe)
        result[lmbd] = (z_focus - z_last) * MM
    return result
