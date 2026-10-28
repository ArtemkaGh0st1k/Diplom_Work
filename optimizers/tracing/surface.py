"""
Перенос класса HarmLens из C++ проекта научного руководителя (Harmlens.cpp).

Один объект Surface — одна преломляющая поверхность (в C++ это одна строка
таблицы «линз»): слева среда n1, справа n2. Линза из стекла = две поверхности.

Соглашения оригинала (сохранены):
- все длины в миллиметрах, длины волн тоже в миллиметрах (550 нм = 0.00055);
- поверхность лежит в z = Z_Lens - Section(r), тело линзы между ней и
  плоскостью z = Z_Lens (см. условие fabs(zL - Z_Lens) < Section[i] в BeamLens);
- профиль задаётся таблицей Section из N = 100000 отсчётов на [0, D/2);
- гармоническая линза: Section = Section mod Hr (Harmonization), а дифракция
  моделируется умножением наклона профиля на коэффициент (Norm_m).

Трассировка векторизована: методы принимают массивы лучей.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Типы поверхностей (поле Type в C++)
PLANE = 0
SPHERE = 1
PARABOLA = 2
AXICON = 3
PARABOLA_MINUS_GENERAL = 4
GENERAL = 5

# Режимы модели дифракции на гармонической поверхности
MODE_CPP = 'cpp'          # как в C++: коэффициент Harm_lam, порядок без дисперсии
MODE_GRATING = 'grating'  # порядок с дисперсией, коэффициент по уравнению решётки


def harm_lam(m: int, lam0: float, lamt: float) -> float:
    """
    Перенос Harm_lam: отношение длины волны lamt к ближайшей
    резонансной длине волны lam0 * m / k, т.е. k * lamt / (m * lam0).
    """
    lf = m * lam0 / lamt
    li = int(lf)
    if abs(lf - li) < 0.5:
        lam_i = lam0 * m / li
    else:
        lam_i = lam0 * m / (li + 1)
    return lamt / lam_i


@dataclass
class Surface:
    """Преломляющая поверхность (перенос HarmLens)."""

    num: int = 1
    type: int = PARABOLA
    P: float = 20.0           # параметр кривизны (радиус при вершине), мм
    diameter: float = 10.0    # мм
    thickness: float = 2.0    # мм
    n1: float = 1.0           # среда до поверхности
    n2: float = 1.5           # среда после поверхности
    z_lens: float = 5.0       # мм
    harm: float = 0           # 0 - обычная, 1 - киноформ, m > 1 - гармоническая
    hr: float = 0.1           # высота рельефа, мм
    lw0: float = 0.0007       # расчётная длина волны, мм
    ns: float = 2.0           # степень обобщённой поверхности
    N: int = 100000
    x: np.ndarray = field(init=False, repr=False)
    section: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.build_profile()

    # ------------------------------------------------------------------
    # Профиль (P_Lens, S_Lens, G_Lens, A_Lens, P_A_Lens, Plane, Harmonization)
    # ------------------------------------------------------------------

    def build_profile(self) -> None:
        x = self.diameter / 2.0 / self.N * np.arange(self.N)
        if self.type == SPHERE:
            kf = 1.0 if self.P > 0 else -1.0
            h = self.P * self.P - x * x
            section = np.where(h > 0,
                               self.thickness + kf * (np.sqrt(np.abs(h)) - self.P),
                               0.0)
        elif self.type == PARABOLA:
            section = self.thickness - x * x / self.P / 2.0
        elif self.type == AXICON:
            section = self.thickness - x / self.P
        elif self.type == PARABOLA_MINUS_GENERAL:
            section = (self.thickness - x * x / self.P / 2.0
                       + x ** self.ns / self.P / 2.0)
        elif self.type == GENERAL:
            section = self.thickness - x ** self.ns / self.P / 2.0
        else:
            section = np.full(self.N, float(self.thickness))
        section = np.maximum(section, 0.0)

        if self.harm > 0:
            # В C++ hd = (int)(Section / h) - отбрасывание целого числа высот
            section = section - np.floor(section / self.hr) * self.hr

        self.x = x
        self.section = section

    # ------------------------------------------------------------------
    # Нормаль (Norm, Norm_m)
    # ------------------------------------------------------------------

    def normal(self, xi: np.ndarray, yi: np.ndarray,
               m: np.ndarray | float = 1.0,
               skip_zone_jumps: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Нормаль к поверхности по конечной разности профиля, наклон
        умножается на m (Norm_m). Ориентация как в C++: (-dx, -dy, 1)/delt.
        """
        dr = self.diameter / self.N / 2.0
        rad = np.sqrt(xi * xi + yi * yi)
        rd = (rad / dr).astype(np.int64)
        inside = rd < self.N - 1
        rd_c = np.minimum(rd, self.N - 2)

        diff = self.section[rd_c + 1] - self.section[rd_c]
        if skip_zone_jumps and self.harm > 0:
            # На скачке зоны берём разность у соседней пары отсчётов
            jump = np.abs(diff) > 0.5 * self.hr
            alt = np.maximum(rd_c - 1, 0)
            diff = np.where(jump, self.section[alt + 1] - self.section[alt], diff)

        df = m * diff / dr
        safe_rad = np.where(rad > 0, rad, 1.0)
        dx = np.where(rad > 0, df * xi / safe_rad, 0.0)
        dy = np.where(rad > 0, df * yi / safe_rad, 0.0)
        delt = np.sqrt(dx * dx + dy * dy + 1.0)

        nx = np.where(inside, -dx / delt, 0.0)
        ny = np.where(inside, -dy / delt, 0.0)
        nz = np.where(inside, 1.0 / delt, 1.0)
        return nx, ny, nz

    # ------------------------------------------------------------------
    # Пересечение и преломление (BeamLens)
    # ------------------------------------------------------------------

    def slope_factor(self, lw: float, n_glass_lw: float, n_glass_lw0: float,
                     mode: str) -> float:
        """
        Множитель наклона профиля для гармонической поверхности.

        - MODE_CPP: как в C++ (Lw/Lw0 при harm == 1, Harm_lam при harm > 1);
        - MODE_GRATING: порядок k = round(Hr * (n(lw) - 1) / lw), а множитель
          k * lw / (m * lw0) * (n(lw0) - 1) / (n(lw) - 1), что в параксиальном
          приближении даёт уравнение решётки f = m * lw0 * f0 / (k * lw).
        """
        if self.harm == 0:
            return 1.0
        if mode == MODE_CPP:
            if self.harm == 1:
                return lw / self.lw0
            return harm_lam(self.harm, self.lw0, lw)

        dn = abs(n_glass_lw - 1.0)
        dn0 = abs(n_glass_lw0 - 1.0)
        k = int(np.floor(self.hr * dn / lw + 0.5))
        if k < 1:
            raise ValueError(f"Поверхность {self.num}: нет рабочего порядка "
                             f"на длине волны {lw * 1e6:.1f} нм")
        return k * lw / (self.harm * self.lw0) * dn0 / dn

    def beam_lens(self, x0, y0, z0, ax, ay, n1: float, n2: float,
                  m_factor: float = 1.0, skip_zone_jumps: bool = False):
        """
        Перенос BeamLens для массива лучей.

        Возвращает (ok, xL, yL, zL, axL, ayL, azL); ok - маска лучей,
        попавших в поверхность (Err > 0 в C++).
        """
        x0, y0, z0, ax, ay = (np.asarray(v, dtype=float) for v in (x0, y0, z0, ax, ay))

        # Vec(): третья компонента направляющего вектора
        q = 1.0 - ax * ax - ay * ay
        az = np.where(q > 0, np.sqrt(np.abs(q)), 0.0)

        safe_az = np.where(az != 0, az, 1.0)
        par = np.where(az != 0, (self.z_lens - z0) / safe_az, 0.0)
        xL = x0 + par * ax
        yL = y0 + par * ay
        rL = np.where(par != 0, np.sqrt(xL * xL + yL * yL), 0.0)
        ok = (rL < self.diameter / 2.0) & (par != 0)

        # Бисекция: «внутри», если |z - Z_Lens| < Section(r)
        dx = self.x[1] - self.x[0]
        t0 = np.zeros_like(par)
        tL = par.copy()
        zL = np.full_like(par, self.z_lens)
        dt = tL - t0
        while np.any(np.abs(dt[ok]) > 1e-9):
            tM = (t0 + tL) / 2.0
            xL = x0 + tM * ax
            yL = y0 + tM * ay
            zL = z0 + tM * az
            rL = np.sqrt(xL * xL + yL * yL)
            i = np.minimum((rL / dx).astype(np.int64), self.N - 1)
            inside = np.abs(zL - self.z_lens) < self.section[i]
            tL = np.where(inside, tM, tL)
            t0 = np.where(inside, t0, tM)
            dt = tL - t0

        nx, ny, nz = self.normal(xL, yL, m_factor, skip_zone_jumps)

        # Закон преломления в форме C++ (знак -G согласован с ориентацией нормали)
        s = ax * nx + ay * ny + az * nz
        eps1 = np.where(s <= 1, np.arccos(np.clip(s, -1.0, 1.0)), np.pi / 2)
        sin2 = n1 / n2 * np.sin(eps1)
        eps2 = np.where(sin2 < 1, np.arcsin(np.clip(sin2, -1.0, 1.0)),
                        np.arcsin(-np.sin(eps1)))
        G = n2 * np.cos(eps2) - n1 * np.cos(eps1)
        axL = n1 * ax / n2 - nx * G / n2
        ayL = n1 * ay / n2 - ny * G / n2
        q = 1.0 - axL * axL - ayL * ayL
        azL = np.where(q > 0, np.sqrt(np.abs(q)), 0.0)

        return ok, xL, yL, zL, axL, ayL, azL


def beam_axe(xL, zL, axL, azL):
    """Перенос BeamAxe: z пересечения луча с осью (0, если луч параллелен оси)."""
    xL, zL, axL, azL = (np.asarray(v, dtype=float) for v in (xL, zL, axL, azL))
    safe = np.where(axL != 0, axL, 1.0)
    return np.where(axL != 0, zL - xL * azL / safe, 0.0)


def beam_plane_r(xL, yL, zL, axL, ayL, azL, z_r: float):
    """Перенос BeamPlaneR: точка пересечения луча с плоскостью z = z_r."""
    t = (z_r - zL) / azL
    return xL + t * axL, yL + t * ayL
