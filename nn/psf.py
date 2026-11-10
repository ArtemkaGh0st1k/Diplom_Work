"""
Полихроматическая функция рассеяния точки (ФРТ) системы гармонических линз
по трассировке лучей (optimizers/tracing, режим grating).

Модель:
- точечный источник на бесконечности на оси, параллельный пучок;
- геометрическая оптика, на каждой длине волны работает один (доминирующий)
  порядок дифракции, дифракционная эффективность не учитывается;
- спектр 400–700 нм с равными весами;
- система осесимметрична, поэтому трассируется меридиональный веер лучей,
  а вес луча пропорционален площади кольца на входном зрачке (~ радиусу);
- плоскость регистрации - плоскость наилучшей фокусировки: минимум
  среднеквадратичного радиуса пятна по всем лучам и длинам волн.

Все длины внутри модуля в мм.
"""

from dataclasses import dataclass

import numpy as np

from optimizers.tracing import build_system
from utils.unit import UnitType

LAMBDA_NM = np.arange(400, 701, 10)    # 31 длина волны
N_RAYS = 96                            # лучей на радиусе зрачка
DIAMETER_MM = 10.0
SUBSTRATE_MM = 0.01
IMAGE_PX = 64
FIELD_MM = 0.5                         # половина стороны кадра ФРТ: пятна в данных не шире 0,5 мм
SUBPIXELS = 8                          # разбиение пикселя при интегрировании освещённости


@dataclass
class PsfResult:
    """Результат расчёта ФРТ одной системы."""

    image: np.ndarray        # (IMAGE_PX, IMAGE_PX), сумма = доля энергии внутри кадра
    z_best_mm: float         # плоскость наилучшей фокусировки относительно последней грани
    rms_mm: float            # среднеквадратичный радиус пятна в этой плоскости
    energy_in_field: float   # доля энергии, попавшей в кадр
    lost_rays: float         # доля потерянных лучей


def trace_rays(dataset: dict, heights_um: list[float]) -> tuple[np.ndarray, ...]:
    """
    Трассирует веер лучей сразу на всех длинах волн.

    Возвращает (rho, a, b, ok, z_last): высота луча на плоскости z равна
    a + b * z, rho - входная высота луча, ok - маска дошедших лучей.
    """
    system = build_system(dataset, {UnitType.MICROMETER: heights_um},
                          diameter_mm=DIAMETER_MM, substrate_mm=SUBSTRATE_MM)
    lws = LAMBDA_NM * 1e-6
    radius = DIAMETER_MM / 2
    # Центры колец равной ширины, крайнее кольцо не касается края апертуры
    rho = (np.arange(N_RAYS) + 0.5) / N_RAYS * radius * 0.98

    shape = (len(lws), N_RAYS)
    x = np.broadcast_to(rho, shape).ravel().copy()
    y = np.zeros_like(x)
    first = system.surfaces[0]
    z = np.full_like(x, first.z_lens - first.section.max() - 1.0)
    ax = np.zeros_like(x)
    ay = np.zeros_like(x)
    ok = np.ones_like(x, dtype=bool)

    for j, surface in enumerate(system.surfaces):
        m1, m2 = system.media[j]
        n1 = np.repeat([m1(lw) for lw in lws], N_RAYS)
        n2 = np.repeat([m2(lw) for lw in lws], N_RAYS)
        factor = np.repeat([system._slope_factor(j, lw) for lw in lws], N_RAYS)
        hit, x, y, z, ax, ay, az = surface.beam_lens(
            x, y, z, ax, ay, n1, n2, m_factor=factor,
            skip_zone_jumps=system.skip_zone_jumps)
        ok &= hit

    z_last = system.surfaces[-1].z_lens
    slope = np.where(az > 0, ax / np.where(az > 0, az, 1.0), 0.0)
    a = x + (z_last - z) * slope          # высота в плоскости последней грани
    rho_all = np.broadcast_to(rho, shape).ravel()
    return rho_all, a, slope, ok & (az > 0), z_last


def compute_psf(dataset: dict, heights_um: list[float],
                field_mm: float = FIELD_MM, image_px: int = IMAGE_PX) -> PsfResult:
    """Полихроматическая геометрическая ФРТ в плоскости наилучшей фокусировки."""
    rho, a, b, ok, _ = trace_rays(dataset, heights_um)
    w = rho[ok]
    a, b = a[ok], b[ok]

    # Плоскость наилучшей фокусировки: min sum w (a + b z)^2
    z_best = -np.sum(w * a * b) / np.sum(w * b * b)
    r = np.abs(a + b * z_best)
    rms = float(np.sqrt(np.sum(w * r * r) / np.sum(w)))

    # Радиальная гистограмма энергии, затем осесимметричное изображение
    n_bins = image_px * 4
    edges = np.linspace(0.0, field_mm * np.sqrt(2), n_bins + 1)
    energy, _ = np.histogram(r, bins=edges, weights=w / np.sum(w))
    # Сглаживание: каждый луч - кольцо конечной ширины (гауссово ядро в 1 бин)
    # (у оси энергия зеркально отражается относительно r = 0, чтобы не терять её)
    kernel = np.exp(-0.5 * (np.arange(-3, 4) / 1.0) ** 2)
    padded = np.concatenate([energy[2::-1], energy, np.zeros(3)])
    energy = np.convolve(padded, kernel / kernel.sum(), mode='valid')
    # Энергия кольца делится между субпикселями (SUB x SUB на пиксель), попавшими в кольцо.
    # Для колец целиком внутри кадра делим на фактическое число субпикселей - энергия
    # сохраняется точно; для колец, выходящих за кадр, - на площадь кольца.
    pixel = 2 * field_mm / image_px
    sub = SUBPIXELS
    sub_area = (pixel / sub) ** 2
    centers = (np.arange(image_px * sub) + 0.5) * (pixel / sub) - field_mm
    xx, yy = np.meshgrid(centers, centers)
    idx = np.minimum((np.sqrt(xx ** 2 + yy ** 2) / edges[1]).astype(int), n_bins - 1)
    count = np.bincount(idx.ravel(), minlength=n_bins).astype(float)
    ring_cells = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2) / sub_area
    inside = edges[1:] <= field_mm
    cells = np.where(inside & (count > 0), count, ring_cells)
    fine = (energy / cells)[idx]
    image = fine.reshape(image_px, sub, image_px, sub).sum(axis=(1, 3))
    energy_in_field = float(np.sum(image))

    return PsfResult(image=image.astype(np.float32), z_best_mm=float(z_best), rms_mm=rms,
                     energy_in_field=energy_in_field, lost_rays=float(1 - ok.mean()))


def encircled_energy(image: np.ndarray, radius_mm: float, field_mm: float = FIELD_MM) -> float:
    """Доля энергии в круге радиуса radius_mm (критерий качества по ФРТ)."""
    n = image.shape[0]
    pixel = 2 * field_mm / n
    centers = (np.arange(n) + 0.5) * pixel - field_mm
    xx, yy = np.meshgrid(centers, centers)
    return float(np.sum(image[np.sqrt(xx ** 2 + yy ** 2) <= radius_mm]))
