"""
Нейросети «параметры оптической системы -> ФРТ (изображение 64 x 64)».

PhysicsPsfNet (основная). Полносвязная сеть предсказывает распределение энергии
по радиальным кольцам (softmax: энергия неотрицательна и в сумме равна 1,
последний выход - энергия за пределами кадра). Фиксированный слой
RadialRenderer переводит кольца в изображение тем же оператором, что и
генератор ФРТ (nn/psf.py). Так в сеть встроено известное физическое свойство:
система осесимметрична и сохраняет энергию (по аналогии с физически-
информированной сетью из диссертации Д. А. Рымова, где в архитектуру встроена
модель дифракции).

ConvPsfNet (для сравнения). Полносвязная часть и свёрточный декодер,
изображение строится напрямую, без знания о симметрии.
"""

from typing import Union

import numpy as np
import torch
from torch import nn

from nn.dataset import MAX_LENSES
from nn.psf import FIELD_MM, IMAGE_PX, SUBPIXELS
from optimizers.two_lens_study import H_GRID_UM
from utils.materials import n_bk7

N_BINS = IMAGE_PX * 4
H_LOW, H_HIGH = float(H_GRID_UM[0]), float(H_GRID_UM[-1])
D_SCALE_MM = 50.0
F_SCALE_MM = 100.0
N_BASE_FEATURES = 4 + 1 + 3 * MAX_LENSES

# Фазовые признаки: фазовая задержка рельефа α = H (n(λ) - 1) / λ на опорных длинах волн.
# Рабочий порядок k = round(α), поэтому ФРТ ступенчато зависит от высоты;
# cos(2πα), sin(2πα) делают эту зависимость гладкой для сети.
PHASE_LAMBDA_UM = np.arange(0.40, 0.701, 0.05)
PHASE_DN = np.array([n_bk7(lam) - 1 for lam in PHASE_LAMBDA_UM], dtype=np.float32)
N_PHASE_FEATURES = 2 * len(PHASE_LAMBDA_UM) * MAX_LENSES

Array = Union[np.ndarray, torch.Tensor]


def n_features(phase: bool = True) -> int:
    """Длина вектора признаков."""
    return N_BASE_FEATURES + (N_PHASE_FEATURES if phase else 0)


def encode(n: Array, d: Array, f: Array, h: Array,
           phase: bool = True) -> Array:
    """
    Вектор признаков (numpy или torch, батч):
    one-hot числа линз (2..5), d / 50 мм, для каждой из 5 позиций:
    сила 100 мм / f, маска наличия линзы, высота, нормированная на полосу в [-1, 1];
    при phase=True ещё cos и sin фазы 2πα на 7 длинах волн для каждой линзы.
    """
    torch_in = torch.is_tensor(h)
    lib = torch if torch_in else np
    n = n if torch_in else np.asarray(n)
    mask = (lib.arange(MAX_LENSES)[None, :] < n[:, None])
    mask = mask.float() if torch_in else mask.astype(np.float32)
    one_hot = (n[:, None] == lib.arange(2, MAX_LENSES + 1)[None, :])
    one_hot = one_hot.float() if torch_in else one_hot.astype(np.float32)
    safe_f = lib.where(mask > 0, f, lib.ones_like(f) if torch_in else np.ones_like(f))
    power = F_SCALE_MM / safe_f * mask
    h_norm = (2 * (h - H_LOW) / (H_HIGH - H_LOW) - 1) * mask
    parts = [one_hot, (d / D_SCALE_MM)[:, None], power, mask, h_norm]
    if phase:
        coef = PHASE_DN / PHASE_LAMBDA_UM.astype(np.float32)
        coef = torch.tensor(coef) if torch_in else coef
        alpha = 2 * np.pi * h[:, :, None] * coef[None, None, :]      # (батч, линза, λ)
        m = mask[:, :, None]
        parts += [(lib.cos(alpha) * m).reshape(len(h), -1),
                  (lib.sin(alpha) * m).reshape(len(h), -1)]
    return torch.cat(parts, dim=1) if torch_in else np.concatenate(parts, axis=1).astype(np.float32)


def render_matrix(field_mm: float = FIELD_MM, image_px: int = IMAGE_PX,
                  sub: int = SUBPIXELS) -> torch.Tensor:
    """
    Линейный оператор (пиксели x кольца): энергия кольца -> изображение.
    Повторяет распределение энергии в nn/psf.compute_psf.
    """
    edges = np.linspace(0.0, field_mm * np.sqrt(2), N_BINS + 1)
    pixel = 2 * field_mm / image_px
    sub_area = (pixel / sub) ** 2
    centers = (np.arange(image_px * sub) + 0.5) * (pixel / sub) - field_mm
    xx, yy = np.meshgrid(centers, centers)
    idx = np.minimum((np.sqrt(xx ** 2 + yy ** 2) / edges[1]).astype(int), N_BINS - 1)
    count = np.bincount(idx.ravel(), minlength=N_BINS).astype(float)
    ring_cells = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2) / sub_area
    cells = np.where((edges[1:] <= field_mm) & (count > 0), count, ring_cells)

    pix_of_sub = (np.arange(image_px * sub) // sub)
    pix_index = (pix_of_sub[:, None] * image_px + pix_of_sub[None, :]).ravel()
    matrix = np.zeros((image_px * image_px, N_BINS))
    np.add.at(matrix, (pix_index, idx.ravel()), 1.0 / cells[idx.ravel()])
    return torch.tensor(matrix, dtype=torch.float32)


class RadialRenderer(nn.Module):
    """Фиксированный (необучаемый) слой: кольца -> изображение."""

    def __init__(self):
        super().__init__()
        self.register_buffer('matrix', render_matrix())

    def forward(self, ring_energy: torch.Tensor) -> torch.Tensor:
        """(батч, N_BINS) -> (батч, IMAGE_PX, IMAGE_PX)."""
        image = ring_energy @ self.matrix.T
        return image.view(-1, IMAGE_PX, IMAGE_PX)


def mlp(sizes: list[int]) -> nn.Sequential:
    """Полносвязная сеть с GELU между слоями (после последнего слоя активации нет)."""
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.GELU()]
    return nn.Sequential(*layers[:-1])


class PhysicsPsfNet(nn.Module):
    """Параметры -> энергия по кольцам (softmax) -> изображение ФРТ."""

    def __init__(self, phase: bool = True, hidden: int = 256, depth: int = 4):
        super().__init__()
        self.phase = phase
        self.body = mlp([n_features(phase)] + [hidden] * depth + [N_BINS + 1])
        self.render = RadialRenderer()

    def rings(self, x: torch.Tensor) -> torch.Tensor:
        """Энергия по кольцам (без последнего выхода - энергии за кадром)."""
        return torch.softmax(self.body(x), dim=1)[:, :N_BINS]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Признаки -> изображение ФРТ (батч, IMAGE_PX, IMAGE_PX)."""
        return self.render(self.rings(x))


class ConvPsfNet(nn.Module):
    """Параметры -> свёрточный декодер -> изображение ФРТ (без встроенной симметрии)."""

    def __init__(self, phase: bool = True, hidden: int = 256):
        super().__init__()
        self.phase = phase
        self.body = mlp([n_features(phase), hidden, hidden, 128 * 8 * 8])

        def up(cin, cout):
            return nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1),
                                 nn.GELU(), nn.Conv2d(cout, cout, 3, padding=1), nn.GELU())

        self.decoder = nn.Sequential(up(128, 64), up(64, 32), up(32, 16),
                                     nn.Conv2d(16, 1, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Признаки -> изображение ФРТ (батч, IMAGE_PX, IMAGE_PX)."""
        z = self.body(x).view(-1, 128, 8, 8)
        logits = self.decoder(z).view(-1, IMAGE_PX * IMAGE_PX)
        # Неотрицательность и нормировка на 1 (энергия в кадре)
        return torch.softmax(logits, dim=1).view(-1, IMAGE_PX, IMAGE_PX)
