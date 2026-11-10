"""
Метрики сравнения ФРТ.

КК и ИСС - как в диссертации Д. А. Рымова (ур. 1.11, 1.12): коэффициент
корреляции и индекс структурного сходства (SSIM). Дополнительно -
физические характеристики ФРТ: доля энергии в круге радиуса R и
среднеквадратичный радиус пятна.
"""

import torch
import torch.nn.functional as F  # noqa: N812

from nn.psf import FIELD_MM, IMAGE_PX

EE_RADIUS_MM = 0.05

_PIXEL = 2 * FIELD_MM / IMAGE_PX
_CENTERS = (torch.arange(IMAGE_PX) + 0.5) * _PIXEL - FIELD_MM
_R2 = _CENTERS[None, :] ** 2 + _CENTERS[:, None] ** 2


def correlation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """КК по каждому изображению батча."""
    a = a.flatten(1) - a.flatten(1).mean(1, keepdim=True)
    b = b.flatten(1) - b.flatten(1).mean(1, keepdim=True)
    return (a * b).sum(1) / torch.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-20)


def ssim(a: torch.Tensor, b: torch.Tensor, window: int = 7,
         sigma: float = 1.5) -> torch.Tensor:
    """ИСС по каждому изображению; оба нормируются на максимум эталона a."""
    scale = a.flatten(1).max(1).values.clamp_min(1e-12)[:, None, None]
    a, b = (a / scale)[:, None], (b / scale)[:, None]
    g = torch.exp(-0.5 * ((torch.arange(window) - window // 2) / sigma) ** 2)
    g = (g[:, None] * g[None, :]) / g.sum() ** 2
    g = g[None, None].to(a)
    mu_a, mu_b = F.conv2d(a, g), F.conv2d(b, g)
    s_aa = F.conv2d(a * a, g) - mu_a ** 2
    s_bb = F.conv2d(b * b, g) - mu_b ** 2
    s_ab = F.conv2d(a * b, g) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    value = (((2 * mu_a * mu_b + c1) * (2 * s_ab + c2))
             / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2)))
    return value.flatten(1).mean(1)


def encircled_energy(image: torch.Tensor,
                     radius_mm: float = EE_RADIUS_MM) -> torch.Tensor:
    """Доля энергии в круге радиуса radius_mm (дифференцируема)."""
    return (image * (_R2.to(image) <= radius_mm ** 2)).flatten(1).sum(1)


def rms_radius(image: torch.Tensor) -> torch.Tensor:
    """Среднеквадратичный радиус пятна по изображению, мм."""
    e = image.flatten(1)
    return torch.sqrt((e * _R2.to(image).flatten()).sum(1) / e.sum(1).clamp_min(1e-12))
