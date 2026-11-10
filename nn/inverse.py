"""
Подбор высот рельефа через обученную сеть по ФРТ.

Критерий качества - доля энергии ФРТ в круге радиуса 50 мкм (EE50), чем больше,
тем лучше. Для заданной конфигурации (число линз, расстояние, фокусы)
высоты ищутся градиентным подъёмом по предсказанию сети (высоты непрерывны
внутри полосы порядка m = 7, много стартов одновременно). Найденные высоты
проверяются трассировкой и сравниваются:
- с исходными высотами H = mλ0/(n - 1);
- с высотами, оптимальными по длине фокального отрезка (параксиально);
- для двух линз - с полным перебором высот трассировкой (эталон).

Запуск: python -m nn.inverse --model physics   (или physics_plain)
Результаты: results/NIR_4_sem/inverse_design_<model>.csv, figures/inverse_*_<model>.png
"""

import argparse
import csv
import time
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from nn.dataset import init_worker, optimize_quick  # noqa: E402
from nn.metrics import EE_RADIUS_MM, encircled_energy  # noqa: E402
from nn.model import H_HIGH, H_LOW, MAX_LENSES, PhysicsPsfNet, encode  # noqa: E402
from nn.psf import FIELD_MM, IMAGE_PX, PsfResult, compute_psf  # noqa: E402
from nn.psf import encircled_energy as ee_numpy  # noqa: E402
from optimizers.multi_lens_study import (WEIGHT_FAMILIES, make_dataset,  # noqa: E402
                                         powers_for_total)
from optimizers.two_lens_study import design_height_um  # noqa: E402

RESULTS = Path('results') / 'NIR_4_sem'
N_STARTS = 256
STEPS = 300
TOP_K = 8          # сколько лучших по сети кандидатов проверять трассировкой


def load_model(name: str = 'physics') -> PhysicsPsfNet:
    """Загрузка обученной PhysicsPsfNet (physics - с фазовыми признаками, physics_plain - без)."""
    phase = name == 'physics'
    model = PhysicsPsfNet(phase=phase)
    model.load_state_dict(torch.load(RESULTS / 'models' / f'{name}.pt'))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def nn_optimize(model: PhysicsPsfNet, f_mm: Sequence[float], d_mm: float,
                seed: int = 0) -> list[tuple[list[float], float]]:
    """
    Градиентный подъём EE50 по высотам; высоты = sigmoid(u) внутри полосы.

    Возвращает различные (с точностью 0,01 мкм) решения из всех стартов,
    упорядоченные по убыванию предсказанной EE50: [(высоты, EE50), ...].
    """
    n = len(f_mm)
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(N_STARTS, n, generator=g, requires_grad=True)
    f_pad = torch.tensor([[float(f) for f in f_mm] + [0.0] * (MAX_LENSES - n)] * N_STARTS)
    n_t = torch.full((N_STARTS,), n)
    d_t = torch.full((N_STARTS,), float(d_mm))
    opt = torch.optim.Adam([u], lr=0.05)

    def heights(u):
        h = H_LOW + (H_HIGH - H_LOW) * torch.sigmoid(u)
        return torch.cat([h, torch.zeros(N_STARTS, MAX_LENSES - n)], dim=1)

    for _ in range(STEPS):
        ee = encircled_energy(model(encode(n_t, d_t, f_pad, heights(u), model.phase)),
                              EE_RADIUS_MM)
        loss = -ee.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        h = heights(u)
        ee = encircled_energy(model(encode(n_t, d_t, f_pad, h, model.phase)), EE_RADIUS_MM)
    out, seen = [], set()
    for k in torch.argsort(ee, descending=True).tolist():
        key = tuple(round(v, 2) for v in h[k, :n].tolist())
        if key not in seen:
            seen.add(key)
            out.append((h[k, :n].tolist(), float(ee[k])))
    return out


def nn_predict_ee(model: PhysicsPsfNet, f_mm: Sequence[float], d_mm: float,
                  h: Sequence[float]) -> float:
    """EE50, предсказанная сетью для заданных высот."""
    n = len(f_mm)
    x = encode(torch.tensor([n]), torch.tensor([float(d_mm)]),
               torch.tensor([[float(v) for v in f_mm] + [0.0] * (MAX_LENSES - n)]),
               torch.tensor([[float(v) for v in h] + [0.0] * (MAX_LENSES - n)]), model.phase)
    with torch.no_grad():
        return float(encircled_energy(model(x), EE_RADIUS_MM)[0])


def traced_ee(f_mm: Sequence[float], d_mm: float,
              h: Sequence[float]) -> tuple[float, PsfResult]:
    """EE50 и ФРТ по трассировке."""
    psf = compute_psf(make_dataset(f_mm, d_mm), list(h))
    return ee_numpy(psf.image, EE_RADIUS_MM), psf


def two_lens_grid(f_mm: Sequence[float], d_mm: float,
                  step: float = 0.04) -> tuple[list[float], float, int]:
    """Эталон для двух линз: полный перебор высот с трассировкой."""
    grid = np.arange(H_LOW, H_HIGH + 1e-9, step)
    best = (-1.0, None)
    for h1 in grid:
        for h2 in grid:
            ee, _ = traced_ee(f_mm, d_mm, [h1, h2])
            if ee > best[0]:
                best = (ee, [float(h1), float(h2)])
    return best[1], best[0], len(grid) ** 2


def cases() -> list[tuple[str, list[float], float]]:
    """Тестовые конфигурации: (название, фокусы в мм, расстояние в мм)."""
    out = []
    for n in (2, 3, 4, 5):
        out.append((f'N={n}, одинаковые, d=20', [100.0 * n] * n, 20.0))
        out.append((f'N={n}, поровну, Φ=const, d=50',
                    powers_for_total(WEIGHT_FAMILIES['equal'](n), 50.0), 50.0))
    return out


def run(name: str = 'physics', grid: bool = True) -> list[dict]:
    """Подбор высот для всех тестовых конфигураций, сохранение таблицы и рисунков."""
    torch.manual_seed(0)
    model = load_model(name)
    init_worker()
    rng = np.random.default_rng(0)
    rows, psfs = [], []
    for case, f_mm, d_mm in cases():
        n = len(f_mm)
        h_design = [design_height_um()] * n
        t = time.time()
        h_df, _ = optimize_quick(f_mm, d_mm, rng)
        t_df = time.time() - t
        t = time.time()
        candidates = nn_optimize(model, f_mm, d_mm)
        t_nn = time.time() - t
        h_nn, ee_nn_pred = candidates[0]
        t = time.time()
        checked = [(traced_ee(f_mm, d_mm, h)[0], h) for h, _ in candidates[:TOP_K]]
        ee_top, h_top = max(checked, key=lambda item: item[0])
        t_top = time.time() - t

        ee_design, _ = traced_ee(f_mm, d_mm, h_design)
        ee_df, psf_df = traced_ee(f_mm, d_mm, h_df)
        ee_nn, _ = traced_ee(f_mm, d_mm, h_nn)
        _, psf_top = traced_ee(f_mm, d_mm, h_top)
        row = {'case': case, 'n': n, 'd_mm': d_mm,
               'f_mm': ';'.join(f'{f:.1f}' for f in f_mm),
               'ee_design': ee_design, 'ee_df_opt': ee_df, 'ee_nn_opt': ee_nn,
               'ee_nn_predicted': ee_nn_pred, 'ee_nn_top': ee_top,
               'ee_df_opt_predicted': nn_predict_ee(model, f_mm, d_mm, h_df),
               'h_df_opt': ';'.join(f'{h:.2f}' for h in h_df),
               'h_nn_opt': ';'.join(f'{h:.2f}' for h in h_nn),
               'h_nn_top': ';'.join(f'{h:.2f}' for h in h_top),
               'time_df_s': round(t_df, 2), 'time_nn_s': round(t_nn, 2),
               'time_top_s': round(t_top, 2),
               'ee_grid': '', 'h_grid': '', 'time_grid_s': ''}
        if n == 2 and grid:
            t = time.time()
            h_grid, ee_grid, _ = two_lens_grid(f_mm, d_mm)
            row.update(ee_grid=ee_grid, h_grid=';'.join(f'{h:.2f}' for h in h_grid),
                       time_grid_s=round(time.time() - t, 1))
        rows.append(row)
        psfs.append((case, psf_df.image, psf_top.image, ee_df, ee_top))
        suffix = f", перебор {row['ee_grid']:.3f}" if row['ee_grid'] != '' else ''
        print(f"{case}: EE50 исходн. {ee_design:.3f}, по Δf {ee_df:.3f}, "
              f"сеть {ee_nn:.3f} (предсказано {ee_nn_pred:.3f}), "
              f"сеть + проверка {TOP_K} кандидатов {ee_top:.3f}"
              + suffix, flush=True)

    with open(RESULTS / f'inverse_design_{name}.csv', 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_bars(rows, RESULTS / 'figures' / f'inverse_ee50_{name}.png')
    plot_psfs(psfs, RESULTS / 'figures' / f'inverse_psf_{name}.png')
    return rows


def plot_bars(rows: list[dict], path: Path) -> None:
    """Столбчатая диаграмма EE50 (трассировка) для всех способов подбора высот."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(rows))
    w = 0.21
    for k, (key, label, color) in enumerate((
            ('ee_design', 'исходные высоты', '#9e9e9e'),
            ('ee_df_opt', 'оптимум по Δf (параксиально)', '#2a78d6'),
            ('ee_nn_opt', 'оптимум по ФРТ через сеть', '#f2a07b'),
            ('ee_nn_top', f'сеть + трассировка {TOP_K} кандидатов', '#d9480f'))):
        ax.bar(x + (k - 1.5) * w, [r[key] for r in rows], width=w * 0.92, color=color, label=label)
    ax.set_xticks(x, [r['case'].replace(', ', '\n') for r in rows], fontsize=7)
    ax.set_ylabel(f'Доля энергии в круге {EE_RADIUS_MM * 1e3:.0f} мкм (трассировка)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_psfs(psfs: list[tuple], path: Path) -> None:
    """ФРТ по трассировке: оптимум по Δf и подбор через сеть (случаи Φ = const)."""
    pick = [p for p in psfs if 'Φ=const' in p[0]]
    fig, axes = plt.subplots(2, len(pick), figsize=(3.2 * len(pick), 6.4))
    zoom, c = 8, IMAGE_PX // 2
    lo, hi = c - zoom, c + zoom
    ext = [-FIELD_MM * zoom / c * 1e3, FIELD_MM * zoom / c * 1e3] * 2
    for k, (name, img_df, img_nn, ee_df, ee_nn) in enumerate(pick):
        vmax = max(img_df.max(), img_nn.max())
        axes[0, k].imshow(img_df[lo:hi, lo:hi], cmap='inferno', vmax=vmax, extent=ext)
        axes[0, k].set_title(f'{name.split(",")[0]}: по Δf\nEE50 = {ee_df:.3f}', fontsize=9)
        axes[1, k].imshow(img_nn[lo:hi, lo:hi], cmap='inferno', vmax=vmax, extent=ext)
        axes[1, k].set_title(f'сеть + проверка {TOP_K}\nEE50 = {ee_nn:.3f}', fontsize=9)
        for ax in axes[:, k]:
            ax.set_xlabel('x, мкм', fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle('ФРТ (трассировка), сила системы постоянна, d = 50 мм', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Подбор высот из командной строки."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['physics', 'physics_plain'], default='physics')
    parser.add_argument('--no-grid', action='store_true', help='без перебора для двух линз')
    args = parser.parse_args()
    run(args.model, grid=not args.no_grid)


if __name__ == '__main__':
    main()
