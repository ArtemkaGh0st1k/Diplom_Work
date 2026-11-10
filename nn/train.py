"""
Обучение сетей «параметры -> ФРТ» и оценка на тестовой выборке.

Разбиение на обучающую, валидационную и тестовую части (80/10/10) делается
по конфигурациям: оптимизированный и случайный варианты одной системы
всегда попадают в одну часть, чтобы не было утечки данных.

Запуск: python -m nn.train --model physics        (основная сеть)
        python -m nn.train --model physics_plain  (без фазовых признаков)
        python -m nn.train --model conv           (свёрточный декодер)
Результаты: results/NIR_4_sem/models, results/NIR_4_sem/metrics_<model>.json,
            results/NIR_4_sem/figures/
"""

import argparse
import json
import time
from typing import Optional
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402

from nn.metrics import correlation, encircled_energy, rms_radius, ssim  # noqa: E402
from nn.model import ConvPsfNet, PhysicsPsfNet, encode  # noqa: E402
from nn.psf import FIELD_MM, IMAGE_PX  # noqa: E402

DATA = Path('nn') / 'data' / 'psf_dataset.npz'
RESULTS = Path('results') / 'NIR_4_sem'
# Имя -> (класс, фазовые признаки на входе)
MODELS = {'physics': (PhysicsPsfNet, True),
          'physics_plain': (PhysicsPsfNet, False),
          'conv': (ConvPsfNet, True)}
IMAGE_SCALE = 1e4          # MSE на ~1e-4 значениях пикселей иначе слишком мал
SEED = 2026
MIN_ENERGY_IN_FIELD = 0.95   # системы, у которых пятно заметно выходит за кадр, исключаются


def load_data(paths: Optional[list[Path]] = None, phase: bool = True) -> tuple:
    """
    Загрузка и объединение датасетов: (признаки, ФРТ с суммой 1, группа
    конфигурации, метаданные). Системы с энергией в кадре меньше
    MIN_ENERGY_IN_FIELD отбрасываются.
    """
    paths = paths or sorted(DATA.parent.glob('psf_dataset*.npz'))
    parts = [np.load(p) for p in paths]
    names = [k for k in parts[0].files if parts[0][k].ndim]
    merged = {k: np.concatenate([p[k] for p in parts]) for k in names}
    keep = merged['energy_in_field'] >= MIN_ENERGY_IN_FIELD
    raw = {k: v[keep] for k, v in merged.items()}
    x = encode(raw['n'], raw['d'], raw['f'], raw['h'], phase=phase)
    y = raw['image'] / np.maximum(raw['image'].sum((1, 2), keepdims=True), 1e-12)
    # Конфигурация = (n, d, фокусы): её варианты должны быть в одной части
    keys = [f"{n}|{d:.6f}|{','.join(f'{v:.3f}' for v in f)}"
            for n, d, f in zip(raw['n'], raw['d'], raw['f'])]
    _, group = np.unique(keys, return_inverse=True)
    meta = {k: raw[k] for k in ('n', 'd', 'f', 'h', 'kind', 'family', 'df_par_mm',
                                'f_sys_mm', 'rms_mm', 'energy_in_field')}
    return torch.tensor(x), torch.tensor(y, dtype=torch.float32), group, meta


def split(group: np.ndarray, seed: int = SEED) -> list[np.ndarray]:
    """Индексы обучающей, валидационной и тестовой частей (80/10/10 по группам)."""
    rng = np.random.default_rng(seed)
    ids = rng.permutation(group.max() + 1)
    n = len(ids)
    part = np.empty(n, dtype=int)
    part[ids[:int(0.8 * n)]] = 0
    part[ids[int(0.8 * n):int(0.9 * n)]] = 1
    part[ids[int(0.9 * n):]] = 2
    p = part[group]
    return [np.where(p == k)[0] for k in range(3)]


def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE изображения + (1 - КК), как функция потерь на основе КК у Рымова (ур. 2.2)."""
    mse = ((pred - target) ** 2).flatten(1).mean(1) * IMAGE_SCALE
    return (mse + (1 - correlation(pred, target))).mean()


@torch.no_grad()
def evaluate(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
             batch: int = 512) -> dict[str, np.ndarray]:
    """Метрики по каждому образцу: КК, ИСС, EE50 и rms-радиус (эталон и сеть)."""
    model.eval()
    out = {'cc': [], 'ssim': [], 'ee_true': [], 'ee_pred': [], 'rms_true': [], 'rms_pred': []}
    for k in range(0, len(x), batch):
        p, t = model(x[k:k + batch]), y[k:k + batch]
        out['cc'].append(correlation(p, t))
        out['ssim'].append(ssim(t, p))
        out['ee_true'].append(encircled_energy(t))
        out['ee_pred'].append(encircled_energy(p))
        out['rms_true'].append(rms_radius(t))
        out['rms_pred'].append(rms_radius(p))
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def summarize(ev: dict[str, np.ndarray]) -> dict[str, float]:
    """Средние метрики и ранговая корреляция EE50 по выборке."""
    return {
        'cc_mean': float(ev['cc'].mean()), 'cc_median': float(np.median(ev['cc'])),
        'ssim_mean': float(ev['ssim'].mean()),
        'ee50_abs_err_mean': float(np.abs(ev['ee_pred'] - ev['ee_true']).mean()),
        'rms_rel_err_mean': float((np.abs(ev['rms_pred'] - ev['rms_true'])
                                   / ev['rms_true']).mean()),
        'ee50_rank_corr': float(np.corrcoef(np.argsort(np.argsort(ev['ee_true'])),
                                            np.argsort(np.argsort(ev['ee_pred'])))[0, 1]),
    }


def train(name: str, epochs: int, lr: float, batch: int, threads: int) -> dict:
    """Обучение сети, сохранение весов, метрик и рисунков; возвращает метрики."""
    torch.manual_seed(SEED)
    torch.set_num_threads(threads)
    cls, phase = MODELS[name]
    x, y, group, meta = load_data(phase=phase)
    tr, va, te = split(group)
    model = cls(phase=phase)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history = {'train': [], 'val': []}
    best, best_state = np.inf, None
    start = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.tensor(tr)[torch.randperm(len(tr))]
        total = 0.0
        for k in range(0, len(perm), batch):
            ids = perm[k:k + batch]
            loss = loss_fn(model(x[ids]), y[ids])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(ids)
        sched.step()
        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(x[va]), y[va]))
        history['train'].append(total / len(tr))
        history['val'].append(val)
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f'{name} эпоха {epoch:3d}: train {history["train"][-1]:.4f} '
                  f'val {val:.4f}  {time.time() - start:.0f} с', flush=True)
    model.load_state_dict(best_state)

    (RESULTS / 'models').mkdir(parents=True, exist_ok=True)
    (RESULTS / 'figures').mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), RESULTS / 'models' / f'{name}.pt')

    ev = evaluate(model, x[te], y[te])
    kinds = meta['kind'][te]
    metrics = {'model': name, 'params': sum(p.numel() for p in model.parameters()),
               'epochs': epochs, 'train_size': len(tr), 'val_size': len(va), 'test_size': len(te),
               'train_time_s': round(time.time() - start, 1),
               'test': summarize(ev),
               'test_optimized': summarize({k: v[kinds == 'optimized'] for k, v in ev.items()}),
               'test_random': summarize({k: v[kinds == 'random'] for k, v in ev.items()})}
    (RESULTS / f'metrics_{name}.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                                                  encoding='utf-8')
    plot_history(history, RESULTS / 'figures' / f'loss_{name}.png', name)
    plot_examples(model, x, y, te, ev, RESULTS / 'figures' / f'examples_{name}.png')
    print(json.dumps(metrics['test'], indent=2, ensure_ascii=False))
    return metrics


def plot_history(history: dict[str, list[float]], path: Path, name: str) -> None:
    """График функции потерь по эпохам."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history['train'], label='обучающая выборка')
    ax.plot(history['val'], label='валидационная выборка')
    ax.set_yscale('log')
    ax.set_xlabel('Эпоха')
    ax.set_ylabel('Функция потерь')
    ax.set_title(f'Обучение сети ({name})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def radial_profile(image: np.ndarray) -> np.ndarray:
    """Радиальный профиль: среднее центральной строки и столбца изображения."""
    c = image.shape[0] // 2
    return 0.5 * (image[c, c:] + image[c:, c])


def plot_examples(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
                  te: np.ndarray, ev: dict[str, np.ndarray], path: Path) -> None:
    """Эталон (трассировка) и предсказание: лучший, медианный и худшие по КК."""
    order = np.argsort(ev['cc'])
    pick = [order[-1], order[len(order) // 2], order[len(order) // 10], order[0]]
    labels = ['лучший', 'медианный', '10-й процентиль', 'худший']
    count = len(pick)
    with torch.no_grad():
        pred = model(x[te[pick]]).numpy()
    zoom, c = 12, IMAGE_PX // 2
    lo, hi = c - zoom, c + zoom
    half = FIELD_MM * zoom / c * 1e3
    extent = [-half, half, -half, half]
    pixel_um = 2 * FIELD_MM / IMAGE_PX * 1e3
    fig, axes = plt.subplots(3, count, figsize=(3.2 * count, 9.4))
    for k, (i, lab) in enumerate(zip(pick, labels)):
        t = y[te[i]].numpy()
        for row, img, title in ((0, t, f'{lab}: трассировка'),
                                (1, pred[k], f'сеть, КК = {ev["cc"][i]:.3f}')):
            axes[row, k].imshow(img[lo:hi, lo:hi], cmap='inferno', extent=extent,
                                norm=PowerNorm(gamma=0.4, vmin=0, vmax=t.max()))
            axes[row, k].set_title(title, fontsize=9)
            axes[row, k].set_xlabel('x, мкм', fontsize=8)
            axes[row, k].tick_params(labelsize=7)
        r = (np.arange(c) + 0.5) * pixel_um
        axes[2, k].plot(r, radial_profile(t), 'o-', ms=3, label='трассировка')
        axes[2, k].plot(r, radial_profile(pred[k]), 's--', ms=3, label='сеть')
        axes[2, k].set_xlim(0, half)
        axes[2, k].set_xlabel('r, мкм', fontsize=8)
        axes[2, k].tick_params(labelsize=7)
        axes[2, k].grid(True, alpha=0.3)
    axes[2, 0].set_ylabel('Доля энергии в пикселе', fontsize=8)
    axes[2, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Обучение из командной строки."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(MODELS), default='physics')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=2e-3)   # как у Рымова: Adam, шаг 0,002
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--threads', type=int, default=6)
    args = parser.parse_args()
    train(args.model, args.epochs, args.lr, args.batch, args.threads)


if __name__ == '__main__':
    main()
