"""
Дополнительные рисунки для отчёта НИР: состав датасета и схема сети.

Запуск: python -m nn.report_figures
Результат: results/NIR_4_sem/figures/dataset_overview.png, network_scheme.png
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

DATA = Path('nn') / 'data'
FIGURES = Path('results') / 'NIR_4_sem' / 'figures'


def dataset_overview(path: Path) -> None:
    """Гистограммы rms-радиуса пятна и EE50 для оптимизированных и случайных систем."""
    parts = [np.load(p) for p in sorted(DATA.glob('psf_dataset*.npz'))]
    kind = np.concatenate([p['kind'] for p in parts])
    rms = np.concatenate([p['rms_mm'] for p in parts]) * 1e3
    n = np.concatenate([p['n'] for p in parts])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    bins = np.linspace(0, 250, 51)
    for k, label, color in (('optimized', 'оптимизированные', '#2a78d6'),
                            ('random', 'случайные', '#eb6834')):
        axes[0].hist(rms[kind == k], bins=bins, alpha=0.6, color=color, label=label)
    axes[0].set_xlabel('rms-радиус пятна, мкм')
    axes[0].set_ylabel('Число систем')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    for k, color in (('optimized', '#2a78d6'), ('random', '#eb6834')):
        med = [np.median(rms[(kind == k) & (n == m)]) for m in range(2, 6)]
        axes[1].plot(range(2, 6), med, 'o-', color=color,
                     label='оптимизированные' if k == 'optimized' else 'случайные')
    axes[1].set_xticks(range(2, 6))
    axes[1].set_xlabel('Число линз N')
    axes[1].set_ylabel('Медианный rms-радиус, мкм')
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def network_scheme(path: Path) -> None:
    """Блок-схема физически-информированной сети и подбора параметров."""
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(-0.2, 3.6)
    ax.axis('off')
    blocks = [
        (0.1, 'Параметры системы\nN, d, f₁…f₅, H₁…H₅\n+ фазовые признаки\n(90 чисел)', '#e8eef8'),
        (2.3, 'Полносвязная сеть\n4 слоя × 256\nGELU', '#e8eef8'),
        (4.5, 'softmax\n256 колец\n+ «за кадром»', '#fdebd9'),
        (6.7, 'Фиксированный\nоператор R\n(4096 × 256)', '#fdebd9'),
        (8.9, 'ФРТ\n64 × 64 пикселя', '#e3f1e3'),
    ]
    w, h, y = 1.9, 1.5, 1.6
    for x, text, color in blocks:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05',
                                    fc=color, ec='#444444'))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=9)
    for (x1, *_), (x2, *_) in zip(blocks[:-1], blocks[1:]):
        ax.add_patch(FancyArrowPatch((x1 + w + 0.05, y + h / 2), (x2 - 0.05, y + h / 2),
                                     arrowstyle='-|>', mutation_scale=14, color='#444444'))
    ax.text(5.5, 3.35, 'обучаемая часть: признаки → энергия по кольцам;  '
            'встроенная физика: симметрия и сохранение энергии', ha='center', fontsize=8.5)
    ax.add_patch(FancyArrowPatch((9.85, 1.55), (1.05, 1.55), connectionstyle='arc3,rad=-0.12',
                                 arrowstyle='-|>', mutation_scale=14, color='#d9480f', ls='--'))
    ax.text(5.5, -0.05, 'подбор параметров: градиент EE50 по высотам H (Adam, 256 стартов) '
            '→ проверка 8 лучших кандидатов трассировкой', ha='center', fontsize=8.5,
            color='#d9480f')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Построение рисунков."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    dataset_overview(FIGURES / 'dataset_overview.png')
    network_scheme(FIGURES / 'network_scheme.png')


if __name__ == '__main__':
    main()
