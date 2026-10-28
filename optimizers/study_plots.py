"""
Графики по исследованиям систем из 2–5 гармонических линз
(results/two_lens_study и results/multi_lens_study).

Читает только CSV, поэтому можно перезапускать без пересчёта:
python -m optimizers.study_plots
Рисунки: results/multi_lens_study/figures/
"""

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

TWO_DIR = Path('results') / 'two_lens_study'
MULTI_DIR = Path('results') / 'multi_lens_study'
FIG_DIR = MULTI_DIR / 'figures'

# Категориальная палитра (проверена на различимость при дальтонизме),
# цвет закреплён за числом линз; маркер - второе кодирование для печати
N_COLOR = {2: '#2a78d6', 3: '#eb6834', 4: '#1baf7a', 5: '#eda100'}
N_MARKER = {2: 'o', 3: 's', 4: '^', 5: 'D'}
FAMILY_COLOR = {'equal': '#2a78d6', 'ramp_up': '#eb6834', 'ramp_down': '#1baf7a',
                'one_strong': '#eda100', 'one_negative': '#6250d6'}
FAMILY_LABEL = {'equal': 'поровну', 'ramp_up': 'по возрастанию',
                'ramp_down': 'по убыванию', 'one_strong': 'одна сильная (60%)',
                'one_negative': 'одна рассеивающая'}
INK = '#0b0b0b'
INK_MUTED = '#52514e'
GRID = '#e4e3df'

plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': INK_MUTED, 'axes.labelcolor': INK,
    'xtick.color': INK_MUTED, 'ytick.color': INK_MUTED, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.spines.top': False,
    'axes.spines.right': False, 'lines.linewidth': 2, 'lines.markersize': 6,
    'legend.frameon': False, 'savefig.dpi': 220, 'savefig.bbox': 'tight',
})


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def num(row: dict, key: str) -> float:
    return float(row[key])


def load() -> dict[str, list[dict]]:
    """Объединяет данные двух исследований в единый формат строк."""
    task1 = []
    for r in read_csv(TWO_DIR / 'task1_distance.csv'):
        task1.append({'n': 2, 'd': num(r, 'd_mm'), 'q': num(r, 'q_opt'),
                      'q_design': num(r, 'df_design_mm') / num(r, 'f_sys_mm'),
                      'q_trace': num(r, 'q_trace_mean'), 'gain': num(r, 'gain'),
                      'h': [num(r, 'h1_opt_um'), num(r, 'h2_opt_um')]})
    for r in read_csv(MULTI_DIR / 'task1_distance.csv'):
        task1.append({'n': int(r['n_lens']), 'd': num(r, 'd_mm'), 'q': num(r, 'q_opt'),
                      'q_design': num(r, 'df_design_mm') / num(r, 'f_sys_mm'),
                      'q_trace': num(r, 'q_trace_mean'), 'gain': num(r, 'gain'),
                      'h': [float(h) for h in r['h_opt_um'].split(';')]})

    # Для двух линз семейства совпадают с долями s: equal = 0.5,
    # one_strong = 0.6, one_negative = 1.25 (ramp при s = 1/3 не считался)
    s_family = {0.5: 'equal', 0.6: 'one_strong', 1.25: 'one_negative'}
    task2a = [{'n': 2, 'family': s_family[num(r, 's')], 'q': num(r, 'q_opt'),
               'q_trace': num(r, 'q_trace_mean')}
              for r in read_csv(TWO_DIR / 'task2a_power_ratio.csv') if num(r, 's') in s_family]
    task2a += [{'n': int(r['n_lens']), 'family': r['family'], 'q': num(r, 'q_opt'),
                'q_trace': num(r, 'q_trace_mean')}
               for r in read_csv(MULTI_DIR / 'task2a_power_split.csv')]

    task2b = [{'n': 2, 'family': 'equal', 'd': num(r, 'd_mm'), 'q': num(r, 'q_opt'),
               'q_trace': num(r, 'q_trace_mean')}
              for r in read_csv(TWO_DIR / 'task2b_power_distance.csv') if num(r, 's') == 0.5]
    task2b += [{'n': int(r['n_lens']), 'family': r['family'], 'd': num(r, 'd_mm'),
                'q': num(r, 'q_opt'), 'q_trace': num(r, 'q_trace_mean')}
               for r in read_csv(MULTI_DIR / 'task2b_power_distance.csv')]
    return {'task1': task1, 'task2a': task2a, 'task2b': task2b}



def fig_task1_q(data, path):
    """Относительный хроматизм от расстояния для N = 2..5."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in sorted({r['n'] for r in data}):
        rows = sorted((r for r in data if r['n'] == n), key=lambda r: r['d'])
        d = [r['d'] for r in rows]
        q = [r['q'] * 100 for r in rows]
        ax.plot(d, q, marker=N_MARKER[n], color=N_COLOR[n], label=f'N = {n}')
        ax.plot(d, [r['q_trace'] * 100 for r in rows], linestyle='none',
                marker='x', ms=7, mew=1.5, color=INK_MUTED)
    ax.plot([], [], linestyle='none', marker='x', color=INK_MUTED, label='трассировка')
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xlim(0, 120)
    ax.set_xlabel('Расстояние между соседними линзами d, мм')
    ax.set_ylabel('Относительный хроматизм Q = Δf / f, %')
    ax.set_title('Оптимальные высоты: хроматизм от расстояния (одинаковые линзы, f = 100 мм при d = 0)',
                 fontsize=10, loc='left')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_task1_gain(data, path):
    """Во сколько раз оптимизация высот уменьшает Δf."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for n in sorted({r['n'] for r in data}):
        rows = sorted((r for r in data if r['n'] == n), key=lambda r: r['d'])
        d = [r['d'] for r in rows]
        gain = [r['gain'] for r in rows]
        ax.plot(d, gain, marker=N_MARKER[n], color=N_COLOR[n], label=f'N = {n}')
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xlim(0, 120)
    ax.set_ylim(bottom=1)
    ax.set_xlabel('Расстояние между соседними линзами d, мм')
    ax.set_ylabel('Δf исходн. / Δf оптим.')
    ax.set_title('Выигрыш от оптимизации высот относительно H = mλ0/(n−1)',
                 fontsize=10, loc='left')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_task1_heights(data, path):
    """
    Оптимальные высоты от расстояния: панель на каждое N.

    Линзы одинаковые, поэтому перестановка высот между ними почти не меняет Δf,
    и «какая линза какую высоту получила» скачет от точки к точке. Устойчив
    именно набор высот, поэтому высоты отсортированы по величине.
    """
    ns = sorted({r['n'] for r in data})
    fig, axes = plt.subplots(1, len(ns), figsize=(3.6 * len(ns), 4), sharey=True)
    lam0_um, n_glass = 0.55, 1.5185224
    unit = lam0_um / (n_glass - 1)
    for ax, n in zip(np.atleast_1d(axes), ns):
        rows = sorted((r for r in data if r['n'] == n), key=lambda r: r['d'])
        d = [r['d'] for r in rows]
        # Ранг высоты - порядковый признак: оттенки одного цвета от светлого к тёмному
        shades = plt.get_cmap('Blues')(np.linspace(0.45, 0.95, n))
        for i in range(n):
            ax.plot(d, [sorted(r['h'])[i] for r in rows], marker='o', ms=4,
                    color=shades[i], label=f'{i + 1}-я по величине')
        for bound in (6.5 * unit, 7 * unit, 7.5 * unit):
            ax.axhline(bound, color=INK_MUTED, lw=0.8,
                       ls='--' if bound == 7 * unit else ':')
        ax.set_xscale('symlog', linthresh=1)
        ax.set_xlim(0, 120)
        ax.set_title(f'N = {n}', fontsize=10)
        ax.set_xlabel('d, мм')
        ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2)
    np.atleast_1d(axes)[0].set_ylabel('Оптимальная высота рельефа, мкм')
    fig.suptitle('Набор оптимальных высот, отсортированных по величине '
                 '(пунктир — H = 7λ0/(n−1), точки — границы полосы порядка 7)',
                 fontsize=10, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_task2a(data, path):
    """Распределение сил при d = 0: сгруппированные полосы."""
    ns = sorted({r['n'] for r in data})
    families = [f for f in FAMILY_LABEL if any(r['family'] == f for r in data)]
    width = 0.8 / len(families)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for k, family in enumerate(families):
        xs, qs = [], []
        for i, n in enumerate(ns):
            row = next((r for r in data if r['n'] == n and r['family'] == family), None)
            if row is not None:
                xs.append(i - 0.4 + width * (k + 0.5))
                qs.append(row['q'] * 100)
        bars = ax.bar(xs, qs, width=width * 0.9, color=FAMILY_COLOR[family],
                      label=FAMILY_LABEL[family], edgecolor='white', linewidth=1)
        ax.bar_label(bars, fmt='%.1f', fontsize=7, padding=2, color=INK)
    ax.set_xticks(range(len(ns)), [f'N = {n}' for n in ns])
    ax.set_ylabel('Q = Δf / f, %')
    ax.set_title('d = 0, общая сила 1/100 мм⁻¹: как распределить силу по линзам',
                 fontsize=10, loc='left')
    ax.grid(axis='x', visible=False)
    ax.legend(ncol=3, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.1))
    fig.savefig(path)
    plt.close(fig)


def fig_task2b(data, path):
    """Постоянная сила системы, разнесение линз: панель на каждое N."""
    ns = sorted({r['n'] for r in data})
    fig, axes = plt.subplots(1, len(ns), figsize=(3.6 * len(ns), 4), sharey=True)
    for ax, n in zip(np.atleast_1d(axes), ns):
        for family in ('equal', 'ramp_up', 'ramp_down'):
            rows = sorted((r for r in data if r['n'] == n and r['family'] == family),
                          key=lambda r: r['d'])
            if not rows:
                continue
            ax.plot([r['d'] for r in rows], [r['q'] * 100 for r in rows], marker='o', ms=4,
                    color=FAMILY_COLOR[family], label=FAMILY_LABEL[family])
        ax.set_title(f'N = {n}', fontsize=10)
        ax.set_xlabel('d, мм')
    np.atleast_1d(axes)[0].set_ylabel('Q = Δf / f, %')
    np.atleast_1d(axes)[-1].legend(fontsize=8)
    fig.suptitle('Сила системы постоянна (1/100 мм⁻¹), линзы разнесены на d',
                 fontsize=10, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_best_by_n(data, path):
    """Лучший достигнутый Q для каждого числа линз."""
    ns = sorted({r['n'] for r in data['task1']} | {r['n'] for r in data['task2b']})
    fig, ax = plt.subplots(figsize=(7, 4.2))
    # Последний элемент - сдвиг подписи по вертикали: у задачи 2б подписи под точками
    for label, rows, color, marker, dy in (
            ('одинаковые линзы, лучшее d (задача 1)', data['task1'], '#2a78d6', 'o', 7),
            ('сила постоянна, лучшее d и распределение (задача 2б)', data['task2b'],
             '#eb6834', 's', -14),
            ('d = 0, исходные высоты', [dict(r, q=r['q_design']) for r in data['task1']
                                        if r['d'] == 0], INK_MUTED, 'x', 7)):
        best = [min((r['q'] for r in rows if r['n'] == n), default=np.nan) * 100 for n in ns]
        ax.plot(ns, best, marker=marker, color=color, label=label)
        for n, value in zip(ns, best):
            if not np.isnan(value):
                ax.annotate(f'{value:.1f}', (n, value), xytext=(0, dy),
                            textcoords='offset points', ha='center', fontsize=8, color=INK)
    ax.set_xticks(ns)
    ax.set_xlabel('Число линз N')
    ax.set_ylabel('Лучший Q = Δf / f, %')
    ax.set_ylim(0, 21)
    ax.set_title('Сколько даёт каждая добавленная линза', fontsize=10, loc='left')
    ax.legend(fontsize=8, loc='lower left')
    fig.savefig(path)
    plt.close(fig)


def run() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load()
    fig_task1_q(data['task1'], FIG_DIR / '1_q_vs_distance.png')
    fig_task1_gain(data['task1'], FIG_DIR / '2_gain_vs_distance.png')
    fig_task1_heights(data['task1'], FIG_DIR / '3_heights_vs_distance.png')
    fig_task2a(data['task2a'], FIG_DIR / '4_power_split_d0.png')
    fig_task2b(data['task2b'], FIG_DIR / '5_power_split_vs_distance.png')
    fig_best_by_n(data, FIG_DIR / '6_best_q_by_n.png')
    print(f'Рисунки сохранены в {FIG_DIR}')


if __name__ == '__main__':
    run()
