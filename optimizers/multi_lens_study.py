"""
Исследование систем из N гармонических линз (обобщение two_lens_study на
3, 4 и 5 линз).

1. Расстояние d между соседними линзами (одинаковое): d = 0, затем
   увеличиваем; одинаковые линзы f0 = 100 * N мм (при d = 0 фокус 100 мм).
2а. d = 0, общая сила Φ = ΣΦi = 1/100 мм^-1, варьируется распределение
    сил по линзам (семейства весов w, Φi = wi * Φ).
2б. d != 0, общая сила системы постоянна: силы первых N - 1 линз задаются
    весами, сила последней подбирается из условия Φ_системы = Φ.

Полный перебор высот при N >= 3 слишком дорогой (107^N комбинаций),
поэтому используется покоординатный спуск по той же сетке высот с
многократными случайными стартами. Каждый шаг перебирает все высоты одной
линзы сразу (векторизовано). Лучшие решения затем полируются перебором по
парам линз (сетка G x G), что выводит из ловушек покоординатного спуска.
Для N = 2 и N = 3 результат сверяется с полным перебором (validate()),
для N = 4 и 5 - повторными запусками с разными seed.

Запуск: python -m optimizers.multi_lens_study
Результаты: results/NIR_3_sem/multi_lens_study/
"""

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from data.dataset_helper import DataSetHelper  # noqa: E402
from optimizers.base import BaseLensOptimizer  # noqa: E402
from optimizers.tracing import trace_focus_dict  # noqa: E402
from optimizers.two_lens_study import (DIAMETER_MM, H_GRID_UM, HARMONICA,  # noqa: E402
                                       LAMBDA_0_NM, LAMBDA_MAX_NM, LAMBDA_MIN_NM,
                                       N_RAYS, PHI_TOTAL, SUBSTRATE_MM,
                                       design_height_um, df_landscape, lambdas_m)
from utils.materials import n_bk7  # noqa: E402
from utils.unit import UnitType  # noqa: E402

N_STARTS = 48               # стартов на одну «степень свободы»: всего N_STARTS * (N - 1)
MAX_SWEEPS = 30
N_POLISH = 8                # сколько лучших решений полировать перебором по парам
SEED = 2026

OUT_DIR = Path('results') / 'NIR_3_sem' / 'multi_lens_study'

# Семейства распределения сил (веса до нормировки)
WEIGHT_FAMILIES = {
    'equal': lambda n: np.ones(n),
    'ramp_up': lambda n: np.arange(1, n + 1, dtype=float),
    'ramp_down': lambda n: np.arange(n, 0, -1, dtype=float),
    'one_strong': lambda n: np.r_[0.6, np.full(n - 1, 0.4 / (n - 1))],
    'one_negative': lambda n: np.r_[np.full(n - 1, 1.25 / (n - 1)), -0.25],
}


@dataclass
class CaseResult:
    task: str
    n_lens: int
    family: str
    d_mm: float
    f_mm: str                 # фокусы линз через ';'
    f_sys_mm: float
    df_design_mm: float
    h_opt_um: str             # оптимальные высоты через ';'
    df_opt_mm: float
    q_opt: float
    gain: float
    starts_hit_best: int      # сколько стартов пришли к лучшему значению
    df_check_base_mm: float
    df_trace_mean_mm: float
    q_trace_mean: float


# ----------------------------------------------------------------------
# Параксиальная модель: матрицы 2x2 как кортежи (A, B, C, D) массивов
# ----------------------------------------------------------------------

def mat_mul(m1, m2):
    a1, b1, c1, d1 = m1
    a2, b2, c2, d2 = m2
    return (a1 * a2 + b1 * c2, a1 * b2 + b1 * d2,
            c1 * a2 + d1 * c2, c1 * b2 + d1 * d2)


def refraction(power):
    return (1.0, 0.0, -power, 1.0)


def transfer(d_mm: float):
    return (1.0, d_mm, 0.0, 1.0)


IDENTITY = (1.0, 0.0, 0.0, 1.0)


def order_table(lam: np.ndarray) -> np.ndarray:
    """
    k(h, λ) * λ / (m * λ0) для всех высот сетки: оптическая сила линзы
    равна этой таблице, делённой на f0 (формула harmonic_lens_focus).
    """
    h = H_GRID_UM[:, None] * 1e-6
    alpha = h * (n_bk7(lam * 1e6) - 1) / lam
    k = np.floor(alpha + 0.5)
    k[k < 1] = np.nan
    return k * lam / (HARMONICA * LAMBDA_0_NM * 1e-9)


def df_for_heights(idx: list[int], f_mm: list[float], d_mm: float,
                   table: np.ndarray) -> float:
    """Δf, мм, для набора индексов высот."""
    m = IDENTITY
    for i, (g, f0) in enumerate(zip(idx, f_mm)):
        if i:
            m = mat_mul(transfer(d_mm), m)
        m = mat_mul(refraction(table[g] / f0), m)
    a, _, c, _ = m
    return float(np.ptp(-a / c))


def segment(idx: list[int], f_mm: list[float], d_mm: float, table: np.ndarray,
            lo: int, hi: int):
    """Матрица линз lo..hi-1 с промежутками между ними (без крайних промежутков)."""
    m = IDENTITY
    for i in range(lo, hi):
        if i > lo:
            m = mat_mul(transfer(d_mm), m)
        m = mat_mul(refraction(table[idx[i]] / f_mm[i]), m)
    return m


def coordinate_step(idx: list[int], j: int, f_mm: list[float], d_mm: float,
                    table: np.ndarray) -> np.ndarray:
    """Δf для всех высот линзы j при фиксированных остальных: массив по сетке."""
    n = len(idx)
    before = segment(idx, f_mm, d_mm, table, 0, j)
    if j:
        before = mat_mul(transfer(d_mm), before)
    after = segment(idx, f_mm, d_mm, table, j + 1, n)
    if j < n - 1:
        after = mat_mul(after, transfer(d_mm))

    lens = refraction(table / f_mm[j])            # (G, L)
    a, _, c, _ = mat_mul(after, mat_mul(lens, before))
    return np.ptp(-a / c, axis=1)


def pair_step(idx: list[int], j: int, k: int, f_mm: list[float], d_mm: float,
              table: np.ndarray) -> np.ndarray:
    """Δf для всех пар высот линз j < k при фиксированных остальных: матрица G x G."""
    n = len(idx)
    before = segment(idx, f_mm, d_mm, table, 0, j)
    if j:
        before = mat_mul(transfer(d_mm), before)
    between = transfer(d_mm)
    if k > j + 1:
        between = mat_mul(transfer(d_mm),
                          mat_mul(segment(idx, f_mm, d_mm, table, j + 1, k), between))
    after = segment(idx, f_mm, d_mm, table, k + 1, n)
    if k < n - 1:
        after = mat_mul(after, transfer(d_mm))

    lens_j = refraction(table[:, None, :] / f_mm[j])     # (G, 1, L)
    lens_k = refraction(table[None, :, :] / f_mm[k])     # (1, G, L)
    m = mat_mul(lens_k, mat_mul(between, mat_mul(lens_j, before)))
    a, _, c, _ = mat_mul(after, m)
    return np.ptp(-a / c, axis=2)


def coordinate_descent(idx: list[int], best: float, f_mm: list[float], d_mm: float,
                       table: np.ndarray, rng: np.random.Generator) -> tuple[list[int], float]:
    for _ in range(MAX_SWEEPS):
        improved = False
        for j in rng.permutation(len(idx)):
            values = coordinate_step(idx, j, f_mm, d_mm, table)
            g = int(np.nanargmin(values))
            if values[g] < best - 1e-12:
                best, idx[j], improved = float(values[g]), g, True
        if not improved:
            break
    return idx, best


def pair_polish(idx: list[int], best: float, f_mm: list[float], d_mm: float,
                table: np.ndarray, rng: np.random.Generator) -> tuple[list[int], float]:
    """Чередует перебор по парам линз и покоординатный спуск, пока есть улучшение."""
    n = len(idx)
    while True:
        improved = False
        for j in range(n):
            for k in range(j + 1, n):
                values = pair_step(idx, j, k, f_mm, d_mm, table)
                gj, gk = np.unravel_index(np.nanargmin(values), values.shape)
                if values[gj, gk] < best - 1e-12:
                    best, idx[j], idx[k], improved = float(values[gj, gk]), int(gj), int(gk), True
        idx, best = coordinate_descent(idx, best, f_mm, d_mm, table, rng)
        if not improved:
            return idx, best


def optimize_heights(f_mm: list[float], d_mm: float, lam: np.ndarray,
                     rng: np.random.Generator) -> tuple[list[int], float, int]:
    """
    Покоординатный спуск с многократными стартами и полировкой по парам линз.
    Возвращает (индексы высот, Δf, число отполированных решений, пришедших
    к лучшему).
    """

    table = order_table(lam)
    n = len(f_mm)
    design = int(np.argmin(np.abs(H_GRID_UM - design_height_um())))
    starts = [[design] * n] + [list(rng.integers(0, len(H_GRID_UM), n))
                               for _ in range(N_STARTS * (n - 1) - 1)]
    finals = []
    for idx in starts:
        idx = [int(g) for g in idx]
        best = df_for_heights(idx, f_mm, d_mm, table)
        finals.append(coordinate_descent(idx, best, f_mm, d_mm, table, rng)[::-1])

    # Лучшие различные решения дополнительно полируются перебором по парам линз
    finals.sort(key=lambda item: item[0])
    polished, seen = [], set()
    for best, idx in finals:
        if tuple(idx) in seen or len(seen) >= N_POLISH:
            continue
        seen.add(tuple(idx))
        polished.append(pair_polish(list(idx), best, f_mm, d_mm, table, rng)[::-1])

    best_df, best_idx = min(polished + finals, key=lambda item: item[0])
    hits = sum(1 for value, _ in polished if value <= best_df * (1 + 1e-9))
    return best_idx, best_df, hits


# ----------------------------------------------------------------------
# Постановка задач
# ----------------------------------------------------------------------

def system_power(f_mm: list[float], d_mm: float) -> float:
    m = IDENTITY
    for i, f0 in enumerate(f_mm):
        if i:
            m = mat_mul(transfer(d_mm), m)
        m = mat_mul(refraction(1 / f0), m)
    return -m[2]


def powers_for_total(weights: np.ndarray, d_mm: float) -> list[float]:
    """
    Фокусы линз: Φi = wi * Φ для первых N - 1, сила последней подбирается так,
    чтобы сила системы была Φ: Φ_N = (Φ + C') / A', где (A', C') - элементы
    матрицы первых N - 1 линз вместе с последним промежутком.
    """
    
    w = weights / weights.sum()
    f_first = [1 / (wi * PHI_TOTAL) for wi in w[:-1]]
    m = IDENTITY
    for i, f0 in enumerate(f_first):
        if i:
            m = mat_mul(transfer(d_mm), m)
        m = mat_mul(refraction(1 / f0), m)
    m = mat_mul(transfer(d_mm), m)
    a, _, c, _ = m
    return f_first + [1 / ((PHI_TOTAL + c) / a)]


def make_dataset(f_mm: list[float], d_mm: float) -> dict:
    n = len(f_mm)
    return DataSetHelper.create_dataset(
        count_linse=n,
        lower_lambda=LAMBDA_MIN_NM,
        upper_lambda=LAMBDA_MAX_NM,
        harmonica={i: HARMONICA for i in range(1, n + 1)},
        focus_0={i: float(f) for i, f in enumerate(f_mm, start=1)},
        lambda_0={i: LAMBDA_0_NM for i in range(1, n + 1)},
        distance={f'{i}-{i + 1}': d_mm for i in range(1, n)},
        refractive_area={f'{i}-{i + 1}': 1. for i in range(1, n)}
    )


def evaluate_case(task: str, family: str, f_mm: list[float], d_mm: float,
                  rng: np.random.Generator, with_trace: bool = True) -> CaseResult:
    lam = lambdas_m()
    table = order_table(lam)
    design = int(np.argmin(np.abs(H_GRID_UM - design_height_um())))
    df_design = df_for_heights([design] * len(f_mm), f_mm, d_mm, table)

    idx, df_opt, hits = optimize_heights(f_mm, d_mm, lam, rng)
    heights = [float(H_GRID_UM[g]) for g in idx]

    dataset = make_dataset(f_mm, d_mm)
    h_dict = {UnitType.MICROMETER: heights}
    base = BaseLensOptimizer().lmbd_focus_dict(dataset, h_dict, lam)
    df_base = float(np.ptp(list(base.values()))) * 1e3

    df_trace = float('nan')
    if with_trace:
        mean = trace_focus_dict(dataset, h_dict, lam, aperture='mean',
                                diameter_mm=DIAMETER_MM, substrate_mm=SUBSTRATE_MM,
                                n_rays=N_RAYS)
        values = np.array(list(mean.values()))
        df_trace = float(np.nanmax(values) - np.nanmin(values)) * 1e3

    f_sys = 1 / system_power(f_mm, d_mm)
    return CaseResult(task=task, n_lens=len(f_mm), family=family, d_mm=d_mm,
                      f_mm=';'.join(f'{f:.1f}' for f in f_mm), f_sys_mm=f_sys,
                      df_design_mm=df_design,
                      h_opt_um=';'.join(f'{h:.2f}' for h in heights),
                      df_opt_mm=df_opt, q_opt=df_opt / abs(f_sys),
                      gain=df_design / df_opt, starts_hit_best=hits,
                      df_check_base_mm=df_base, df_trace_mean_mm=df_trace,
                      q_trace_mean=df_trace / abs(f_sys))


DISTANCES_1 = (0, 1, 2, 5, 10, 20, 50, 100)
DISTANCES_2B = (0, 5, 10, 20, 50)
FAMILIES_2B = ('equal', 'ramp_up', 'ramp_down')


def task_1(n: int, rng) -> list[CaseResult]:
    return [evaluate_case('1', 'equal', [100.0 * n] * n, d, rng) for d in DISTANCES_1]


def task_2a(n: int, rng) -> list[CaseResult]:
    rows = []
    for family in ('equal', 'ramp_up', 'one_strong', 'one_negative'):
        f_mm = powers_for_total(WEIGHT_FAMILIES[family](n), 0.0)
        rows.append(evaluate_case('2a', family, f_mm, 0.0, rng))
    return rows


def task_2b(n: int, rng) -> list[CaseResult]:
    return [evaluate_case('2b', family,
                          powers_for_total(WEIGHT_FAMILIES[family](n), d), d, rng)
            for family in FAMILIES_2B for d in DISTANCES_2B]


# ----------------------------------------------------------------------
# Проверка оптимизатора и вывод
# ----------------------------------------------------------------------

def validate() -> None:
    """Сравнение покоординатного спуска с полным перебором для N = 2 и N = 3."""
    rng = np.random.default_rng(SEED)
    lam = lambdas_m()
    for d in (0.0, 10.0, 50.0):
        exhaustive = float(np.nanmin(df_landscape(200.0, 200.0, d, lam)))
        _, found, hits = optimize_heights([200.0, 200.0], d, lam, rng)
        print(f'N=2, d={d:>4}: перебор {exhaustive:.4f}, спуск {found:.4f}, '
              f'стартов в оптимуме {hits}')

    table = order_table(lam)
    f_mm, d = [300.0] * 3, 10.0
    best = np.inf
    for g1 in range(len(H_GRID_UM)):
        m = refraction(table[g1] / f_mm[0])
        m = mat_mul(transfer(d), m)
        m2 = mat_mul(refraction(table[:, None, :] / f_mm[1]), m)   # (G2, 1, L)
        m2 = mat_mul(transfer(d), m2)
        a, _, c, _ = mat_mul(refraction(table[None, :, :] / f_mm[2]), m2)
        best = min(best, float(np.nanmin(np.ptp(-a / c, axis=2))))
    _, found, hits = optimize_heights(f_mm, d, lam, rng)
    print(f'N=3, d=10: перебор {best:.4f}, спуск {found:.4f}, '
          f'стартов в оптимуме {hits}')


def save_csv(rows: list[CaseResult], path: Path) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def plot_task_1(rows: list[CaseResult], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in sorted({r.n_lens for r in rows}):
        part = [r for r in rows if r.n_lens == n]
        d = [r.d_mm for r in part]
        line, = ax.plot(d, [r.q_opt * 100 for r in part], 'o-', label=f'N = {n}, оптимум')
        ax.plot(d, [r.df_design_mm / abs(r.f_sys_mm) * 100 for r in part], '--',
                color=line.get_color(), alpha=0.6, label=f'N = {n}, исходные высоты')
        ax.plot(d, [r.q_trace_mean * 100 for r in part], 'x', color=line.get_color())
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xlim(left=0)
    ax.set_xlabel('Расстояние между соседними линзами d, мм')
    ax.set_ylabel('Относительный хроматизм Q = Δf/f, %')
    ax.set_title('Задача 1: одинаковые линзы f0 = 100·N мм (x — трассировка)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_task_2b(rows: list[CaseResult], path: Path) -> None:
    ns = sorted({r.n_lens for r in rows})
    fig, axes = plt.subplots(1, len(ns), figsize=(5 * len(ns), 4.5), sharey=True)
    for ax, n in zip(np.atleast_1d(axes), ns):
        for family in FAMILIES_2B:
            part = [r for r in rows if r.n_lens == n and r.family == family]
            ax.plot([r.d_mm for r in part], [r.q_opt * 100 for r in part], 'o-',
                    label=family)
        ax.set_title(f'N = {n}')
        ax.set_xlabel('d, мм')
        ax.grid(True, alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel('Q = Δf/f, %')
    np.atleast_1d(axes)[0].legend()
    fig.suptitle('Задача 2б: сила системы постоянна (1/100 мм⁻¹), разные распределения сил')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run(lens_counts: tuple[int, ...] = (3, 4, 5)) -> dict[str, list[CaseResult]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    results = {'task1_distance': [], 'task2a_power_split': [], 'task2b_power_distance': []}
    for n in lens_counts:
        for name, task in (('task1_distance', task_1),
                           ('task2a_power_split', task_2a),
                           ('task2b_power_distance', task_2b)):
            start = time.time()
            rows = task(n, rng)
            results[name].extend(rows)
            print(f'N={n} {name}: {len(rows)} вариантов за {time.time() - start:.1f} с',
                  flush=True)
            save_csv(results[name], OUT_DIR / f'{name}.csv')
    plot_task_1(results['task1_distance'], OUT_DIR / 'task1_distance.png')
    plot_task_2b(results['task2b_power_distance'], OUT_DIR / 'task2b_power_distance.png')
    return results


if __name__ == '__main__':
    validate()
    run()
