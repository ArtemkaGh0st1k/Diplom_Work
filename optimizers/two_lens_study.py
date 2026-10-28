"""
Исследование системы из двух гармонических линз (направления научного
руководителя):

1. Расстояние d между линзами: d = 0, затем увеличиваем, при каждом d
   ищем оптимальные высоты рельефа h1, h2.
2а. Общая оптическая сила постоянна (d = 0, Φ = Φ1 + Φ2), меняется
    соотношение сил s = Φ1 / Φ.
2б. Общая сила постоянна при d != 0: Φ = Φ1 + Φ2 - d * Φ1 * Φ2,
    варьируются s и d.

Для каждого варианта: перебор высот по сетке (параксиальная модель base.py,
векторизованная), затем проверка найденных высот трассировкой лучей
(optimizers/tracing, режим grating).

Запуск: python -m optimizers.two_lens_study
Результаты: results/NIR_3_sem/two_lens_study/
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
from utils.materials import n_bk7  # noqa: E402
from utils.unit import UnitType  # noqa: E402

# Общие параметры исследования
LAMBDA_MIN_NM = 400          # видимый диапазон
LAMBDA_MAX_NM = 700
LAMBDA_COUNT = 301           # шаг 1 нм
LAMBDA_0_NM = 550
HARMONICA = 7
H_STEP_UM = 0.01


def harmonic_band_grid(m: float = HARMONICA) -> np.ndarray:
    """
    Сетка высот, при которых на λ0 работает порядок m:
    round(H (n(λ0) - 1) / λ0) = m, т.е. H в [(m - 0.5), (m + 0.5)) * λ0 / (n(λ0) - 1).

    Без этого ограничения перебор уходит к большим H: растёт рабочий порядок k
    и хроматизм падает (Q ~ 1/(2k + 1)), но фокус на λ0 перестаёт быть f0.
    """
    lam0_um = LAMBDA_0_NM * 1e-3
    unit = lam0_um / (n_bk7(lam0_um) - 1)
    return np.round(np.arange((m - 0.5) * unit, (m + 0.5) * unit, H_STEP_UM), 4)


H_GRID_UM = harmonic_band_grid()
PHI_TOTAL = 1 / 100.0        # мм^-1, общая сила для задач 2а и 2б (f = 100 мм)
DIAMETER_MM = 10.0
SUBSTRATE_MM = 0.01          # тонкая подложка - ближе к тонким линзам base.py
N_RAYS = 10                  # веер лучей как в C++ (N12Click)
NEAR_OPT_TOL = 0.01          # «почти оптимальные» точки: в пределах 1% от минимума

OUT_DIR = Path('results') / 'NIR_3_sem' / 'two_lens_study'


@dataclass
class CaseResult:
    task: str
    d_mm: float
    s: float
    f1_mm: float
    f2_mm: float
    f_sys_mm: float
    h1_design_um: float
    h2_design_um: float
    df_design_mm: float
    h1_opt_um: float
    h2_opt_um: float
    df_opt_mm: float
    q_opt: float
    gain: float
    near_opt_count: int
    df_check_base_mm: float
    df_trace_parax_mm: float
    df_trace_mean_mm: float
    q_trace_mean: float


def lambdas_m() -> np.ndarray:
    return np.linspace(LAMBDA_MIN_NM, LAMBDA_MAX_NM, LAMBDA_COUNT) * 1e-9


def design_height_um(m: float = HARMONICA) -> float:
    lam0 = LAMBDA_0_NM * 1e-3
    return m * lam0 / (n_bk7(lam0) - 1)


def focus_table(f0_mm: float, lam: np.ndarray, m: float = HARMONICA) -> np.ndarray:
    """
    f(h, λ) для всех высот сетки, мм: та же формула, что
    BaseLensOptimizer.harmonic_lens_focus, но сразу для массива высот.
    """
    h = H_GRID_UM[:, None] * 1e-6
    alpha = h * (n_bk7(lam * 1e6) - 1) / lam
    k = np.floor(alpha + 0.5)
    k[k < 1] = np.nan
    return (m * LAMBDA_0_NM * 1e-9) / (k * lam) * f0_mm


def df_landscape(f1_mm: float, f2_mm: float, d_mm: float,
                 lam: np.ndarray) -> np.ndarray:
    """
    Длина фокального отрезка (разброс заднего отрезка по λ), мм,
    для всех пар (h1, h2) сетки: матрица [i(h1), j(h2)].
    """
    inv_f1 = 1 / focus_table(f1_mm, lam)
    inv_f2 = 1 / focus_table(f2_mm, lam)
    result = np.empty((len(H_GRID_UM), len(H_GRID_UM)))
    for i, d1 in enumerate(inv_f1):
        power = d1 + inv_f2 - d_mm * d1 * inv_f2
        bfl = (1 - d_mm * d1) / power
        result[i] = np.ptp(bfl, axis=1)
    return result


def system_focus_mm(f1_mm: float, f2_mm: float, d_mm: float) -> float:
    """Фокус системы на λ0: 1 / (Φ1 + Φ2 - d Φ1 Φ2)."""
    return 1 / (1 / f1_mm + 1 / f2_mm - d_mm / (f1_mm * f2_mm))


def make_dataset(f1_mm: float, f2_mm: float, d_mm: float) -> dict:
    return DataSetHelper.create_dataset(
        count_linse=2,
        lower_lambda=LAMBDA_MIN_NM,
        upper_lambda=LAMBDA_MAX_NM,
        harmonica={1: HARMONICA, 2: HARMONICA},
        focus_0={1: f1_mm, 2: f2_mm},
        lambda_0={1: LAMBDA_0_NM, 2: LAMBDA_0_NM},
        distance={'1-2': d_mm}
    )


def evaluate_case(task: str, f1_mm: float, f2_mm: float, d_mm: float,
                  s: float = float('nan')) -> CaseResult:
    lam = lambdas_m()
    landscape = df_landscape(f1_mm, f2_mm, d_mm, lam)
    i, j = np.unravel_index(np.nanargmin(landscape), landscape.shape)
    h1, h2 = float(H_GRID_UM[i]), float(H_GRID_UM[j])
    df_opt = float(landscape[i, j])
    near_opt = int(np.sum(landscape <= df_opt * (1 + NEAR_OPT_TOL)))

    h_design = design_height_um()
    i_d = int(np.argmin(np.abs(H_GRID_UM - h_design)))
    df_design = float(landscape[i_d, i_d])

    dataset = make_dataset(f1_mm, f2_mm, d_mm)
    heights = {UnitType.MICROMETER: [h1, h2]}

    # Контроль: тот же оптимум через lmbd_focus_dict
    base = BaseLensOptimizer().lmbd_focus_dict(dataset, heights, lam)
    df_base = float(np.ptp(list(base.values()))) * 1e3

    trace_kwargs = dict(diameter_mm=DIAMETER_MM, substrate_mm=SUBSTRATE_MM, n_rays=N_RAYS)
    parax = trace_focus_dict(dataset, heights, lam, aperture='paraxial',
                             **{**trace_kwargs, 'n_rays': 100})
    mean = trace_focus_dict(dataset, heights, lam, aperture='mean', **trace_kwargs)
    df_parax = float(np.nanmax(list(parax.values())) - np.nanmin(list(parax.values()))) * 1e3
    df_mean = float(np.nanmax(list(mean.values())) - np.nanmin(list(mean.values()))) * 1e3

    f_sys = system_focus_mm(f1_mm, f2_mm, d_mm)
    return CaseResult(task=task, d_mm=d_mm, s=s, f1_mm=f1_mm, f2_mm=f2_mm,
                      f_sys_mm=f_sys,
                      h1_design_um=round(h_design, 4), h2_design_um=round(h_design, 4),
                      df_design_mm=df_design,
                      h1_opt_um=h1, h2_opt_um=h2, df_opt_mm=df_opt,
                      q_opt=df_opt / abs(f_sys), gain=df_design / df_opt,
                      near_opt_count=near_opt,
                      df_check_base_mm=df_base,
                      df_trace_parax_mm=df_parax, df_trace_mean_mm=df_mean,
                      q_trace_mean=df_mean / abs(f_sys))


def split_power(s: float, d_mm: float) -> tuple[float, float]:
    """Φ1 = s Φ, Φ2 из условия Φ1 + Φ2 - d Φ1 Φ2 = Φ. Возвращает фокусы, мм."""
    phi1 = s * PHI_TOTAL
    phi2 = (PHI_TOTAL - phi1) / (1 - d_mm * phi1)
    return 1 / phi1, 1 / phi2


def task_1() -> list[CaseResult]:
    return [evaluate_case('1', 200.0, 200.0, d)
            for d in (0, 1, 2, 5, 10, 20, 50, 100)]


def task_2a() -> list[CaseResult]:
    return [evaluate_case('2a', *split_power(s, 0.0), 0.0, s)
            for s in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.25, 1.5, 2.0)]


def task_2b() -> list[CaseResult]:
    return [evaluate_case('2b', *split_power(s, d), d, s)
            for s in (0.3, 0.5, 0.7, 1.5)
            for d in (0, 5, 10, 20, 50)]


def save_csv(rows: list[CaseResult], path: Path) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def plot_task_1(rows: list[CaseResult], path: Path) -> None:
    d = [r.d_mm for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(d, [r.h1_opt_um for r in rows], 'o-', label='h1*')
    ax1.plot(d, [r.h2_opt_um for r in rows], 's-', label='h2*')
    ax1.axhline(rows[0].h1_design_um, color='gray', ls='--', label='H = mλ0/(n-1)')
    ax1.set_xlabel('Расстояние между линзами d, мм')
    ax1.set_ylabel('Оптимальная высота рельефа, мкм')
    ax1.set_xscale('symlog', linthresh=1)
    ax1.set_xlim(left=0)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(d, [r.df_design_mm for r in rows], 'o--', label='исходные высоты')
    ax2.plot(d, [r.df_opt_mm for r in rows], 'o-', label='оптимум, параксиально')
    ax2.plot(d, [r.df_trace_mean_mm for r in rows], 'x:', label='оптимум, трассировка')
    ax2.set_xlabel('Расстояние между линзами d, мм')
    ax2.set_ylabel('Фокальный отрезок Δf, мм')
    ax2.set_xscale('symlog', linthresh=1)
    ax2.set_xlim(left=0)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig.suptitle('Задача 1: две линзы f0 = 200 мм, m = 7, варьируется d')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_task_2a(rows: list[CaseResult], path: Path) -> None:
    s = [r.s for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(s, [r.q_opt * 100 for r in rows], 'o-', label='оптимум, параксиально')
    ax1.plot(s, [r.q_trace_mean * 100 for r in rows], 's-', label='оптимум, трассировка')
    ax1.plot(s, [r.df_design_mm / abs(r.f_sys_mm) * 100 for r in rows], 'o--',
             label='исходные высоты')
    ax1.set_xlabel('Доля силы первой линзы s = Φ1/Φ')
    ax1.set_ylabel('Относительный хроматизм Q = Δf/f, %')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(s, [r.h1_opt_um for r in rows], 'o-', label='h1*')
    ax2.plot(s, [r.h2_opt_um for r in rows], 's-', label='h2*')
    ax2.set_xlabel('Доля силы первой линзы s = Φ1/Φ')
    ax2.set_ylabel('Оптимальная высота рельефа, мкм')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig.suptitle('Задача 2а: d = 0, Φ = Φ1 + Φ2 = 1/100 мм⁻¹, варьируется соотношение сил')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_task_2b(rows: list[CaseResult], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in sorted({r.s for r in rows}):
        part = [r for r in rows if r.s == s]
        ax.plot([r.d_mm for r in part], [r.q_opt * 100 for r in part], 'o-',
                label=f's = {s} (паракс.)')
        ax.plot([r.d_mm for r in part], [r.q_trace_mean * 100 for r in part], 'x:',
                color=ax.lines[-1].get_color(), label=f's = {s} (трасс.)')
    ax.set_xlabel('Расстояние между линзами d, мм')
    ax.set_ylabel('Относительный хроматизм Q = Δf/f, %')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title('Задача 2б: Φ1 + Φ2 - dΦ1Φ2 = 1/100 мм⁻¹')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run() -> dict[str, list[CaseResult]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = {'task1_distance': (task_1, plot_task_1),
             'task2a_power_ratio': (task_2a, plot_task_2a),
             'task2b_power_distance': (task_2b, plot_task_2b)}
    results = {}
    for name, (task, plot) in tasks.items():
        start = time.time()
        rows = task()
        save_csv(rows, OUT_DIR / f'{name}.csv')
        plot(rows, OUT_DIR / f'{name}.png')
        results[name] = rows
        print(f'{name}: {len(rows)} вариантов за {time.time() - start:.1f} с')
    return results


if __name__ == '__main__':
    run()
