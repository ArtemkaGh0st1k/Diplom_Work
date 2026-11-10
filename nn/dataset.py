"""
Генерация датасета «параметры системы -> ФРТ».

Для каждой случайной конфигурации (число линз, расстояние, распределение сил)
высоты рельефа оптимизируются по длине фокального отрезка (быстрый
покоординатный спуск из multi_lens_study), затем по трассировке считается ФРТ.
В датасет попадают два варианта каждой конфигурации:
- оптимизированные высоты (основная часть выборки - «оптимизированные системы»);
- случайные высоты из той же полосы, чтобы сеть видела и плохие системы
  (без них нельзя искать оптимум через сеть).

Запуск: python -m nn.dataset --configs 3000 --workers 6
Результат: nn/data/<name>.npz и results/NIR_4_sem/<name>_summary.csv
(обучение берёт все файлы nn/data/psf_dataset*.npz)
"""

import argparse
import csv
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from nn.psf import FIELD_MM, IMAGE_PX, compute_psf
from optimizers.multi_lens_study import (coordinate_descent, df_for_heights, make_dataset,
                                         order_table, system_power)
from optimizers.two_lens_study import H_GRID_UM, PHI_TOTAL, design_height_um, lambdas_m

MAX_LENSES = 5
D_MAX_MM = 60.0
MIN_ABS_FOCUS_MM = 30.0
N_STARTS = 4

DATA_DIR = Path('nn') / 'data'
RESULTS_DIR = Path('results') / 'NIR_4_sem'

_TABLE = None
_LAM = None


def init_worker() -> None:
    """Инициализация процесса: длины волн и таблица порядков (одна на процесс)."""
    global _TABLE, _LAM
    _LAM = lambdas_m()
    _TABLE = order_table(_LAM)


def sample_config(rng: np.random.Generator) -> tuple[list[float], float, str]:
    """Случайная конфигурация: (фокусы линз в мм, расстояние в мм, семейство)."""
    n = int(rng.integers(2, MAX_LENSES + 1))
    d = 0.0 if rng.random() < 0.15 else float(rng.uniform(0.0, D_MAX_MM))
    if rng.random() < 0.3:
        # Одинаковые линзы (как в задаче 1): при d = 0 фокус системы 100 мм
        return [100.0 * n] * n, d, 'identical'

    weights = rng.dirichlet(np.full(n, 2.0))
    family = 'split'
    if rng.random() < 0.1:
        # Одна из первых линз рассеивающая
        weights[int(rng.integers(0, n - 1))] = -0.25
        family = 'negative'
    weights = weights / weights.sum()

    # Силы первых n - 1 линз заданы, сила последней - из условия Φ_системы = Φ
    f_first = [1 / (w * PHI_TOTAL) for w in weights[:-1]]
    # Матрица первых n - 1 линз с промежутками и последним промежутком
    m = np.eye(2)
    for i, f0 in enumerate(f_first):
        if i:
            m = np.array([[1, d], [0, 1]]) @ m
        m = np.array([[1, 0], [-1 / f0, 1]]) @ m
    m = np.array([[1, d], [0, 1]]) @ m
    phi_last = (PHI_TOTAL + m[1, 0]) / m[0, 0]
    return f_first + [1 / phi_last], d, family


def valid(f_mm: list[float], d: float) -> bool:
    """Конфигурация допустима: фокусы не короче 30 мм и система собирающая."""
    return (all(abs(f) >= MIN_ABS_FOCUS_MM for f in f_mm)
            and system_power(f_mm, d) > 0)


def optimize_quick(f_mm: list[float], d: float,
                   rng: np.random.Generator) -> tuple[list[float], float]:
    """Покоординатный спуск по длине фокального отрезка из нескольких стартов."""
    n = len(f_mm)
    design = int(np.argmin(np.abs(H_GRID_UM - design_height_um())))
    starts = [[design] * n] + [list(map(int, rng.integers(0, len(H_GRID_UM), n)))
                               for _ in range(N_STARTS - 1)]
    best = None
    for idx in starts:
        df = df_for_heights(idx, f_mm, d, _TABLE)
        idx, df = coordinate_descent(list(idx), df, f_mm, d, _TABLE, rng)
        if best is None or df < best[1]:
            best = (idx, df)
    return [float(H_GRID_UM[g]) for g in best[0]], best[1]


def make_record(f_mm: list[float], d: float, family: str, heights: list[float],
                kind: str, df_par: float) -> dict:
    """Запись датасета: параметры системы, ФРТ по трассировке и её характеристики."""
    psf = compute_psf(make_dataset(f_mm, d), heights)
    f_sys = 1 / system_power(f_mm, d)
    return {
        'n': len(f_mm), 'd': d, 'family': family, 'kind': kind,
        'f': f_mm, 'h': heights, 'df_par_mm': df_par, 'f_sys_mm': f_sys,
        'image': psf.image, 'rms_mm': psf.rms_mm, 'z_best_mm': psf.z_best_mm,
        'energy_in_field': psf.energy_in_field, 'lost_rays': psf.lost_rays,
    }


def worker(seed: int) -> list[dict]:
    """Одна конфигурация: оптимизированный и случайный варианты (или [] при ошибке)."""
    rng = np.random.default_rng(seed)
    while True:
        f_mm, d, family = sample_config(rng)
        if valid(f_mm, d):
            break
    try:
        heights, df_opt = optimize_quick(f_mm, d, rng)
        h_rand = [float(h) for h in rng.choice(H_GRID_UM, len(f_mm))]
        df_rand = df_for_heights([int(np.argmin(np.abs(H_GRID_UM - h))) for h in h_rand],
                                 f_mm, d, _TABLE)
        return [make_record(f_mm, d, family, heights, 'optimized', df_opt),
                make_record(f_mm, d, family, h_rand, 'random', df_rand)]
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return []


def pad(values: list[float], fill: float = 0.0) -> list[float]:
    """Дополнение списка до MAX_LENSES элементов."""
    return list(values) + [fill] * (MAX_LENSES - len(values))


def save(records: list[dict], path_npz: Path, path_csv: Path) -> None:
    """Сохранение датасета в npz и краткой сводки (без изображений) в CSV."""
    path_npz.parent.mkdir(parents=True, exist_ok=True)
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path_npz,
        n=np.array([r['n'] for r in records], dtype=np.int64),
        d=np.array([r['d'] for r in records], dtype=np.float32),
        f=np.array([pad(r['f']) for r in records], dtype=np.float32),
        h=np.array([pad(r['h']) for r in records], dtype=np.float32),
        kind=np.array([r['kind'] for r in records]),
        family=np.array([r['family'] for r in records]),
        df_par_mm=np.array([r['df_par_mm'] for r in records], dtype=np.float32),
        f_sys_mm=np.array([r['f_sys_mm'] for r in records], dtype=np.float32),
        rms_mm=np.array([r['rms_mm'] for r in records], dtype=np.float32),
        z_best_mm=np.array([r['z_best_mm'] for r in records], dtype=np.float32),
        energy_in_field=np.array([r['energy_in_field'] for r in records], dtype=np.float32),
        image=np.stack([r['image'] for r in records]).astype(np.float32),
        field_mm=np.float32(FIELD_MM), image_px=np.int64(IMAGE_PX))
    with open(path_csv, 'w', newline='', encoding='utf-8') as fh:
        keys = ['n', 'd', 'family', 'kind', 'f', 'h', 'df_par_mm', 'f_sys_mm', 'rms_mm',
                'z_best_mm', 'energy_in_field', 'lost_rays']
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in records:
            row = {k: r[k] for k in keys}
            row['f'] = ';'.join(f'{v:.1f}' for v in r['f'])
            row['h'] = ';'.join(f'{v:.2f}' for v in r['h'])
            writer.writerow(row)


def main() -> None:
    """Генерация датасета из командной строки."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=int, default=3000)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--name', default='psf_dataset', help='имя файла в nn/data')
    args = parser.parse_args()

    start = time.time()
    seeds = np.random.SeedSequence(args.seed).generate_state(args.configs)
    records = []
    with Pool(args.workers, initializer=init_worker) as pool:
        for k, rows in enumerate(pool.imap_unordered(worker, seeds, chunksize=8), 1):
            records.extend(rows)
            if k % 250 == 0:
                print(f'{k}/{args.configs} конфигураций, {time.time() - start:.0f} с', flush=True)
    save(records, DATA_DIR / f'{args.name}.npz', RESULTS_DIR / f'{args.name}_summary.csv')
    print(f'Готово: {len(records)} систем за {time.time() - start:.0f} с')


if __name__ == '__main__':
    main()
