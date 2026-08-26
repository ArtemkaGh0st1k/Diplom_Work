from __future__ import annotations

import ast
import json
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from data.dataset_helper import DataSetHelper  # noqa: E402
from optimizers.base import BaseLensOptimizer  # noqa: E402
from optimizers.two_lens import TwoLensOptimizer  # noqa: E402
from optimizers.three_lens import ThreeLensOptimizer  # noqa: E402
from utils.unit import UnitType, set_default_unit  # noqa: E402
from validators.input_validator import InputValidator  # noqa: E402


@dataclass(frozen=True)
class DatasetParseResult:
    dataset: dict[str, Any]
    warnings: list[str]


def dataset_to_pretty_json(dataset: dict[str, Any]) -> str:
    def default(o: Any):
        if isinstance(o, UnitType):
            return o.name
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    return json.dumps(dataset, ensure_ascii=False, indent=2, sort_keys=True, default=default)


def _unit_from_any(v: Any) -> UnitType:
    if isinstance(v, UnitType):
        return v
    if isinstance(v, str):
        name = v.strip()
        try:
            return UnitType[name]
        except KeyError as e:
            raise ValueError(f"Неизвестная единица измерения: {v!r}") from e
    raise TypeError(f"unit должен быть UnitType или str, получено: {type(v).__name__}")


def normalize_dataset(raw: dict[str, Any]) -> DatasetParseResult:
    """
    Принимает словарь датасета из GUI и приводит к формату, который ждут оптимизаторы.
    Поддерживает `unit` как:
      - реальные UnitType
      - строки: 'NANOMETER', 'MILLIMETER', ...
    """
    if not isinstance(raw, dict):
        raise TypeError("Датасет должен быть словарём")

    ds = dict(raw)
    warnings: list[str] = []

    if "unit" not in ds or ds["unit"] is None:
        ds["unit"] = set_default_unit(return_unit_type=True)
        warnings.append("Ключ 'unit' отсутствовал — подставлены единицы измерения по умолчанию.")
    else:
        unit_raw = ds["unit"]
        if not isinstance(unit_raw, dict):
            raise TypeError("dataset['unit'] должен быть dict")
        ds["unit"] = {k: _unit_from_any(v) for k, v in unit_raw.items()}

    # normalize int keys for per-lens dicts if user entered them as strings
    for k in ("refractive_index", "harmonica", "lambda_0", "focus_0"):
        if k in ds and isinstance(ds[k], dict):
            fixed: dict[int, Any] = {}
            for kk, vv in ds[k].items():
                if isinstance(kk, int):
                    fixed[kk] = vv
                elif isinstance(kk, str) and kk.strip().isdigit():
                    fixed[int(kk.strip())] = vv
                else:
                    fixed[kk] = vv
            ds[k] = fixed

    # normalize "1-2" keys: allow tuples (1,2) -> "1-2"
    for k in ("distance", "refractive_area"):
        if k in ds and isinstance(ds[k], dict):
            fixed2: dict[str, Any] = {}
            for kk, vv in ds[k].items():
                if isinstance(kk, str):
                    fixed2[kk] = vv
                elif isinstance(kk, tuple) and len(kk) == 2:
                    fixed2[f"{kk[0]}-{kk[1]}"] = vv
                else:
                    fixed2[str(kk)] = vv
            ds[k] = fixed2

    InputValidator.validate_input_dataset(ds)  # may raise
    return DatasetParseResult(dataset=ds, warnings=warnings)


def parse_dataset_text(text: str) -> dict[str, Any]:
    """
    Принимает текст из редактора.
    Поддерживаем форматы:
      - Python dict (ast.literal_eval)
      - JSON
    """
    t = text.strip()
    if not t:
        raise ValueError("Пустой ввод датасета")

    # JSON first if it looks like JSON
    if t.startswith("{") and '"' in t:
        try:
            return json.loads(t)
        except Exception:
            pass

    try:
        obj = ast.literal_eval(t)
    except Exception as e:
        raise ValueError("Не удалось разобрать датасет. Введите Python-словарь или JSON.") from e

    if not isinstance(obj, dict):
        raise ValueError("Ввод должен быть словарём")
    return obj


class App(ttk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.master = master

        self._status_var = tk.StringVar(value="Готово")
        self._dataset_mode = tk.StringVar(value="default")
        self._dep_height_vars: list[tk.StringVar] = []
        self._opt2_h_vars: list[tk.StringVar] = []
        self._opt3_h_vars: list[tk.StringVar] = []
        self._opt4_h_vars: list[tk.StringVar] = []
        self._active_default_count = tk.IntVar(value=2)
        self._last_log_line: str = ""

        self._build_ui()

    def _build_ui(self) -> None:
        self.master.title("Оптимизация гармонических линз")
        self.master.minsize(980, 680)

        root = self.master
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        self.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self)
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        top.columnconfigure(3, weight=1)

        ttk.Label(top, text="Источник датасета:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            top, text="По умолчанию", variable=self._dataset_mode, value="default", command=self._sync_dataset_editor
        ).grid(row=0, column=1, sticky="w", padx=(6, 0))
        ttk.Radiobutton(
            top, text="Вручную", variable=self._dataset_mode, value="manual", command=self._sync_dataset_editor
        ).grid(row=0, column=2, sticky="w", padx=(6, 0))

        self._fill_default_btn = ttk.Button(top, text="Заполнить по умолчанию", command=self._fill_default_dataset)
        self._fill_default_btn.grid(row=0, column=4, sticky="e")

        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        left = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left, weight=2)
        main.add(right, weight=3)

        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)

        ttk.Label(left, text="Датасет (Python dict или JSON):").grid(row=0, column=0, sticky="w")
        self._dataset_text = ScrolledText(left, height=18, wrap=tk.NONE)
        self._dataset_text.grid(row=1, column=0, sticky="nsew", pady=(6, 10))

        ttk.Label(left, text="Лог / результаты:").grid(row=2, column=0, sticky="w")
        btns = ttk.Frame(left)
        btns.grid(row=2, column=0, sticky="e")
        ttk.Button(btns, text="Копировать последний лог", command=self._copy_last_log_to_clipboard).grid(row=0, column=0, sticky="e")
        ttk.Button(btns, text="Копировать лог", command=self._copy_log_to_clipboard).grid(row=0, column=1, sticky="e", padx=(8, 0))
        self._log = ScrolledText(left, height=12, wrap=tk.WORD, state="disabled")
        self._log.grid(row=3, column=0, sticky="nsew", pady=(6, 0))

        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self._nb = ttk.Notebook(right)
        self._nb.grid(row=0, column=0, sticky="nsew")

        self._tab_dep = ttk.Frame(self._nb)
        self._tab_loss2 = ttk.Frame(self._nb)
        self._tab_opt2 = ttk.Frame(self._nb)
        self._tab_opt3 = ttk.Frame(self._nb)
        self._tab_opt4 = ttk.Frame(self._nb)
        self._nb.add(self._tab_dep, text="Зависимость f(λ)")
        self._nb.add(self._tab_loss2, text="Ландшафт потерь (2 линзы)")
        self._nb.add(self._tab_opt2, text="Оптимизация 2 линзы")
        self._nb.add(self._tab_opt3, text="Оптимизация 3 линзы")
        self._nb.add(self._tab_opt4, text="Оптимизация 4 линзы")

        self._build_dep_tab(self._tab_dep)
        self._build_loss2_tab(self._tab_loss2)
        self._build_opt2_tab(self._tab_opt2)
        self._build_opt3_tab(self._tab_opt3)
        self._build_opt4_tab(self._tab_opt4)

        status = ttk.Label(self, textvariable=self._status_var, anchor="w")
        status.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 8))

        self._fill_default_dataset()
        self._sync_dataset_editor()
        self._nb.bind("<<NotebookTabChanged>>", lambda _e: self._sync_default_dataset_target())
        self._sync_default_dataset_target()

    def _build_dep_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=0)

        frm = ttk.Frame(parent)
        frm.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        frm.columnconfigure(9, weight=1)

        ttk.Label(frm, text="Кол-во линз:").grid(row=0, column=0, sticky="w")
        self._dep_count = tk.IntVar(value=1)
        ttk.Spinbox(frm, from_=1, to=4, textvariable=self._dep_count, width=5).grid(row=0, column=1, sticky="w", padx=(6, 14))

        self._dep_heights_frame = ttk.LabelFrame(frm, text="Высоты (мкм)")
        self._dep_heights_frame.grid(row=0, column=2, columnspan=2, sticky="w", padx=(0, 14))

        ttk.Label(frm, text="Лямбда шагов:").grid(row=0, column=4, sticky="w")
        self._dep_steps = tk.IntVar(value=601)
        ttk.Spinbox(frm, from_=51, to=5001, increment=50, textvariable=self._dep_steps, width=8).grid(row=0, column=5, sticky="w", padx=(6, 14))

        ttk.Button(frm, text="Рассчитать и построить", command=self._run_dep).grid(row=0, column=6, sticky="w")
        ttk.Label(parent, text="График строится в отдельном окне matplotlib (plt.show()).").grid(
            row=2, column=0, sticky="w", padx=10, pady=(0, 10)
        )

        self._dep_count.trace_add("write", lambda *_: self._rebuild_dep_height_entries())
        self._rebuild_dep_height_entries()
        self._dep_count.trace_add("write", lambda *_: self._sync_default_dataset_target())

    def _rebuild_dep_height_entries(self) -> None:
        for w in self._dep_heights_frame.winfo_children():
            w.destroy()

        count = int(self._dep_count.get())
        while len(self._dep_height_vars) < count:
            self._dep_height_vars.append(tk.StringVar(value=""))
        self._dep_height_vars = self._dep_height_vars[:count]

        for i in range(count):
            ttk.Label(self._dep_heights_frame, text=f"h{i+1}:").grid(row=0, column=2 * i, sticky="w", padx=(6 if i > 0 else 0, 4), pady=2)
            ttk.Entry(self._dep_heights_frame, textvariable=self._dep_height_vars[i], width=7).grid(
                row=0, column=2 * i + 1, sticky="w", padx=(0, 6), pady=2
            )

    def _build_loss2_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        frm = ttk.Frame(parent)
        frm.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        frm.columnconfigure(10, weight=1)

        ttk.Label(frm, text="h_min (мкм):").grid(row=0, column=0, sticky="w")
        self._loss2_hmin = tk.DoubleVar(value=5.0)
        ttk.Entry(frm, textvariable=self._loss2_hmin, width=8).grid(row=0, column=1, sticky="w", padx=(6, 14))

        ttk.Label(frm, text="h_max (мкм):").grid(row=0, column=2, sticky="w")
        self._loss2_hmax = tk.DoubleVar(value=10.0)
        ttk.Entry(frm, textvariable=self._loss2_hmax, width=8).grid(row=0, column=3, sticky="w", padx=(6, 14))

        ttk.Label(frm, text="Точек:").grid(row=0, column=4, sticky="w")
        self._loss2_points = tk.IntVar(value=50)
        ttk.Spinbox(frm, from_=10, to=300, textvariable=self._loss2_points, width=6).grid(row=0, column=5, sticky="w", padx=(6, 14))

        ttk.Button(frm, text="Построить ландшафт потерь", command=self._run_loss2).grid(row=0, column=10, sticky="e")

        ttk.Label(
            parent,
            text="Примечание: откроется 3D-график matplotlib (и сохранится картинка в `results/loss_landscape.png`).",
        ).grid(row=1, column=0, sticky="w", padx=10)

    def _build_opt2_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        frm = ttk.Frame(parent)
        frm.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        for c in range(10):
            frm.columnconfigure(c, weight=0)
        frm.columnconfigure(9, weight=1)

        self._opt2_h_frame = ttk.LabelFrame(frm, text="init_h (мкм)")
        self._opt2_h_frame.grid(row=0, column=0, columnspan=2, sticky="w", padx=(0, 14))
        self._opt2_h_vars = [tk.StringVar(value="7"), tk.StringVar(value="7")]
        ttk.Label(self._opt2_h_frame, text="h1:").grid(row=0, column=0, sticky="w")
        ttk.Entry(self._opt2_h_frame, textvariable=self._opt2_h_vars[0], width=7).grid(row=0, column=1, sticky="w", padx=(4, 8))
        ttk.Label(self._opt2_h_frame, text="h2:").grid(row=0, column=2, sticky="w")
        ttk.Entry(self._opt2_h_frame, textvariable=self._opt2_h_vars[1], width=7).grid(row=0, column=3, sticky="w", padx=(4, 0))

        ttk.Label(frm, text="hbounds (мкм):").grid(row=0, column=2, sticky="w")
        self._opt2_bmin = tk.DoubleVar(value=5.0)
        self._opt2_bmax = tk.DoubleVar(value=12.0)
        bfrm = ttk.Frame(frm)
        bfrm.grid(row=0, column=3, sticky="w", padx=(6, 14))
        ttk.Entry(bfrm, textvariable=self._opt2_bmin, width=7).grid(row=0, column=0, sticky="w")
        ttk.Label(bfrm, text="..").grid(row=0, column=1, sticky="w", padx=4)
        ttk.Entry(bfrm, textvariable=self._opt2_bmax, width=7).grid(row=0, column=2, sticky="w")

        ttk.Label(frm, text="m1, m2:").grid(row=0, column=4, sticky="w")
        self._opt2_m1 = tk.DoubleVar(value=7.0)
        self._opt2_m2 = tk.DoubleVar(value=7.0)
        mfrm = ttk.Frame(frm)
        mfrm.grid(row=0, column=5, sticky="w", padx=(6, 14))
        ttk.Entry(mfrm, textvariable=self._opt2_m1, width=6).grid(row=0, column=0, sticky="w")
        ttk.Label(mfrm, text=",").grid(row=0, column=1, sticky="w", padx=4)
        ttk.Entry(mfrm, textvariable=self._opt2_m2, width=6).grid(row=0, column=2, sticky="w")

        ttk.Label(frm, text="Оптимизировать:").grid(row=0, column=6, sticky="w")
        self._opt2_target = tk.StringVar(value="h2")
        ttk.Combobox(frm, values=["h1", "h2"], state="readonly", textvariable=self._opt2_target, width=4).grid(
            row=0, column=7, sticky="w", padx=(6, 14)
        )

        self._opt2_check_nei = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="Показать соседний минимум", variable=self._opt2_check_nei).grid(row=0, column=8, sticky="w")

        ttk.Button(frm, text="Запустить оптимизацию", command=self._run_opt2).grid(row=0, column=9, sticky="e")

        frm2 = ttk.Frame(parent)
        frm2.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        frm2.columnconfigure(3, weight=1)
        ttk.Label(frm2, text="Δf для текущих h1/h2:").grid(row=0, column=0, sticky="w")
        ttk.Button(frm2, text="Посчитать (аналит.)", command=self._run_opt2_analytic).grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Button(frm2, text="Посчитать (матр.)", command=self._run_opt2_matrix).grid(row=0, column=2, sticky="w", padx=(8, 0))

        ttk.Label(
            parent,
            text="Примечание: оптимизация 2 линз открывает графики через matplotlib (как в `main.py`).",
        ).grid(row=2, column=0, sticky="w", padx=10)

    def _build_opt3_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        frm = ttk.Frame(parent)
        frm.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        frm.columnconfigure(10, weight=1)

        self._opt3_h_frame = ttk.LabelFrame(frm, text="init_h (мкм)")
        self._opt3_h_frame.grid(row=0, column=0, columnspan=2, sticky="w", padx=(0, 14))
        self._opt3_h_vars = [tk.StringVar(value="7"), tk.StringVar(value="7"), tk.StringVar(value="7")]
        for i, v in enumerate(self._opt3_h_vars):
            ttk.Label(self._opt3_h_frame, text=f"h{i+1}:").grid(row=0, column=2 * i, sticky="w", padx=(0 if i == 0 else 8, 4))
            ttk.Entry(self._opt3_h_frame, textvariable=v, width=7).grid(row=0, column=2 * i + 1, sticky="w")

        ttk.Label(frm, text="hbounds (мкм):").grid(row=0, column=2, sticky="w")
        self._opt3_bmin = tk.DoubleVar(value=5.0)
        self._opt3_bmax = tk.DoubleVar(value=20.0)
        bfrm = ttk.Frame(frm)
        bfrm.grid(row=0, column=3, sticky="w", padx=(6, 14))
        ttk.Entry(bfrm, textvariable=self._opt3_bmin, width=7).grid(row=0, column=0, sticky="w")
        ttk.Label(bfrm, text="..").grid(row=0, column=1, sticky="w", padx=4)
        ttk.Entry(bfrm, textvariable=self._opt3_bmax, width=7).grid(row=0, column=2, sticky="w")

        ttk.Label(frm, text="Фиксировать:").grid(row=0, column=4, sticky="w")
        self._opt3_static = tk.StringVar(value="h1")
        ttk.Combobox(frm, values=["h1", "h2", "h3"], state="readonly", textvariable=self._opt3_static, width=4).grid(
            row=0, column=5, sticky="w", padx=(6, 14)
        )

        ttk.Button(frm, text="Запустить оптимизацию", command=self._run_opt3).grid(row=0, column=10, sticky="e")

        ttk.Label(
            parent,
            text="Примечание: оптимизация 3 линз открывает тепловую карту через matplotlib (как в вашем оптимизаторе).",
        ).grid(row=1, column=0, sticky="w", padx=10)

    def _build_opt4_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        frm = ttk.Frame(parent)
        frm.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        frm.columnconfigure(10, weight=1)

        self._opt4_h_frame = ttk.LabelFrame(frm, text="init_h (мкм)")
        self._opt4_h_frame.grid(row=0, column=0, columnspan=2, sticky="w", padx=(0, 14))
        self._opt4_h_vars = [tk.StringVar(value="7"), tk.StringVar(value="7"), tk.StringVar(value="7"), tk.StringVar(value="7")]
        for i, v in enumerate(self._opt4_h_vars):
            ttk.Label(self._opt4_h_frame, text=f"h{i+1}:").grid(row=0, column=2 * i, sticky="w", padx=(0 if i == 0 else 8, 4))
            ttk.Entry(self._opt4_h_frame, textvariable=v, width=7).grid(row=0, column=2 * i + 1, sticky="w")

        ttk.Label(frm, text="hbounds (мкм):").grid(row=0, column=2, sticky="w")
        self._opt4_bmin = tk.DoubleVar(value=5.0)
        self._opt4_bmax = tk.DoubleVar(value=15.0)
        bfrm = ttk.Frame(frm)
        bfrm.grid(row=0, column=3, sticky="w", padx=(6, 14))
        ttk.Entry(bfrm, textvariable=self._opt4_bmin, width=7).grid(row=0, column=0, sticky="w")
        ttk.Label(bfrm, text="..").grid(row=0, column=1, sticky="w", padx=4)
        ttk.Entry(bfrm, textvariable=self._opt4_bmax, width=7).grid(row=0, column=2, sticky="w")

        ttk.Label(frm, text="Фиксировать:").grid(row=0, column=4, sticky="w")
        self._opt4_static = tk.StringVar(value="h1")
        ttk.Combobox(frm, values=["h1", "h2", "h3", "h4"], state="readonly", textvariable=self._opt4_static, width=4).grid(
            row=0, column=5, sticky="w", padx=(6, 14)
        )

        ttk.Button(frm, text="Запустить оптимизацию", command=self._run_opt4).grid(row=0, column=10, sticky="e")

        ttk.Label(
            parent,
            text="Примечание: оптимизация 4 линз строит 3D Volume через plotly (как в `FourLensOptimizer`).",
        ).grid(row=1, column=0, sticky="w", padx=10)

    def _log_write(self, msg: str) -> None:
        self._last_log_line = msg.rstrip()
        self._log.configure(state="normal")
        self._log.insert("end", msg.rstrip() + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _copy_log_to_clipboard(self) -> None:
        text = self._log.get("1.0", "end").strip()
        self.master.clipboard_clear()
        self.master.clipboard_append(text)
        self._set_status("Лог скопирован в буфер обмена")

    def _copy_last_log_to_clipboard(self) -> None:
        self.master.clipboard_clear()
        self.master.clipboard_append(self._last_log_line)
        self._set_status("Последняя строка лога скопирована")

    def _sync_default_dataset_target(self) -> None:
        """
        Определяет, сколько линз нужно для кнопки "Заполнить по умолчанию":
        - на вкладке f(λ): берём значение "Кол-во линз"
        - на вкладках оптимизации: фиксированное число (2/3/4)
        """
        try:
            tab = self._nb.select()
            widget = self.master.nametowidget(tab)
        except Exception:
            return

        if widget is self._tab_dep:
            self._active_default_count.set(int(self._dep_count.get()))
        elif widget is self._tab_opt2 or widget is self._tab_loss2:
            self._active_default_count.set(2)
        elif widget is self._tab_opt3:
            self._active_default_count.set(3)
        elif widget is self._tab_opt4:
            self._active_default_count.set(4)
        else:
            self._active_default_count.set(2)

    def _set_status(self, msg: str) -> None:
        self._status_var.set(msg)
        self.master.update_idletasks()

    def _fill_default_dataset(self) -> None:
        count = int(self._active_default_count.get())
        ds = DataSetHelper.create_default_dataset(count_linse=count)
        prev_state = str(self._dataset_text.cget("state"))
        try:
            if prev_state != "normal":
                self._dataset_text.configure(state="normal")
            self._dataset_text.delete("1.0", "end")
            self._dataset_text.insert("1.0", dataset_to_pretty_json(ds))
        finally:
            if prev_state != "normal":
                self._dataset_text.configure(state=prev_state)
        self._log_write(f"Датасет заполнен значениями по умолчанию ({count} линз).")

    def _sync_dataset_editor(self) -> None:
        mode = self._dataset_mode.get()
        if mode == "default":
            self._dataset_text.configure(state="disabled")
        else:
            self._dataset_text.configure(state="normal")

    def _get_dataset(self, count_linse: int) -> DatasetParseResult:
        if self._dataset_mode.get() == "default":
            ds = DataSetHelper.create_default_dataset(count_linse=count_linse)
            return DatasetParseResult(dataset=ds, warnings=[])

        raw = parse_dataset_text(self._dataset_text.get("1.0", "end"))
        # If user provided count_linse different from needed for a tab, override it but warn.
        if "count_linse" in raw and int(raw["count_linse"]) != int(count_linse):
            raw = dict(raw)
            raw["count_linse"] = int(count_linse)
            w = [f"count_linse был переопределён на {count_linse} для выбранного режима."]
        else:
            w = []

        res = normalize_dataset(raw)
        return DatasetParseResult(dataset=res.dataset, warnings=w + res.warnings)

    def _parse_float(self, s: str, name: str) -> float:
        t = s.strip().replace(",", ".")
        if not t:
            raise ValueError(f"Поле {name} пустое")
        return float(t)

    def _run_in_thread(self, title: str, fn: Callable[[], None]) -> None:
        def wrapper() -> None:
            try:
                self._set_status(f"{title}…")
                t0 = time.perf_counter()
                fn()
                dt = time.perf_counter() - t0
                self._log_write(f"[time] {title}: {dt:.3f} c")
                self._set_status("Готово")
            except Exception as e:
                self._set_status("Ошибка")
                self._log_write(f"[ОШИБКА] {e}")
                self._log_write(traceback.format_exc())

        threading.Thread(target=wrapper, daemon=True).start()

    def _run_dep(self) -> None:
        def job() -> None:
            count = int(self._dep_count.get())
            ds_res = self._get_dataset(count_linse=count)
            for w in ds_res.warnings:
                self._log_write(f"[warn] {w}")

            heights: Optional[dict[UnitType, list[float]]] = None
            # Heights are optional: if any field is filled => require all
            if any(v.get().strip() for v in self._dep_height_vars):
                h_list = [self._parse_float(v.get(), f"h{i+1}") for i, v in enumerate(self._dep_height_vars)]
                heights = {UnitType.MICROMETER: h_list}

            steps = int(self._dep_steps.get())
            lmbd0 = ds_res.dataset["lower_lambda"]
            lmbd1 = ds_res.dataset["upper_lambda"]
            # units are inside dataset; base expects values already scaled by unit.value[0]
            # We pass explicit lambda_massive in SI to avoid confusion.
            l_unit = ds_res.dataset["unit"]["lower_lambda"].value[0]
            u_unit = ds_res.dataset["unit"]["upper_lambda"].value[0]
            lambda_massive = list(__import__("numpy").linspace(lmbd0 * l_unit, lmbd1 * u_unit, steps))

            opt = BaseLensOptimizer()
            lfd = opt.lmbd_focus_dict(dataset=ds_res.dataset, heights=heights, lambda_massive=lambda_massive, return_dict=True)
            foc_dist_m = opt.calc_focus_dist_static(lfd)
            self._log_write(f"Δf = {foc_dist_m * 1e3:.6f} мм")

            # Важно: plt.show() лучше вызывать из главного потока.
            self.master.after(0, lambda: opt.visualize_depend_f_lmbd(return_fig_ax=False, blockAndShow=True))

        self._run_in_thread("Расчёт зависимости f(λ)", job)

    def _run_opt2(self) -> None:
        def job() -> None:
            ds_res = self._get_dataset(count_linse=2)
            for w in ds_res.warnings:
                self._log_write(f"[warn] {w}")

            init_h = [self._parse_float(self._opt2_h_vars[0].get(), "h1"), self._parse_float(self._opt2_h_vars[1].get(), "h2")]
            b = (float(self._opt2_bmin.get()), float(self._opt2_bmax.get()))
            m = (float(self._opt2_m1.get()), float(self._opt2_m2.get()))
            target = self._opt2_target.get()

            self._log_write(f"Старт 2-линз: init_h={init_h}, hbounds={b}, m={m}, target={target}")
            opt = TwoLensOptimizer()
            # Внутри оптимизатора сейчас используется DATA_SET_2 если dataset не передан.
            # Для GUI пробрасываем dataset через методы расчёта/визуализации зависимостей отдельно.
            # Grid-метод в текущей реализации dataset не принимает — запускаем как есть.
            opt.generate_grid_with_fixed_height(
                init_h=init_h,
                hbounds=b,
                m1=m[0],
                m2=m[1],
                h1_or_h2_optimize=target,
                check_neighbour_min=bool(self._opt2_check_nei.get()),
            )

        self._run_in_thread("Оптимизация 2 линзы", job)

    def _run_opt2_analytic(self) -> None:
        def job() -> None:
            ds_res = self._get_dataset(count_linse=2)
            for w in ds_res.warnings:
                self._log_write(f"[warn] {w}")

            h1 = self._parse_float(self._opt2_h_vars[0].get(), "h1")
            h2 = self._parse_float(self._opt2_h_vars[1].get(), "h2")
            m1 = float(self._opt2_m1.get())
            m2 = float(self._opt2_m2.get())

            opt = TwoLensOptimizer()
            # Используем режим 'h1': h_optimize=h1, h_static=h2
            df = opt.calc_focus_dist_2_lisne_analytical(
                h_optimize=h1,
                h_static=h2,
                m1=m1,
                m2=m2,
                h1_or_h2_optimize="h1",
                dataset=ds_res.dataset,
            )
            self._log_write(f"Δf (аналит.) = {df * 1e3:.6f} мм  |  h=[{h1}, {h2}] мкм, m=[{m1}, {m2}]")

        self._run_in_thread("Δf (аналит., 2 линзы)", job)

    def _run_opt2_matrix(self) -> None:
        def job() -> None:
            ds_res = self._get_dataset(count_linse=2)
            for w in ds_res.warnings:
                self._log_write(f"[warn] {w}")

            h1 = self._parse_float(self._opt2_h_vars[0].get(), "h1")
            h2 = self._parse_float(self._opt2_h_vars[1].get(), "h2")
            m1 = float(self._opt2_m1.get())
            m2 = float(self._opt2_m2.get())

            opt = TwoLensOptimizer()
            df = opt.calc_focus_dist_2_linse_matrix(
                h_optimize=h1,
                h_static=h2,
                m1=m1,
                m2=m2,
                h1_or_h2_optimize="h1",
                dataset=ds_res.dataset,
            )
            self._log_write(f"Δf (матр.) = {df * 1e3:.6f} мм  |  h=[{h1}, {h2}] мкм, m=[{m1}, {m2}]")

        self._run_in_thread("Δf (матр., 2 линзы)", job)

    def _run_opt3(self) -> None:
        def job() -> None:
            ds_res = self._get_dataset(count_linse=3)
            for w in ds_res.warnings:
                self._log_write(f"[warn] {w}")

            init_h = [self._parse_float(v.get(), f"h{i+1}") for i, v in enumerate(self._opt3_h_vars)]
            b = (float(self._opt3_bmin.get()), float(self._opt3_bmax.get()))
            h_static = self._opt3_static.get()

            self._log_write(f"Старт 3-линз: init_h={init_h}, hbounds={b}, static={h_static}")
            opt = ThreeLensOptimizer()
            opt.generate_grid_with_fixed_height(
                init_h={UnitType.MICROMETER: init_h},
                hbounds=b,
                dataset=ds_res.dataset,
                h_static=h_static,
            )

        self._run_in_thread("Оптимизация 3 линзы", job)

    def _run_opt4(self) -> None:
        def job() -> None:
            ds_res = self._get_dataset(count_linse=4)
            for w in ds_res.warnings:
                self._log_write(f"[warn] {w}")

            init_h = [self._parse_float(v.get(), f"h{i+1}") for i, v in enumerate(self._opt4_h_vars)]
            b = (float(self._opt4_bmin.get()), float(self._opt4_bmax.get()))
            h_static = self._opt4_static.get()

            self._log_write(f"Старт 4-линз: init_h={init_h}, hbounds={b}, static={h_static}")
            try:
                from optimizers.four_lens import FourLensOptimizer
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "Для оптимизации 4-х линз нужен пакет 'plotly'. Установите: pip install plotly"
                ) from e

            opt = FourLensOptimizer()
            opt.generate_grid_with_fixed_height(
                init_h={UnitType.MICROMETER: init_h},
                hbounds=b,
                dataset=ds_res.dataset,
                h_static=h_static,
            )

        self._run_in_thread("Оптимизация 4 линзы", job)

    def _run_loss2(self) -> None:
        def job() -> None:
            hmin = float(self._loss2_hmin.get())
            hmax = float(self._loss2_hmax.get())
            n = int(self._loss2_points.get())
            if hmax <= hmin:
                raise ValueError("h_max должен быть больше h_min")

            self._log_write(f"Старт ландшафта потерь (2 линзы): h=[{hmin}, {hmax}], points={n}")
            h_range = __import__("numpy").linspace(hmin, hmax, n)
            BaseLensOptimizer().visualize_dummy_loss_by_two_lens(h_range)

        self._run_in_thread("Ландшафт потерь (2 линзы)", job)


def run_app() -> None:
    root = tk.Tk()
    # ttk styling
    try:
        root.call("tk", "scaling", 1.0)
    except Exception:
        pass
    App(root)
    root.mainloop()

