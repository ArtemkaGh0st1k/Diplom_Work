"""
Трассировка лучей через гармонические линзы.

Перенос C++ программы Harmony научного руководителя (Harmlens.cpp, Unit1.cpp)
и связка с параксиальной моделью проекта.
"""

from optimizers.tracing.surface import MODE_CPP, MODE_GRATING, Surface
from optimizers.tracing.system import OpticalSystem
from optimizers.tracing.harmonic_system import build_system, trace_focus_dict

__all__ = ['MODE_CPP', 'MODE_GRATING', 'Surface', 'OpticalSystem',
           'build_system', 'trace_focus_dict']
