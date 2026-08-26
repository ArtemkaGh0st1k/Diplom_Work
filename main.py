from optimizers.tracing_and_paraxial.optimize_and_trace import OptimizationAndTracing
from optimizers.tracing_and_paraxial.quick_optimize import QuickRayTracer
from optimizers.real_tracing import run_full_comparison

if __name__ == '__main__':
   print("Запуск оптимизации и параксиальной трассировки...")
   optimizerAndTrace = OptimizationAndTracing()
   optimizerAndTrace.start()
   
   print("\n")
   
   print("Запуск сравнения параксиальной и реальной трассировки...")
   run_full_comparison()
