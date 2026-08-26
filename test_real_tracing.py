"""
Тестовый модуль для проверки реализации реальной трассировки
"""

import numpy as np
import matplotlib.pyplot as plt
from data.dataset_helper import DataSetHelper
from optimizers.real_tracing import RealRayTracer, ParaxialVsRealComparison
from optimizers.advanced_real_tracing import AdvancedRealRayTracer, ComprehensiveComparison
from optimizers.tracing_and_paraxial.optimize_and_trace import RayTracer


def test_basic_functionality():
    """Тест базовой функциональности"""
    print("="*50)
    print("ТЕСТ: Базовая функциональность")
    print("="*50)
    
    # Создаём простой датасет
    dataset = DataSetHelper.create_dataset(
        count_linse=2,
        focus_0={1: 200, 2: 200},
        harmonica={1: 7, 2: 7.5},
        distance={'1-2': 10}
    )
    
    heights = [7.0, 7.5]  # в мкм
    lmbd = 550e-9  # 550 нм
    
    # Тестируем базовый трассировщик
    print("1. Тест базового трассировщика...")
    basic_tracer = RayTracer(dataset)
    h, alpha = basic_tracer.trace_ray(1e-3, lmbd, heights)
    print(f"   Параксиальный результат: h={h:.6f}, alpha={alpha:.6f}")
    
    # Тестируем реальный трассировщик
    print("2. Тест реального трассировщика...")
    real_tracer = RealRayTracer(dataset)
    h_r, alpha_r, trace = real_tracer.trace_ray_real(1e-3, lmbd, heights)
    print(f"   Реальный результат: h={h_r:.6f}, alpha={alpha_r:.6f}")
    
    # Тестируем улучшенный трассировщик
    print("3. Тест улучшенного трассировщика...")
    advanced_tracer = AdvancedRealRayTracer(dataset)
    h_a, alpha_a, trace_a = advanced_tracer.trace_ray_advanced(1e-3, lmbd, heights)
    print(f"   Улучшенный результат: h={h_a:.6f}, alpha={alpha_a:.6f}")
    
    # Проверяем, что результаты близки для малых высот
    assert abs(h - h_r) < 1e-3, "Разница между параксиальным и реальным слишком велика"
    assert abs(h - h_a) < 1e-3, "Разница между параксиальным и улучшенным слишком велика"
    
    print("   ✓ Все тесты пройдены")
    return True


def test_spherical_aberration():
    """Тест сферической аберрации"""
    print("\n" + "="*50)
    print("ТЕСТ: Сферическая аберрация")
    print("="*50)
    
    dataset = DataSetHelper.create_dataset(
        count_linse=2,
        focus_0={1: 200, 2: 200},
        harmonica={1: 7, 2: 7.5},
        distance={'1-2': 10}
    )
    
    heights = [7.0, 7.5]
    lmbd = 550e-9
    
    # Тестируем сферическую аберрацию
    real_tracer = RealRayTracer(dataset)
    focus_info = real_tracer.find_focus_with_spherical_aberration(lmbd, heights)
    
    print(f"   Параксиальный фокус: {focus_info['paraxial_focus']*1e3:.4f} мм")
    print(f"   Краевой фокус: {focus_info['edge_focus']*1e3:.4f} мм")
    print(f"   Продольная аберрация: {focus_info['longitudinal_SA']*1e3:.4f} мм")
    
    # Проверяем, что есть аберрация
    assert abs(focus_info['longitudinal_SA']) > 1e-6, "Сферическая аберрация слишком мала"
    
    # Тестируем улучшенный анализатор
    advanced_tracer = AdvancedRealRayTracer(dataset)
    sa_analysis = advanced_tracer.analyze_spherical_aberration(lmbd, heights)
    
    print(f"   Максимальная аберрация (точный): {np.max(np.abs(sa_analysis['spherical_aberration']))*1e3:.4f} мм")
    
    print("   ✓ Тест сферической аберрации пройден")
    return True


def test_wavelength_dependence():
    """Тест зависимости от длины волны"""
    print("\n" + "="*50)
    print("ТЕСТ: Зависимость от длины волны")
    print("="*50)
    
    dataset = DataSetHelper.create_dataset(
        count_linse=2,
        focus_0={1: 200, 2: 200},
        harmonica={1: 7, 2: 7.5},
        distance={'1-2': 10}
    )
    
    heights = [7.0, 7.5]
    wavelengths = np.linspace(400e-9, 700e-9, 10)
    
    # Сравниваем параксиальный и реальный подходы
    paraxial_foci = []
    real_foci = []
    
    for lmbd in wavelengths:
        # Параксиальный
        basic_tracer = RayTracer(dataset)
        h_p, alpha_p = basic_tracer.trace_ray(1e-3, lmbd, heights)
        f_p = abs(h_p / alpha_p) if abs(alpha_p) > 1e-10 else float('inf')
        
        # Реальный
        real_tracer = RealRayTracer(dataset)
        focus_info = real_tracer.find_focus_with_spherical_aberration(lmbd, heights)
        f_r = focus_info['paraxial_focus']
        
        paraxial_foci.append(f_p)
        real_foci.append(f_r)
    
    paraxial_foci = np.array(paraxial_foci)
    real_foci = np.array(real_foci)
    
    # Проверяем хроматическую аберрацию
    paraxial_chromatic = np.max(paraxial_foci) - np.min(paraxial_foci)
    real_chromatic = np.max(real_foci) - np.min(real_foci)
    
    print(f"   Параксиальная хроматическая аберрация: {paraxial_chromatic*1e3:.4f} мм")
    print(f"   Реальная хроматическая аберрация: {real_chromatic*1e3:.4f} мм")
    
    # Проверяем, что аберрации существуют
    assert paraxial_chromatic > 1e-6, "Параксиальная хроматическая аберрация слишком мала"
    assert real_chromatic > 1e-6, "Реальная хроматическая аберрация слишком мала"
    
    print("   ✓ Тест зависимости от длины волны пройден")
    return True


def test_aperture_dependence():
    """Тест зависимости от апертуры"""
    print("\n" + "="*50)
    print("ТЕСТ: Зависимость от апертуры")
    print("="*50)
    
    dataset = DataSetHelper.create_dataset(
        count_linse=2,
        focus_0={1: 200, 2: 200},
        harmonica={1: 7, 2: 7.5},
        distance={'1-2': 10}
    )
    
    heights = [7.0, 7.5]
    lmbd = 550e-9
    
    aperture_radii = np.linspace(5e-3, 20e-3, 5)  # от 5 до 20 мм
    
    results = []
    
    for aperture in aperture_radii:
        real_tracer = RealRayTracer(dataset)
        focus_info = real_tracer.find_focus_with_spherical_aberration(lmbd, heights, aperture)
        
        results.append({
            'aperture': aperture,
            'paraxial_focus': focus_info['paraxial_focus'],
            'edge_focus': focus_info['edge_focus'],
            'longitudinal_SA': focus_info['longitudinal_SA']
        })
        
        print(f"   Апертура {aperture*1e3:.0f} мм: аберрация {abs(focus_info['longitudinal_SA'])*1e3:.4f} мм")
    
    # Проверяем, что аберрация растёт с апертурой
    aberrations = [abs(r['longitudinal_SA']) for r in results]
    assert all(aberrations[i] <= aberrations[i+1] for i in range(len(aberrations)-1)), \
        "Аберрация должна расти с увеличением апертуры"
    
    print("   ✓ Тест зависимости от апертуры пройден")
    return True


def run_visualization_tests():
    """Запуск визуализационных тестов"""
    print("\n" + "="*50)
    print("ТЕСТ: Визуализация")
    print("="*50)
    
    dataset = DataSetHelper.create_dataset(
        count_linse=2,
        focus_0={1: 200, 2: 200},
        harmonica={1: 7, 2: 7.5},
        distance={'1-2': 10}
    )
    
    heights = [7.0, 7.5]
    lmbd = 550e-9
    
    # Сравнение методов
    comparator = ParaxialVsRealComparison(dataset)
    comparison = comparator.compare_tracing(lmbd, heights)
    
    # Построение графиков
    try:
        comparator.plot_comparison(comparison, save_path='results/two_linse/test_comparison.png')
        print("   ✓ Графики успешно построены и сохранены")
    except Exception as e:
        print(f"   ✗ Ошибка при построении графиков: {e}")
        return False
    
    # Комплексное сравнение
    try:
        comprehensive_comparator = ComprehensiveComparison(dataset)
        comprehensive_comparison = comprehensive_comparator.compare_all_methods(lmbd, heights)
        comprehensive_comparator.plot_comprehensive_comparison(
            comprehensive_comparison, 
            save_path='results/two_linse/test_comprehensive.png'
        )
        print("   ✓ Комплексные графики успешно построены")
    except Exception as e:
        print(f"   ✗ Ошибка при построении комплексных графиков: {e}")
        return False
    
    return True


def run_all_tests():
    """Запуск всех тестов"""
    print("ЗАПУСК ТЕСТОВ РЕАЛЬНОЙ ТРАССИРОВКИ")
    print("="*60)
    
    tests = [
        test_basic_functionality,
        test_spherical_aberration,
        test_wavelength_dependence,
        test_aperture_dependence,
        run_visualization_tests
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✓ {test.__name__} - ПРОЙДЕН")
            else:
                print(f"✗ {test.__name__} - НЕ ПРОЙДЕН")
        except Exception as e:
            print(f"✗ {test.__name__} - ОШИБКА: {e}")
    
    print("\n" + "="*60)
    print(f"РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ: {passed}/{total} тестов пройдено")
    print("="*60)
    
    if passed == total:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Реализация работает корректно.")
    else:
        print("⚠️  Некоторые тесты не пройдены. Проверьте реализацию.")
    
    return passed == total


if __name__ == '__main__':
    run_all_tests()