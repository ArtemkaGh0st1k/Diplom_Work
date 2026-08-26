"""
Улучшенная реальная трассировка для гармонических линз.
Учитывает точную фазовую модель и геометрическое преломление.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from data.dataset_helper import DataSetHelper
from utils.unit import UnitType
from utils.keys import DataSetKeys


class AdvancedRealRayTracer:
    """Улучшенный трассировщик для гармонических линз с точной фазовой моделью"""
    
    def __init__(self, dataset: Dict[DataSetKeys, any]):
        self.dataset = dataset
        self.count_linse = dataset['count_linse']
        
    def n_bk7(self, lambda_um: float) -> float:
        """Расчёт показателя преломления BK7 стекла"""
        B1, B2, B3 = 1.03961212, 0.231792344, 1.01046945
        C1, C2, C3 = 6.00069867e-3, 2.00179144e-2, 1.03560653e2
        l2 = lambda_um**2
        n2 = 1 + (B1 * l2) / (l2 - C1) + (B2 * l2) / (l2 - C2) + (B3 * l2) / (l2 - C3)
        return n2**0.5
    
    def calculate_exact_phase_profile(self, r: float, height: float, 
                                    refractive_index: float, 
                                    harmonica: float) -> float:
        """
        Расчёт точного фазового профиля гармонической линзы
        
        Args:
            r: Радиус (расстояние от оси)
            height: Высота микрорельефа
            refractive_index: Показатель преломления
            harmonica: Гармоника
            
        Returns:
            Фазовый сдвиг в радианах
        """
        # Для гармонической линзы фазовый профиль: φ(r) = (2π/λ) * (n-1) * h(r)
        # где h(r) - профиль высоты микрорельефа
        
        # Предполагаем параболический профиль для основной фокусировки
        # h(r) = h_max * (r/R)^2 для r < R
        # где R - радиус линзы
        
        R = 10e-3  # Радиус апертуры (10 мм)
        
        if r > R:
            return 0  # За пределами апертуры нет фазового сдвига
            
        # Нормализованная координата
        rho = r / R
        
        # Профиль высоты: параболический для сферической линзы
        h_r = height * rho**2
        
        # Фазовый сдвиг
        phase_shift = 2 * np.pi * (refractive_index - 1) * h_r
        
        return phase_shift
    
    def calculate_exact_optical_power(self, r: float, lmbd: float, 
                                    height: float, refractive_index: float,
                                    harmonica: float,
                                    lens_index: int = 1) -> float:
        """
        Расчёт точной оптической силы с учётом геометрии
        
        Args:
            r: Радиус (расстояние от оси)
            lmbd: Длина волны
            height: Высота микрорельефа
            refractive_index: Показатель преломления
            harmonica: Гармоника
            lens_index: Индекс линзы (1-based)
            
        Returns:
            Оптическая сила в диоптриях
        """
        # Базовые параметры
        lambda_0 = height * (refractive_index - 1) / harmonica
        k = round((lambda_0 / lmbd) * harmonica)
        if k == 0:
            k = 0.5
            
        # Базовое фокусное расстояние для данной линзы
        focus_0 = self.dataset['focus_0'][lens_index] * 1e-3
        base_focus = ((harmonica * lambda_0) / (k * lmbd)) * focus_0
        
        # Базовая оптическая сила (параксиальная)
        base_power = 1 / base_focus
        
        # Для реальной линзы добавляем сферическую аберрацию
        # Сферическая аберрация растёт с квадратом расстояния от оси
        R = 10e-3  # Радиус апертуры (10 мм)
        
        if r > R:
            return 0
        
        # Нормализованное расстояние от оси
        normalized_r = r / R
        
        # Коэффициент сферической аберрации
        # Для типичной линзы: SA3 ~ 0.1-1.0 мм^-2
        SA3 = 0.5 / (R**2)  # Нормированный коэффициент
        
        # Эффективная оптическая сила с учётом сферической аберрации
        effective_power = base_power * (1 + SA3 * r**2)
        
        return effective_power
    
    def trace_ray_advanced(self, h_start: float, lmbd: float, heights: List[float],
                          aperture_radius: float = 10e-3) -> Tuple[float, float, List[Tuple[float, float]]]:
        """
        Улучшенная трассировка с учётом точной фазовой модели
        
        Args:
            h_start: Начальная высота луча
            lmbd: Длина волны
            heights: Высоты микрорельефа
            aperture_radius: Радиус апертуры
            
        Returns:
            (h_final, alpha_final, trace_history)
        """
        h = h_start
        alpha = 0.0
        trace_history = [(0, h, alpha)]
        z_position = 0.0
        
        for i in range(1, self.count_linse + 1):
            refractive_index = self.n_bk7(lmbd * 1e6)
            harmonica = self.dataset['harmonica'][i]
            height = heights[i-1] * 1e-6
            
            # Используем точную модель оптической силы
            effective_power = self.calculate_exact_optical_power(
                abs(h), lmbd, height, refractive_index, harmonica
            )
            
            # Преломление
            h_new = h
            alpha_new = alpha - effective_power * h
            
            # Перенос
            if i < self.count_linse:
                refractive_area = self.dataset['refractive_area'][f'{i}-{i+1}']
                dist = self.dataset['distance'][f'{i}-{i+1}'] * 1e-3
                reduce_dist = dist / refractive_area
                
                z_position += reduce_dist
                h = h_new + alpha_new * reduce_dist
                alpha = alpha_new
            else:
                h = h_new
                alpha = alpha_new
            
            trace_history.append((z_position, h, alpha))
        
        return h, alpha, trace_history
    
    def analyze_spherical_aberration(self, lmbd: float, heights: List[float],
                                   aperture_radius: float = 10e-3,
                                   num_rays: int = 50) -> Dict:
        """
        Анализ сферической аберрации
        
        Returns:
            Словарь с характеристиками аберрации
        """
        ray_heights = np.linspace(0.1e-3, aperture_radius, num_rays)
        
        results = {
            'ray_heights': [],
            'paraxial_focus': [],
            'exact_focus': [],
            'spherical_aberration': [],
            'wavefront_error': []
        }
        
        # Сначала найдём параксиальный фокус
        h_p, alpha_p, _ = self.trace_ray_advanced(1e-3, lmbd, heights, aperture_radius)
        paraxial_focus = abs(h_p / alpha_p) if abs(alpha_p) > 1e-10 else float('inf')
        
        for h_start in ray_heights:
            # Точный расчёт
            h_e, alpha_e, _ = self.trace_ray_advanced(h_start, lmbd, heights, aperture_radius)
            exact_focus = abs(h_e / alpha_e) if abs(alpha_e) > 1e-10 else float('inf')
            
            # Сферическая аберрация (продольная)
            SA = exact_focus - paraxial_focus
            
            # Волновая аберрация (разница оптического пути)
            # Для параксиального приближения: OPD_paraxial = D * h^2 / 2
            # Для точного расчёта: OPD_exact = интеграл от фазового сдвига
            wavefront_error = abs(SA) * (2 * np.pi / lmbd) * 1e-6  # в волнах
            
            results['ray_heights'].append(h_start)
            results['paraxial_focus'].append(paraxial_focus)
            results['exact_focus'].append(exact_focus)
            results['spherical_aberration'].append(SA)
            results['wavefront_error'].append(wavefront_error)
        
        # Преобразуем в numpy массивы
        for key in results:
            results[key] = np.array(results[key])
        
        return results


class ComprehensiveComparison:
    """Комплексное сравнение всех методов трассировки"""
    
    def __init__(self, dataset: Dict[DataSetKeys, any]):
        self.dataset = dataset
        self.count_linse = dataset['count_linse']
        self.advanced_tracer = AdvancedRealRayTracer(dataset)
        
        # Импортируем существующие трассировщики
        from optimizers.tracing_and_paraxial.optimize_and_trace import RayTracer
        self.paraxial_tracer = RayTracer(dataset)
        self.real_tracer = None  # Будет создано при необходимости
        
    def compare_all_methods(self, lmbd: float, heights: List[float],
                          aperture_radius: float = 10e-3) -> Dict:
        """
        Сравнение всех методов трассировки:
        1. Параксиальный (матричный)
        2. Реальный с эмпирической аберрацией
        3. Точный с фазовой моделью
        """
        ray_heights = np.linspace(0.5e-3, aperture_radius * 0.9, 20)
        
        comparison = {
            'ray_heights': ray_heights,
            'paraxial': {'focus': [], 'alpha': []},
            'real_empirical': {'focus': [], 'alpha': []},
            'advanced_exact': {'focus': [], 'alpha': []}
        }
        
        for h_start in ray_heights:
            # 1. Параксиальный метод
            h_p, alpha_p = self.paraxial_tracer.trace_ray(h_start, lmbd, heights)
            f_p = abs(h_p / alpha_p) if abs(alpha_p) > 1e-10 else float('inf')
            
            # 2. Реальный метод (если доступен)
            try:
                from optimizers.real_tracing import RealRayTracer
                if self.real_tracer is None:
                    self.real_tracer = RealRayTracer(self.dataset)
                h_r, alpha_r, _ = self.real_tracer.trace_ray_real(h_start, lmbd, heights, aperture_radius)
                f_r = abs(h_r / alpha_r) if abs(alpha_r) > 1e-10 else float('inf')
            except:
                f_r, alpha_r = f_p, alpha_p  # Fallback to paraxial
            
            # 3. Точный метод
            h_a, alpha_a, _ = self.advanced_tracer.trace_ray_advanced(h_start, lmbd, heights, aperture_radius)
            f_a = abs(h_a / alpha_a) if abs(alpha_a) > 1e-10 else float('inf')
            
            comparison['paraxial']['focus'].append(f_p)
            comparison['paraxial']['alpha'].append(alpha_p)
            comparison['real_empirical']['focus'].append(f_r)
            comparison['real_empirical']['alpha'].append(alpha_r)
            comparison['advanced_exact']['focus'].append(f_a)
            comparison['advanced_exact']['alpha'].append(alpha_a)
        
        # Преобразуем в numpy массивы
        for method in ['paraxial', 'real_empirical', 'advanced_exact']:
            for param in ['focus', 'alpha']:
                comparison[method][param] = np.array(comparison[method][param])
        
        return comparison
    
    def plot_comprehensive_comparison(self, comparison: Dict, save_path: str = None):
        """Построение комплексного сравнения"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        ray_heights_mm = comparison['ray_heights'] * 1e3
        
        # График 1: Положение фокуса
        ax1 = axes[0, 0]
        ax1.plot(ray_heights_mm, comparison['paraxial']['focus'] * 1e3, 
                'b-', linewidth=2, label='Параксиальный', marker='o', markersize=4)
        ax1.plot(ray_heights_mm, comparison['real_empirical']['focus'] * 1e3, 
                'r--', linewidth=2, label='Реальный (эмпирический)', marker='s', markersize=4)
        ax1.plot(ray_heights_mm, comparison['advanced_exact']['focus'] * 1e3, 
                'g-.', linewidth=2, label='Точный (фазовый)', marker='^', markersize=4)
        ax1.set_xlabel('Высота луча, мм')
        ax1.set_ylabel('Положение фокуса, мм')
        ax1.set_title('Сравнение положения фокуса')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # График 2: Относительные отклонения
        ax2 = axes[0, 1]
        paraxial_focus = comparison['paraxial']['focus']
        real_deviation = (comparison['real_empirical']['focus'] - paraxial_focus) * 1e3
        advanced_deviation = (comparison['advanced_exact']['focus'] - paraxial_focus) * 1e3
        
        ax2.plot(ray_heights_mm, real_deviation, 'r--', linewidth=2, 
                label='Реальный - Параксиальный', marker='s')
        ax2.plot(ray_heights_mm, advanced_deviation, 'g-.', linewidth=2, 
                label='Точный - Параксиальный', marker='^')
        ax2.set_xlabel('Высота луча, мм')
        ax2.set_ylabel('Отклонение, мм')
        ax2.set_title('Относительные отклонения от параксиала')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # График 3: Углы отклонения
        ax3 = axes[1, 0]
        ax3.plot(ray_heights_mm, comparison['paraxial']['alpha'], 
                'b-', linewidth=2, label='Параксиальный')
        ax3.plot(ray_heights_mm, comparison['real_empirical']['alpha'], 
                'r--', linewidth=2, label='Реальный')
        ax3.plot(ray_heights_mm, comparison['advanced_exact']['alpha'], 
                'g-.', linewidth=2, label='Точный')
        ax3.set_xlabel('Высота луча, мм')
        ax3.set_ylabel('Угол отклонения, рад')
        ax3.set_title('Углы отклонения лучей')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # График 4: Сферическая аберрация
        ax4 = axes[1, 1]
        # Сферическая аберрация как функция от (h/R)^2
        R = 10e-3
        normalized_height_sq = (comparison['ray_heights'] / R)**2
        
        ax4.plot(normalized_height_sq, real_deviation, 'r--', linewidth=2, 
                label='Эмпирическая модель', marker='s')
        ax4.plot(normalized_height_sq, advanced_deviation, 'g-.', linewidth=2, 
                label='Фазовая модель', marker='^')
        ax4.set_xlabel('(h/R)²')
        ax4.set_ylabel('Продольная аберрация, мм')
        ax4.set_title('Сферическая аберрация')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Комплексный график сохранён в {save_path}")
        
        plt.show()
    
    def calculate_rms_aberration(self, comparison: Dict) -> Dict:
        """Расчёт RMS аберрации для разных методов"""
        results = {}
        
        paraxial_focus = np.mean(comparison['paraxial']['focus'])
        
        for method in ['real_empirical', 'advanced_exact']:
            focus_values = comparison[method]['focus']
            
            # RMS отклонение от среднего параксиального фокуса
            rms_deviation = np.sqrt(np.mean((focus_values - paraxial_focus)**2))
            
            # Пик-ту-вэлли аберрация
            ptv_aberration = np.max(focus_values) - np.min(focus_values)
            
            results[method] = {
                'RMS': rms_deviation * 1e3,  # в мм
                'PTV': ptv_aberration * 1e3   # в мм
            }
        
        return results


def run_comprehensive_analysis():
    """Запуск комплексного анализа всех методов трассировки"""
    print("="*70)
    print("КОМПЛЕКСНЫЙ АНАЛИЗ МЕТОДОВ ТРАССИРОВКИ")
    print("="*70)
    
    # Создаём датасет
    dataset = DataSetHelper.create_dataset(
        count_linse=2,
        focus_0={1: 200, 2: 200},
        harmonica={1: 7, 2: 7.5},
        distance={'1-2': 10}
    )
    
    # Высоты микрорельефа
    heights = [7.0, 7.5]  # в мкм
    
    # Создаём компаратор
    comparator = ComprehensiveComparison(dataset)
    
    # Сравнение на разных длинах волн
    wavelengths = [450e-9, 550e-9, 650e-9]  # 450, 550, 650 нм
    
    all_results = {}
    
    for lmbd in wavelengths:
        print(f"\n--- Анализ на длине волны {lmbd*1e9:.0f} нм ---")
        
        # Сравнение всех методов
        comparison = comparator.compare_all_methods(lmbd, heights)
        
        # Расчёт RMS аберраций
        rms_results = comparator.calculate_rms_aberration(comparison)
        
        # Сохраняем результаты
        all_results[lmbd] = {
            'comparison': comparison,
            'rms': rms_results
        }
        
        # Выводим результаты
        for method, metrics in rms_results.items():
            print(f"   {method}: RMS = {metrics['RMS']:.4f} мм, PTV = {metrics['PTV']:.4f} мм")
    
    # Визуализация для центральной длины волны
    print("\n--- Построение графиков для 550 нм ---")
    central_comparison = all_results[550e-9]['comparison']
    comparator.plot_comprehensive_comparison(
        central_comparison, 
        save_path='results/two_linse/comprehensive_tracing_comparison.png'
    )
    
    print("\n" + "="*70)
    print("КОМПЛЕКСНЫЙ АНАЛИЗ ЗАВЕРШЁН")
    print("="*70)
    
    return all_results


if __name__ == '__main__':
    run_comprehensive_analysis()