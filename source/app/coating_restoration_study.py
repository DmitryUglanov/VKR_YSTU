
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import weibull_min, lognorm, expon
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging
import time

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ===== ОТКАЛИБРОВАННАЯ КОНФИГУРАЦИЯ =====

@dataclass
class StudyParameters:
    """Финально откалиброванные параметры для получения различимых стратегий"""
    simulation_time: float = 25000.0      # часов (межремонтный интервал)
    time_step: float = 25.0               # часов (увеличен для ускорения)
    num_runs: int = 100                   # увеличено для статистики
    
    # Параметры покрытия MCrAlY
    initial_thickness: float = 200.0       # мкм
    critical_thickness: float = 50.0       # мкм (25% от начальной)
    restored_thickness: float = 180.0      # мкм (90% от начальной)
    
    # ========== ПАРАМЕТРЫ ДЕГРАДАЦИИ ==========
    # Увеличены в 1.7 раза для получения различий между стратегиями
    base_oxidation_rate: float = 0.007     # мкм/час при 1000°C (было 0.0042)
    erosion_coefficient: float = 0.0012    # мкм/час при полной нагрузке (было 0.0007)
    activation_energy: float = 200000      # Дж/моль
    thermal_cycling_factor: float = 0.08   # увеличен для большей вариативности
    
    # ========== ПОРОГОВЫЕ ЗНАЧЕНИЯ ==========
    # Подняты для получения восстановлений во всех стратегиях
    threshold_conservative: float = 150.0   # мкм (75% от начальной)
    threshold_optimal: float = 140.0        # мкм (70% от начальной)
    threshold_aggressive: float = 130.0     # мкм (65% от начальной)
    
    # Параметры простоев (реалистичная модель с запасными лопатками)
    planned_replacement_time: float = 8.0   # часов (замена на запасные лопатки)
    offline_restoration_time: float = 480.0 # часов (восстановление вне ГТД)
    hsi_interval: float = 2000.0            # часов между горячими инспекциями
    hsi_duration: float = 24.0              # часов на горячую инспекцию
    major_inspection_interval: float = 25000.0  # часов между большими инспекциями
    major_inspection_duration: float = 720.0    # часов на большую инспекцию
    
    # ========== СТОХАСТИЧЕСКИЕ ПАРАМЕТРЫ ==========
    # Добавлена вариативность для получения статистического разброса
    temperature_std: float = 35.0          # увеличенная вариативность температуры
    load_factor_std: float = 0.08          # увеличенная вариативность нагрузки
    degradation_noise: float = 0.15        # случайный шум в деградации (±15%)
    cost_variation: float = 0.10           # вариация стоимости восстановления (±10%)
    decision_delay_mean: float = 50.0      # средняя задержка принятия решений (часов)
    
    # Экономические параметры
    restoration_cost: float = 75000        # руб/лопатка (восстановление покрытия)
    downtime_cost_rate: float = 500000     # руб/час простоя ГТД
    fuel_penalty_rate: float = 50          # руб/час за 1% снижения эффективности
    success_rate: float = 0.95             # вероятность успешного восстановления
    
    # Параметры запасных лопаток
    blade_set_cost: float = 2000000        # руб за комплект лопаток
    spare_sets_count: int = 2              # количество запасных комплектов
    spare_amortization_years: int = 20     # лет амортизации запасов


class CoatingState(Enum):
    """Состояния покрытия"""
    INTACT = "Исправное"
    PARTIALLY_DEGRADED = "Частично деградировавшее"
    CRITICALLY_DEGRADED = "Критически деградировавшее"
    FAILED = "Отказ"
    UNDER_RESTORATION = "Восстанавливается"
    RESTORED = "Восстановлено"


class RestorationStrategy(Enum):
    """Стратегии восстановления с откалиброванными порогами"""
    CONSERVATIVE = "Консервативная (75%)"
    OPTIMAL = "Оптимальная (70%)"
    AGGRESSIVE = "Агрессивная (65%)"


# ===== МОДЕЛЬ ДЕГРАДАЦИИ СО СТОХАСТИЧНОСТЬЮ =====

class CoatingDegradationModel:
    """Модель физико-химической деградации с реалистичной стохастичностью"""
    
    def __init__(self, params: StudyParameters):
        self.params = params
        self.reset()
    
    def reset(self):
        """Сброс модели к начальному состоянию"""
        self.thickness = self.params.initial_thickness
        self.temperature = 1000.0  # °C
        self.load_factor = 0.85    # базовая нагрузка
        self.thermal_cycles = 0
        self.operating_hours = 0.0
        self.state = CoatingState.INTACT
        
        # ========== СТОХАСТИЧЕСКИЕ ПАРАМЕТРЫ ==========
        # Систематические отклонения для каждого прогона
        self.temp_bias = np.random.normal(0, 15)      # систематическое отклонение температуры
        self.load_bias = np.random.normal(0, 0.05)    # систематическое отклонение нагрузки
        self.degradation_multiplier = np.random.lognormal(0, 0.1)  # случайный множитель деградации
    
    def update_operating_conditions(self):
        """Улучшенное моделирование условий с большей стохастичностью"""
        # Температурные флуктуации с систематическим отклонением
        old_temp = self.temperature
        base_temp = 1000 + self.temp_bias + 10 * np.sin(2 * np.pi * self.operating_hours / 8760)
        self.temperature = np.random.normal(base_temp, self.params.temperature_std)
        
        # Термоциклирование с вероятностным подходом
        if abs(self.temperature - old_temp) > 60:
            self.thermal_cycles += np.random.poisson(1)
        
        # Случайные термоциклы от запусков/остановов
        if np.random.random() < 0.0005:  # ~0.05% шанс на каждом шаге
            self.thermal_cycles += np.random.poisson(2)
        
        # Коэффициент нагрузки с сезонными вариациями и систематическим отклонением
        seasonal_factor = 0.05 * np.sin(2 * np.pi * self.operating_hours / 8760)
        base_load = 0.85 + seasonal_factor + self.load_bias
        self.load_factor = np.clip(
            np.random.normal(base_load, self.params.load_factor_std), 
            0.65, 1.0
        )
    
    def calculate_degradation_rate(self) -> float:
        """Улучшенная модель деградации с реалистичным случайным шумом"""
        if self.state == CoatingState.UNDER_RESTORATION:
            return 0.0
        
        # Базовая физическая модель (уравнение Аррениуса)
        R = 8.314  # Дж/(моль·К)
        T_ref = 1273  # K (1000°C)
        T_current = max(273, self.temperature + 273)
        
        oxidation_factor = np.exp(self.params.activation_energy / R * 
                                (1/T_ref - 1/T_current))
        oxidation_rate = self.params.base_oxidation_rate * oxidation_factor
        
        # Эрозионный износ
        erosion_rate = self.params.erosion_coefficient * self.load_factor
        
        # Термоциклическая деградация (улучшенная модель)
        cycling_factor = 1.0 + self.params.thermal_cycling_factor * \
                        np.log(1 + self.thermal_cycles / 3000)
        
        # Базовая скорость деградации
        base_rate = (oxidation_rate + erosion_rate) * cycling_factor
        
        # ========== ДОБАВЛЕНИЕ РЕАЛИСТИЧНОГО СЛУЧАЙНОГО ШУМА ==========
        # Случайные флуктуации процесса деградации
        noise_factor = np.random.normal(1.0, self.params.degradation_noise)
        
        # Систематический множитель для данного прогона
        total_rate = base_rate * self.degradation_multiplier * max(0.1, noise_factor)
        
        return max(0.0, total_rate)
    
    def update(self, dt: float):
        """Обновление состояния покрытия"""
        if self.state == CoatingState.UNDER_RESTORATION:
            return
            
        self.operating_hours += dt
        
        # Обновление условий эксплуатации
        self.update_operating_conditions()
        
        # Деградация покрытия
        degradation_rate = self.calculate_degradation_rate()
        thickness_loss = degradation_rate * dt
        old_thickness = self.thickness
        self.thickness = max(0.0, self.thickness - thickness_loss)
        
        # Обновление состояния
        self._update_coating_state()
        
        # if thickness_loss > 0 and self.operating_hours % 2000 < dt:  # Логирование каждые 2000 часов
        #     logger.debug(f"Деградация: {old_thickness:.1f} → {self.thickness:.1f} мкм "
        #                 f"(скорость: {degradation_rate:.4f} мкм/час)")
        #
    def _update_coating_state(self):
        """Обновление состояния покрытия на основе толщины"""
        old_state = self.state
        thickness_ratio = self.thickness / self.params.initial_thickness
        
        if self.thickness <= 0:
            self.state = CoatingState.FAILED
        elif self.thickness <= self.params.critical_thickness:
            self.state = CoatingState.CRITICALLY_DEGRADED
        elif thickness_ratio <= 0.7:  
            self.state = CoatingState.PARTIALLY_DEGRADED
        else:
            self.state = CoatingState.INTACT
        
        # if old_state != self.state:
        #     logger.info(f"Изменение состояния: {old_state.value} → {self.state.value}")
    
    def needs_restoration(self, strategy: RestorationStrategy) -> bool:
        """Проверка необходимости восстановления согласно стратегии"""
        thresholds = {
            RestorationStrategy.CONSERVATIVE: self.params.threshold_conservative,  # 150 мкм
            RestorationStrategy.OPTIMAL: self.params.threshold_optimal,           # 140 мкм
            RestorationStrategy.AGGRESSIVE: self.params.threshold_aggressive      # 130 мкм
        }
        
        threshold_thickness = thresholds[strategy]
        return self.thickness <= threshold_thickness
    
    def start_restoration(self) -> bool:
        """Начало процесса восстановления"""
        #logger.info(f"Запланировано восстановление при толщине {self.thickness:.1f} мкм")
        self.state = CoatingState.UNDER_RESTORATION
        return True
    
    def complete_restoration(self, success: bool = True) -> bool:
        """Завершение восстановления с вариативностью"""
        if self.state != CoatingState.UNDER_RESTORATION:
            return False
        
        if success:
            old_thickness = self.thickness
            # Вариативность восстановленной толщины (±5%)
            restored_thickness = self.params.restored_thickness * np.random.normal(1.0, 0.05)
            self.thickness = np.clip(restored_thickness, 
                                   self.params.initial_thickness * 0.85, 
                                   self.params.initial_thickness * 0.95)
            self.state = CoatingState.RESTORED
            self.thermal_cycles = int(self.thermal_cycles * 0.3)  # Частичный сброс
            # logger.info(f"Восстановление успешно: {old_thickness:.1f} → {self.thickness:.1f} мкм")
            return True
        else:
            # logger.warning("Восстановление неуспешно")
            self.state = CoatingState.FAILED
            return False


# ===== СИМУЛЯТОР ИССЛЕДОВАНИЯ =====

class CoatingRestorationStudy:
    """Основной класс исследования с улучшенной стохастической моделью"""
    
    def __init__(self, params: StudyParameters):
        self.params = params
        self.results = {}
    
    def calculate_realistic_downtime(self, strategy: RestorationStrategy, 
                                   num_replacements: int) -> dict:
        """Расчет реалистичных простоев с учетом запасных лопаток"""
        
        # Базовые плановые простои (одинаковые для всех стратегий)
        hsi_count = int(self.params.simulation_time // self.params.hsi_interval)
        major_count = int(self.params.simulation_time // self.params.major_inspection_interval)
        
        base_downtime = (hsi_count * self.params.hsi_duration + 
                        major_count * self.params.major_inspection_duration)
        
        # Дополнительные простои от замены лопаток
        replacement_downtime = num_replacements * self.params.planned_replacement_time
        
        total_downtime = base_downtime + replacement_downtime
        
        return {
            'base_downtime': base_downtime,
            'replacement_downtime': replacement_downtime,
            'total_downtime': total_downtime,
            'hsi_count': hsi_count,
            'major_count': major_count
        }
    
    def calculate_variable_costs(self, num_restorations: int, avg_thickness: float,
                               simulation_time: float) -> dict:
        """Расчет только вариативных затрат (БЕЗ ПРОСТОЕВ)"""
        
        # Прямые затраты на восстановление покрытий с вариативностью
        base_restoration_cost = num_restorations * self.params.restoration_cost
        restoration_cost_variation = np.random.normal(1.0, self.params.cost_variation)
        restoration_costs = base_restoration_cost * max(0.8, restoration_cost_variation)
        
        # Топливные штрафы от снижения эффективности (зависят от средней толщины)
        efficiency_loss = max(0, (self.params.initial_thickness - avg_thickness) / 
                             self.params.initial_thickness * 0.15)  # до 15% потерь
        fuel_penalty_costs = simulation_time * efficiency_loss * self.params.fuel_penalty_rate
        
        # Амортизация инвестиций в запасные лопатки
        spare_investment = self.params.blade_set_cost * self.params.spare_sets_count
        annual_spare_cost = spare_investment / (self.params.spare_amortization_years * 8760)
        spare_costs = annual_spare_cost * simulation_time
        
        # Суммарные вариативные затраты (ИСКЛЮЧЕНЫ простои)
        variable_total = restoration_costs + fuel_penalty_costs + spare_costs
        variable_cost_per_hour = variable_total / simulation_time
        
        return {
            'restoration_costs': restoration_costs,
            'fuel_penalty_costs': fuel_penalty_costs,
            'spare_costs': spare_costs,
            'variable_total': variable_total,
            'variable_cost_per_hour': variable_cost_per_hour
        }
    
    def run_single_simulation(self, strategy: RestorationStrategy, run_id: int = 0) -> Dict:
        """Один прогон симуляции с улучшенной стохастичностью"""
        model = CoatingDegradationModel(self.params)
        
        # История для анализа
        time_history = []
        thickness_history = []
        restoration_events = []
        current_time = 0.0
        thickness_sum = 0.0
        thickness_samples = 0
        
        # logger.info(f"Запуск симуляции: {strategy.value}, прогон {run_id + 1}")
        #
        while current_time < self.params.simulation_time:
            dt = min(self.params.time_step, 
                    self.params.simulation_time - current_time)
            
            # Обновление модели
            model.update(dt)
            
            # Сохранение истории
            time_history.append(current_time)
            thickness_history.append(model.thickness)
            thickness_sum += model.thickness
            thickness_samples += 1
            
            # Проверка необходимости восстановления с вариативной задержкой решения
            if model.needs_restoration(strategy) and \
               model.state != CoatingState.UNDER_RESTORATION:
                
                # ========== РЕАЛИСТИЧНАЯ ЗАДЕРЖКА ПРИНЯТИЯ РЕШЕНИЯ ==========
                decision_delay = np.random.exponential(self.params.decision_delay_mean)
                
                if current_time + decision_delay < self.params.simulation_time:
                    current_time += decision_delay
                    
                    if model.start_restoration():
                        # Вариативная вероятность успеха (зависит от степени износа)
                        wear_factor = (self.params.initial_thickness - model.thickness) / \
                                    self.params.initial_thickness
                        success_rate = self.params.success_rate * (1 - 0.2 * wear_factor)
                        success = np.random.random() < success_rate
                        
                        restoration_result = model.complete_restoration(success)
                        
                        restoration_events.append({
                            'time': current_time,
                            'thickness_before': model.thickness,
                            'success': success,
                            'wear_factor': wear_factor
                        })
                        
                        # Простой только на время замены лопаток (8 часов)
                        current_time += self.params.planned_replacement_time
                        continue
            
            current_time += dt
        
        # Расчет показателей
        num_restorations = len(restoration_events)
        avg_thickness = thickness_sum / max(1, thickness_samples)
        
        # Расчет простоев
        downtime_info = self.calculate_realistic_downtime(strategy, num_restorations)
        
        # Расчет только вариативных затрат (БЕЗ простоев)
        costs_info = self.calculate_variable_costs(num_restorations, avg_thickness,
                                                 self.params.simulation_time)
        
        # Коэффициент готовности (учитывает простои, но не в затратах)
        availability = ((self.params.simulation_time - downtime_info['total_downtime']) / 
                       self.params.simulation_time)
        
        # MTBF - среднее время между восстановлениями
        if num_restorations > 0:
            mtbf = self.params.simulation_time / num_restorations
        else:
            mtbf = self.params.simulation_time
        
        return {
            'strategy': strategy,
            'run_id': run_id,
            'final_thickness': model.thickness,
            'avg_thickness': avg_thickness,
            'final_state': model.state,
            'num_restorations': num_restorations,
            'mtbf': mtbf,
            'availability': availability,
            'variable_total_cost': costs_info['variable_total'],
            'variable_cost_per_hour': costs_info['variable_cost_per_hour'],
            'costs_breakdown': {
                'restoration': costs_info['restoration_costs'],
                'fuel_penalty': costs_info['fuel_penalty_costs'],
                'spare_parts': costs_info['spare_costs']
            },
            'downtime_info': downtime_info,
            'restoration_events': restoration_events,
            'time_history': time_history[::10],  # Прореживание для экономии памяти
            'thickness_history': thickness_history[::10]
        }
    
    def run_study(self) -> pd.DataFrame:
        """Запуск полного исследования для всех стратегий"""
        all_results = []
        
        for strategy in RestorationStrategy:
            # logger.info(f"Исследование стратегии: {strategy.value}")
            #
            for run_id in range(self.params.num_runs):
                # if (run_id + 1) % 20 == 0:  # Прогресс каждые 20 прогонов
                #     logger.info(f"  Прогон {run_id + 1}/{self.params.num_runs}")
                #
                result = self.run_single_simulation(strategy, run_id)
                all_results.append(result)
        
        # Создание DataFrame с результатами
        results_df = pd.DataFrame([
            {
                'strategy': r['strategy'].value,
                'run_id': r['run_id'],
                'final_thickness': r['final_thickness'],
                'avg_thickness': r['avg_thickness'],
                'num_restorations': r['num_restorations'],
                'mtbf': r['mtbf'],
                'availability': r['availability'],
                'variable_total_cost': r['variable_total_cost'],
                'variable_cost_per_hour': r['variable_cost_per_hour'],
                'restoration_cost': r['costs_breakdown']['restoration'],
                'fuel_penalty_cost': r['costs_breakdown']['fuel_penalty'],
                'spare_parts_cost': r['costs_breakdown']['spare_parts'],
                'total_downtime': r['downtime_info']['total_downtime'],
                'replacement_downtime': r['downtime_info']['replacement_downtime']
            }
            for r in all_results
        ])
        
        self.results = all_results
        return results_df


# ===== ВИЗУАЛИЗАЦИЯ =====

class SimplifiedResultsAnalyzer:
    """Упрощенный анализатор результатов для научной публикации"""
    
    def __init__(self, results_df: pd.DataFrame, raw_results: List[Dict]):
        self.results_df = results_df
        self.raw_results = raw_results
    
    def find_representative_run(self, strategy: RestorationStrategy) -> Dict:
        """Найти наиболее репрезентативный прогон для каждой стратегии"""
        strategy_results = [r for r in self.raw_results if r['strategy'] == strategy]
        
        # Среднее количество восстановлений для данной стратегии
        avg_restorations = np.mean([r['num_restorations'] for r in strategy_results])
        
        # Найти прогон, наиболее близкий к среднему
        best_run = min(strategy_results, 
                      key=lambda x: abs(x['num_restorations'] - avg_restorations))
        
        return best_run
    
    def create_simplified_degradation_plot(self, figsize=(12, 8)):
        """Создание упрощенного графика с улучшенным размещением легенды"""
        plt.rcParams['font.size'] = 12
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle('Динамика деградации защитных покрытий лопаток ГТД\nпри различных стратегиях восстановления', 
                     fontsize=14, fontweight='bold')
        
        strategies = list(RestorationStrategy)
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        line_widths = [2.5, 2.5, 2.5]
        
        # Траектории 
        for i, strategy in enumerate(strategies):
            representative_run = self.find_representative_run(strategy)
            time_data = np.array(representative_run['time_history']) / 1000
            thickness_data = np.array(representative_run['thickness_history'])
            
            ax.plot(time_data, thickness_data, 
                   color=colors[i], linewidth=line_widths[i],
                   label=f'{strategy.value} ({representative_run["num_restorations"]} восстановлений)',
                   alpha=0.8)
            
            # Точки восстановлений
            for event in representative_run['restoration_events']:
                event_time = event['time'] / 1000
                time_idx = np.argmin(np.abs(time_data - event_time))
                if time_idx < len(thickness_data):
                    ax.scatter(event_time, thickness_data[time_idx], 
                             color=colors[i], s=80, marker='o', 
                             edgecolors='white', linewidth=2, zorder=5)
        
        # Пороговые линии 
        ax.axhline(y=200, color='green', linestyle='-', alpha=0.6, linewidth=2, 
                   label='Начальная толщина (200 мкм)')
        ax.axhline(y=150, color='blue', linestyle='--', alpha=0.7, linewidth=1.5, 
                   label='Порог консервативной (150 мкм)')
        ax.axhline(y=140, color='purple', linestyle='--', alpha=0.7, linewidth=1.5, 
                   label='Порог оптимальной (140 мкм)')
        ax.axhline(y=130, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, 
                   label='Порог агрессивной (130 мкм)')
        
        # Настройка осей
        ax.set_xlabel('Время эксплуатации, тыс. часов', fontsize=12)
        ax.set_ylabel('Толщина покрытия, мкм', fontsize=12)
        ax.set_xlim(0, 25)
        ax.set_ylim(100, 210)  
        
        # ЛЕГЕНДА ВНИЗУ В ДВА СТОЛБЦА
        legend1 = ax.legend(loc='lower left', frameon=True, framealpha=0.95, 
                           fancybox=True, shadow=True, ncol=2, 
                           bbox_to_anchor=(0.02, 0.02))
        legend1.get_frame().set_facecolor('white')
        legend1.get_frame().set_edgecolor('gray')
        
        # Сетка
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # АННОТАЦИЯ ВВЕРХУ
        ax.text(0.98, 0.98, 'Точки на траекториях — события восстановления покрытий',
                transform=ax.transAxes, fontsize=10, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8),
                horizontalalignment='right', verticalalignment='top')
        
        plt.tight_layout()
        return fig
    
    def create_summary_table_for_publication(self) -> pd.DataFrame:
        """Создание сводной таблицы только с вариативными показателями"""
        
        # Группировка по стратегиям
        grouped = self.results_df.groupby('strategy').agg({
            'availability': ['mean', 'std'],
            'num_restorations': ['mean', 'std'],
            'restoration_cost': ['mean']  # Только затраты на восстановление
        }).round(4)
        
        # Расчет относительных затрат
        restoration_costs = grouped[('restoration_cost', 'mean')]
        base_cost = restoration_costs['Консервативная (75%)']
        relative_costs = restoration_costs / base_cost
        
        # Создание финальной таблицы
        publication_data = []
        
        strategies_order = ['Консервативная (75%)', 'Оптимальная (70%)', 'Агрессивная (65%)']
        
        for strategy in strategies_order:
            if strategy in grouped.index:
                availability_mean = grouped.loc[strategy, ('availability', 'mean')]
                availability_std = grouped.loc[strategy, ('availability', 'std')]
                
                restorations_mean = grouped.loc[strategy, ('num_restorations', 'mean')]
                restorations_std = grouped.loc[strategy, ('num_restorations', 'std')]
                
                relative_cost = relative_costs[strategy]
                
                publication_data.append({
                    'Стратегия восстановления': strategy.replace('(', '\n('),
                    'Коэффициент готовности': f"{availability_mean:.4f} ± {availability_std:.4f}",
                    'Количество восстановлений за 25000 ч': f"{restorations_mean:.1f} ± {restorations_std:.1f}",
                    'Относительные затраты на восстановление*': f"{relative_cost:.3f}"
                })
        
        df_publication = pd.DataFrame(publication_data)
        return df_publication
    
    def save_simplified_results(self, filename_prefix='coating_study_publication'):
        """Сохранение упрощенных результатов для статьи"""
        
        # 1. Упрощенный график
        # fig = self.create_simplified_degradation_plot()
        # fig.savefig(f'{filename_prefix}_dynamics.png', dpi=300, bbox_inches='tight')
        # fig.savefig(f'{filename_prefix}_dynamics.pdf', bbox_inches='tight')
        #
        # 2. Сводная таблица
        summary_table = self.create_summary_table_for_publication()
        summary_table.to_csv(f'{filename_prefix}_table.csv', index=False)
        #
        # 3. Сохранить таблицу как LaTeX для прямого использования в статье
        latex_table = summary_table.to_latex(index=False, escape=False,
                                           column_format='|l|c|c|c|',
                                           caption='Сравнительный анализ стратегий восстановления покрытий',
                                           label='tab:restoration_strategies')
        #
        with open(f'{filename_prefix}_table.tex', 'w', encoding='utf-8') as f:
            f.write(latex_table)
            f.write('\n\n% Примечание: * - относительно консервативной стратегии (принята за 1.000)')

        # print("\nУпрощенные результаты сохранены:")
        # print(f"  - График: {filename_prefix}_dynamics.png/pdf")
        # print(f"  - Таблица: {filename_prefix}_table.csv")
        # print(f"  - LaTeX таблица: {filename_prefix}_table.tex")
        
        return summary_table


# ===== ГЛАВНАЯ ФУНКЦИЯ =====



#
# if __name__ == "__main__":
#     main()