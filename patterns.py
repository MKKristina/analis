"""
Модуль для обнаружения паттернов на графиках
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union

def detect_candle_patterns(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Обнаруживает паттерны свечей
    
    Args:
        df: DataFrame с данными OHLC
        
    Returns:
        Словарь с Series для каждого паттерна (1 - паттерн найден, 0 - нет)
    """
    patterns = {}
    
    # Функция для проверки доджи (открытие ~= закрытие)
    def is_doji(open_price, close_price, threshold=0.001):
        return abs(open_price - close_price) / ((open_price + close_price) / 2) < threshold
    
    # Доджи
    patterns['doji'] = pd.Series(0, index=df.index)
    for i in range(len(df)):
        if is_doji(df['open'].iloc[i], df['close'].iloc[i]):
            patterns['doji'].iloc[i] = 1
    
    # Молот и повешенный (длинная нижняя тень)
    patterns['hammer'] = pd.Series(0, index=df.index)
    for i in range(len(df)):
        body_size = abs(df['open'].iloc[i] - df['close'].iloc[i])
        lower_shadow = min(df['open'].iloc[i], df['close'].iloc[i]) - df['low'].iloc[i]
        upper_shadow = df['high'].iloc[i] - max(df['open'].iloc[i], df['close'].iloc[i])
        
        if lower_shadow > 2 * body_size and upper_shadow < body_size * 0.5:
            if i > 0 and df['close'].iloc[i-1] < df['open'].iloc[i-1]:  # Предыдущая свеча медвежья
                patterns['hammer'].iloc[i] = 1
    
    # Поглощение
    patterns['engulfing'] = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        prev_body_size = abs(df['open'].iloc[i-1] - df['close'].iloc[i-1])
        curr_body_size = abs(df['open'].iloc[i] - df['close'].iloc[i])
        
        # Бычье поглощение
        if df['close'].iloc[i-1] < df['open'].iloc[i-1] and df['close'].iloc[i] > df['open'].iloc[i] and \
           df['open'].iloc[i] < df['close'].iloc[i-1] and df['close'].iloc[i] > df['open'].iloc[i-1] and \
           curr_body_size > prev_body_size:
            patterns['engulfing'].iloc[i] = 1
        
        # Медвежье поглощение
        elif df['close'].iloc[i-1] > df['open'].iloc[i-1] and df['close'].iloc[i] < df['open'].iloc[i] and \
             df['open'].iloc[i] > df['close'].iloc[i-1] and df['close'].iloc[i] < df['open'].iloc[i-1] and \
             curr_body_size > prev_body_size:
            patterns['engulfing'].iloc[i] = -1
    
    # Утренняя звезда
    patterns['morning_star'] = pd.Series(0, index=df.index)
    for i in range(2, len(df)):
        # 1-я свеча - медвежья
        # 2-я свеча - маленькое тело
        # 3-я свеча - бычья
        if df['close'].iloc[i-2] < df['open'].iloc[i-2] and \
           abs(df['open'].iloc[i-1] - df['close'].iloc[i-1]) < abs(df['open'].iloc[i-2] - df['close'].iloc[i-2]) * 0.3 and \
           df['close'].iloc[i] > df['open'].iloc[i] and \
           df['close'].iloc[i] > (df['open'].iloc[i-2] + df['close'].iloc[i-2]) / 2:
            patterns['morning_star'].iloc[i] = 1
    
    # Вечерняя звезда
    patterns['evening_star'] = pd.Series(0, index=df.index)
    for i in range(2, len(df)):
        # 1-я свеча - бычья
        # 2-я свеча - маленькое тело
        # 3-я свеча - медвежья
        if df['close'].iloc[i-2] > df['open'].iloc[i-2] and \
           abs(df['open'].iloc[i-1] - df['close'].iloc[i-1]) < abs(df['open'].iloc[i-2] - df['close'].iloc[i-2]) * 0.3 and \
           df['close'].iloc[i] < df['open'].iloc[i] and \
           df['close'].iloc[i] < (df['open'].iloc[i-2] + df['close'].iloc[i-2]) / 2:
            patterns['evening_star'].iloc[i] = -1
    
    return patterns


def detect_chart_patterns(df: pd.DataFrame, period: int = 50, threshold: float = 0.03) -> Dict[str, List]:
    """
    Обнаруживает графические паттерны
    
    Args:
        df: DataFrame с данными OHLC
        period: Период для поиска паттернов
        threshold: Порог для определения паттернов
        
    Returns:
        Словарь с информацией о найденных паттернах
    """
    patterns = {
        'head_and_shoulders': [],
        'double_top': [],
        'double_bottom': [],
        'triangle_ascending': [],
        'triangle_descending': [],
        'triangle_symmetric': [],
        'channel_up': [],
        'channel_down': [],
    }
    
    # Простая реализация обнаружения двойной вершины
    def find_peaks(data, min_distance=5):
        peaks = []
        for i in range(1, len(data)-1):
            if data[i-1] < data[i] and data[i] > data[i+1]:
                peaks.append(i)
        
        # Фильтрация пиков, которые находятся слишком близко друг к другу
        filtered_peaks = [peaks[0]]
        for i in range(1, len(peaks)):
            if peaks[i] - filtered_peaks[-1] >= min_distance:
                filtered_peaks.append(peaks[i])
                
        return filtered_peaks
    
    # Поиск двойных вершин и дна
    if len(df) >= period:
        # Используем последние period точек
        recent_data = df['close'].values[-period:]
        
        # Находим пики (локальные максимумы)
        peaks = find_peaks(recent_data)
        
        # Проверяем двойные вершины (пики примерно на одном уровне)
        if len(peaks) >= 2:
            for i in range(len(peaks)-1):
                for j in range(i+1, len(peaks)):
                    # Если пики примерно на одном уровне
                    if abs(recent_data[peaks[i]] - recent_data[peaks[j]]) / recent_data[peaks[i]] < threshold:
                        patterns['double_top'].append({
                            'first_peak': df.index[-(period-peaks[i])],
                            'second_peak': df.index[-(period-peaks[j])],
                            'confidence': 1 - abs(recent_data[peaks[i]] - recent_data[peaks[j]]) / recent_data[peaks[i]]
                        })
        
        # Находим впадины (локальные минимумы) - аналогично пикам
        def find_troughs(data, min_distance=5):
            troughs = []
            for i in range(1, len(data)-1):
                if data[i-1] > data[i] and data[i] < data[i+1]:
                    troughs.append(i)
            
            filtered_troughs = [troughs[0]] if troughs else []
            for i in range(1, len(troughs)):
                if troughs[i] - filtered_troughs[-1] >= min_distance:
                    filtered_troughs.append(troughs[i])
                    
            return filtered_troughs
        
        troughs = find_troughs(recent_data)
        
        # Проверяем двойное дно (впадины примерно на одном уровне)
        if len(troughs) >= 2:
            for i in range(len(troughs)-1):
                for j in range(i+1, len(troughs)):
                    if abs(recent_data[troughs[i]] - recent_data[troughs[j]]) / recent_data[troughs[i]] < threshold:
                        patterns['double_bottom'].append({
                            'first_trough': df.index[-(period-troughs[i])],
                            'second_trough': df.index[-(period-troughs[j])],
                            'confidence': 1 - abs(recent_data[troughs[i]] - recent_data[troughs[j]]) / recent_data[troughs[i]]
                        })
    
    return patterns


def identify_supports_resistances(df: pd.DataFrame, window: int = 20, 
                               threshold: float = 0.02) -> Dict[str, List[float]]:
    """
    Определяет уровни поддержки и сопротивления на основе исторических данных
    
    Args:
        df: DataFrame с данными свечей
        window: Размер окна для поиска уровней
        threshold: Порог близости цены для группировки уровней
        
    Returns:
        Словарь с уровнями поддержки и сопротивления
    """
    # Находим локальные минимумы и максимумы
    local_min = []
    local_max = []
    
    for i in range(window, len(df) - window):
        # Локальный минимум - цена ниже, чем все соседние точки в окне
        if df['low'].iloc[i] == df['low'].iloc[i-window:i+window].min():
            local_min.append(df['low'].iloc[i])
        
        # Локальный максимум - цена выше, чем все соседние точки в окне
        if df['high'].iloc[i] == df['high'].iloc[i-window:i+window].max():
            local_max.append(df['high'].iloc[i])
    
    # Группируем близкие уровни
    def group_levels(levels, threshold):
        if not levels:
            return []
            
        levels.sort()
        groups = [[levels[0]]]
        
        for level in levels[1:]:
            if level > groups[-1][-1] * (1 + threshold):
                groups.append([level])
            else:
                groups[-1].append(level)
        
        # Вычисляем среднее для каждой группы
        return [sum(group) / len(group) for group in groups]
    
    support_levels = group_levels(local_min, threshold)
    resistance_levels = group_levels(local_max, threshold)
    
    return {
        'support': support_levels,
        'resistance': resistance_levels
    }


def detect_divergence(df: pd.DataFrame, indicator: str = 'rsi_14', 
                     window: int = 20, threshold: float = 0.05) -> pd.Series:
    """
    Обнаруживает дивергенции между ценой и индикатором
    
    Args:
        df: DataFrame с данными свечей и техническими индикаторами
        indicator: Название индикатора для проверки дивергенции
        window: Окно для поиска локальных максимумов и минимумов
        threshold: Порог для определения значимого максимума/минимума
        
    Returns:
        Series с отметками дивергенций (1 - бычья, -1 - медвежья, 0 - нет)
    """
    if indicator not in df.columns:
        raise ValueError(f"Индикатор '{indicator}' отсутствует в DataFrame")
    
    # Создадим Series для результата
    divergence = pd.Series(0, index=df.index)
    
    # Функция для нахождения локальных максимумов и минимумов
    def find_local_extremes(s, window):
        max_idx = []
        min_idx = []
        
        for i in range(window, len(s) - window):
            if s.iloc[i] > s.iloc[i-window:i+window].max() * (1 - threshold):
                max_idx.append(i)
            if s.iloc[i] < s.iloc[i-window:i+window].min() * (1 + threshold):
                min_idx.append(i)
                
        return max_idx, min_idx
    
    # Находим экстремумы цены и индикатора
    price_max_idx, price_min_idx = find_local_extremes(df['close'], window)
    ind_max_idx, ind_min_idx = find_local_extremes(df[indicator], window)
    
    # Проверяем на бычью дивергенцию
    for p in price_min_idx:
        for i in ind_min_idx:
            if abs(p - i) < window/2:  # Близкие экстремумы
                if p > 0 and i > 0 and p < len(df) - 1 and i < len(df) - 1:
                    # Если цена делает более низкий минимум, а индикатор - более высокий
                    if (df['close'].iloc[p] < df['close'].iloc[p-window]) and \
                       (df[indicator].iloc[i] > df[indicator].iloc[i-window]):
                        divergence.iloc[p] = 1  # Бычья дивергенция
    
    # Проверяем на медвежью дивергенцию
    for p in price_max_idx:
        for i in ind_max_idx:
            if abs(p - i) < window/2:  # Близкие экстремумы
                if p > 0 and i > 0 and p < len(df) - 1 and i < len(df) - 1:
                    # Если цена делает более высокий максимум, а индикатор - более низкий
                    if (df['close'].iloc[p] > df['close'].iloc[p-window]) and \
                       (df[indicator].iloc[i] < df[indicator].iloc[i-window]):
                        divergence.iloc[p] = -1  # Медвежья дивергенция
    
    return divergence