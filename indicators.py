"""
Модуль для работы с техническими индикаторами.
Обертка над нашей собственной реализацией, заменяющая pandas_ta.
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, List, Optional, Tuple

# Импортируем наши функции из технического модуля
from analysis.technical_indicators import (
    sma, ema, rsi, macd, bollinger_bands, stochastic_oscillator,
    atr, obv, adx, cci, calculate_all_indicators
)

def calculate_indicators(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Рассчитывает технические индикаторы для данного датафрейма
    
    Args:
        df: DataFrame с данными свечей (требуются столбцы open, high, low, close, volume)
        columns: Список индикаторов для расчета (None - все индикаторы)
        
    Returns:
        DataFrame с добавленными индикаторами
    """
    # Проверим наличие необходимых колонок
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"В DataFrame отсутствует колонка '{col}', необходимая для расчета индикаторов")
    
    # Если список индикаторов не указан, считаем все индикаторы
    if columns is None:
        return calculate_all_indicators(df)
    
    # Копируем датафрейм, чтобы не изменять оригинал
    result = df.copy()
    
    # Словарь всех доступных индикаторов
    all_indicators = {
        'sma_20': lambda: sma(result['close'], 20),
        'sma_50': lambda: sma(result['close'], 50),
        'sma_200': lambda: sma(result['close'], 200),
        'ema_20': lambda: ema(result['close'], 20),
        'ema_50': lambda: ema(result['close'], 50),
        'rsi_14': lambda: rsi(result['close'], 14),
        'macd': lambda: macd(result['close'])[0],
        'macd_signal': lambda: macd(result['close'])[1],
        'macd_hist': lambda: macd(result['close'])[2],
        'bb_upper': lambda: bollinger_bands(result['close'])[0],
        'bb_middle': lambda: bollinger_bands(result['close'])[1],
        'bb_lower': lambda: bollinger_bands(result['close'])[2],
        'stoch_k': lambda: stochastic_oscillator(result['high'], result['low'], result['close'])[0],
        'stoch_d': lambda: stochastic_oscillator(result['high'], result['low'], result['close'])[1],
        'atr_14': lambda: atr(result['high'], result['low'], result['close']),
        'obv': lambda: obv(result['close'], result['volume']),
        'adx': lambda: adx(result['high'], result['low'], result['close'])[0],
        'plus_di': lambda: adx(result['high'], result['low'], result['close'])[1],
        'minus_di': lambda: adx(result['high'], result['low'], result['close'])[2],
        'cci': lambda: cci(result['high'], result['low'], result['close'])
    }
    
    # Рассчитываем только запрошенные индикаторы
    for col in columns:
        if col in all_indicators:
            result[col] = all_indicators[col]()
        else:
            raise ValueError(f"Неизвестный индикатор: {col}")
    
    return result


def get_trend_strength(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Определяет силу тренда на основе ADX, направления и Price Rate of Change
    
    Args:
        df: DataFrame с данными свечей и техническими индикаторами
        window: Окно для расчета изменения цены
        
    Returns:
        Series со значениями силы тренда от -100 до 100
    """
    # Проверим наличие ADX
    if 'adx' not in df.columns:
        adx_values, plus_di, minus_di = adx(df['high'], df['low'], df['close'])
        df['adx'] = adx_values
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
    
    # Рассчитаем ROC (Rate of Change)
    price_roc = (df['close'] / df['close'].shift(window)) * 100 - 100
    
    # Определяем направление тренда на основе DI+ и DI-
    trend_direction = np.where(df['plus_di'] > df['minus_di'], 1, -1)
    
    # Рассчитываем силу тренда от -100 до 100
    trend_strength = df['adx'] * trend_direction
    
    # Учитываем скорость изменения цены
    trend_strength = trend_strength * (1 + price_roc / 100)
    
    # Масштабируем до диапазона -100 до 100
    max_value = max(abs(trend_strength.min()), abs(trend_strength.max()))
    if max_value > 0:
        trend_strength = trend_strength * (100 / max_value)
    
    return trend_strength


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


def identify_supports_resistances(df: pd.DataFrame, window: int = 20, 
                               threshold: float = 0.02) -> Tuple[List[float], List[float]]:
    """
    Определяет уровни поддержки и сопротивления на основе исторических данных
    
    Args:
        df: DataFrame с данными свечей
        window: Размер окна для поиска уровней
        threshold: Порог близости цены для группировки уровней
        
    Returns:
        Кортеж с двумя списками (уровни поддержки, уровни сопротивления)
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
    
    return support_levels, resistance_levels


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

















# """
# Модуль расчета технических индикаторов для анализа криптовалют.

# Содержит функции для расчета основных технических индикаторов,
# используемых в техническом анализе криптовалютного рынка.
# """

# import pandas as pd
# import numpy as np
# import logging

# # Импортируем pandas-ta вместо talib
# import pandas_ta as ta

# # Настройка логирования
# logger = logging.getLogger(__name__)

# def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Расчет технических индикаторов для данных свечей используя pandas_ta
    
#     Args:
#         df: DataFrame с данными свечей (OHLCV)
        
#     Returns:
#         DataFrame с добавленными индикаторами
#     """
#     if df.empty:
#         return df
    
#     # Копируем DataFrame для добавления индикаторов
#     result_df = df.copy()
    
#     try:
#         # RSI (Relative Strength Index)
#         result_df['rsi_14'] = ta.rsi(df['close'], length=14)
        
#         # MACD (Moving Average Convergence Divergence)
#         macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
#         result_df['macd'] = macd['MACD_12_26_9']
#         result_df['macd_signal'] = macd['MACDs_12_26_9']
#         result_df['macd_hist'] = macd['MACDh_12_26_9']
        
#         # SMA (Simple Moving Average)
#         for period in [9, 20, 50, 200]:
#             result_df[f'sma_{period}'] = ta.sma(df['close'], length=period)
        
#         # EMA (Exponential Moving Average)
#         for period in [9, 20, 50, 200]:
#             result_df[f'ema_{period}'] = ta.ema(df['close'], length=period)
        
#         # Bollinger Bands
#         bb = ta.bbands(df['close'], length=20, std=2)
#         result_df['bb_upper'] = bb['BBU_20_2.0']
#         result_df['bb_middle'] = bb['BBM_20_2.0']
#         result_df['bb_lower'] = bb['BBL_20_2.0']
        
#         # ATR (Average True Range)
#         result_df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
#         # OBV (On-Balance Volume)
#         result_df['obv'] = ta.obv(df['close'], df['volume'])
        
#         # Stochastic Oscillator
#         stoch = ta.stoch(df['high'], df['low'], df['close'], k=5, d=3, smooth_k=3)
#         result_df['stoch_k'] = stoch['STOCHk_5_3_3']
#         result_df['stoch_d'] = stoch['STOCHd_5_3_3']
        
#         # ADX (Average Directional Index)
#         adx = ta.adx(df['high'], df['low'], df['close'], length=14)
#         result_df['adx'] = adx['ADX_14']
        
#         # CCI (Commodity Channel Index)
#         result_df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=14)
        
#         # Ichimoku Cloud
#         ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])
#         result_df['tenkan_sen'] = ichimoku['ISA_9']
#         result_df['kijun_sen'] = ichimoku['ISB_26']
#         result_df['senkou_span_a'] = ichimoku['ITS_9']
#         result_df['senkou_span_b'] = ichimoku['IKS_26']
#         result_df['chikou_span'] = ichimoku['ICS_26']
        
#     except Exception as e:
#         logger.error(f"Ошибка расчета индикаторов: {str(e)}")
    
#     return result_df

# def detect_support_resistance(df: pd.DataFrame, window_size: int = 10, threshold: float = 0.02) -> dict:
#     """
#     Определение уровней поддержки и сопротивления
    
#     Args:
#         df: DataFrame с данными свечей
#         window_size: Размер окна для поиска локальных максимумов и минимумов
#         threshold: Порог близости цен для группировки уровней
        
#     Returns:
#         Словарь с уровнями поддержки и сопротивления
#     """
#     if df.empty:
#         return {"support": [], "resistance": []}
    
#     try:
#         # Находим локальные минимумы и максимумы
#         local_min = []
#         local_max = []
        
#         for i in range(window_size, len(df) - window_size):
#             # Проверяем окно слева и справа
#             if all(df['low'].iloc[i] <= df['low'].iloc[i-j] for j in range(1, window_size+1)) and \
#                all(df['low'].iloc[i] <= df['low'].iloc[i+j] for j in range(1, window_size+1)):
#                 local_min.append(df['low'].iloc[i])
            
#             if all(df['high'].iloc[i] >= df['high'].iloc[i-j] for j in range(1, window_size+1)) and \
#                all(df['high'].iloc[i] >= df['high'].iloc[i+j] for j in range(1, window_size+1)):
#                 local_max.append(df['high'].iloc[i])
        
#         # Группировка близких уровней
#         support_levels = []
#         resistance_levels = []
        
#         # Текущая цена
#         current_price = df['close'].iloc[-1]
        
#         # Группировка уровней поддержки
#         local_min.sort()
        
#         i = 0
#         while i < len(local_min):
#             level = local_min[i]
#             group = [level]
#             j = i + 1
            
#             while j < len(local_min) and (local_min[j] - level) / level < threshold:
#                 group.append(local_min[j])
#                 j += 1
            
#             support_levels.append(sum(group) / len(group))
#             i = j
        
#         # Группировка уровней сопротивления
#         local_max.sort()
        
#         i = 0
#         while i < len(local_max):
#             level = local_max[i]
#             group = [level]
#             j = i + 1
            
#             while j < len(local_max) and (local_max[j] - level) / level < threshold:
#                 group.append(local_max[j])
#                 j += 1
            
#             resistance_levels.append(sum(group) / len(group))
#             i = j
        
#         # Фильтруем уровни, чтобы поддержка была ниже текущей цены, а сопротивление - выше
#         support_levels = [level for level in support_levels if level < current_price]
#         resistance_levels = [level for level in resistance_levels if level > current_price]
        
#         # Сортируем по удалённости от текущей цены
#         support_levels.sort(key=lambda x: current_price - x)
#         resistance_levels.sort(key=lambda x: x - current_price)
        
#         return {
#             "support": [float(level) for level in support_levels[:3]],  # Возвращаем 3 ближайших уровня
#             "resistance": [float(level) for level in resistance_levels[:3]]  # Возвращаем 3 ближайших уровня
#         }
    
#     except Exception as e:
#         logger.error(f"Ошибка определения уровней поддержки и сопротивления: {str(e)}")
#         return {"support": [], "resistance": []}

# def detect_patterns(df: pd.DataFrame) -> dict:
#     """
#     Обнаружение паттернов на свечном графике c использованием pandas-ta
    
#     Args:
#         df: DataFrame с данными свечей
        
#     Returns:
#         Словарь с обнаруженными паттернами
#     """
#     if df.empty:
#         return {"patterns": []}
    
#     try:
#         patterns = {"patterns": []}
        
#         # Используем pandas-ta для обнаружения паттернов
#         # Проверка дожи (Doji)
#         doji = ta.cdl_pattern(df.reset_index(), name="doji")
#         if doji is not None and not doji.empty:
#             doji_dates = doji.loc[doji.iloc[:, 0] != 0].index
#             for idx in doji_dates:
#                 if idx >= len(df) - 5:  # Только последние 5 свечей
#                     patterns["patterns"].append({
#                         "name": "Doji",
#                         "type": "neutral",
#                         "date": df.index[idx].strftime('%Y-%m-%d %H:%M:%S'),
#                         "strength": 1
#                     })
        
#         # Молот (Hammer)
#         hammer = ta.cdl_pattern(df.reset_index(), name="hammer")
#         if hammer is not None and not hammer.empty:
#             hammer_dates = hammer.loc[hammer.iloc[:, 0] != 0].index
#             for idx in hammer_dates:
#                 if idx >= len(df) - 5:  # Только последние 5 свечей
#                     patterns["patterns"].append({
#                         "name": "Hammer",
#                         "type": "bullish",
#                         "date": df.index[idx].strftime('%Y-%m-%d %H:%M:%S'),
#                         "strength": 2
#                     })
        
#         # Поглощающая свеча (Engulfing)
#         engulfing = ta.cdl_pattern(df.reset_index(), name="engulfing")
#         if engulfing is not None and not engulfing.empty:
#             engulfing_dates = engulfing.loc[engulfing.iloc[:, 0] != 0].index
#             for idx in engulfing_dates:
#                 if idx >= len(df) - 5:  # Только последние 5 свечей
#                     pattern_type = "bullish" if engulfing.iloc[idx, 0] > 0 else "bearish"
#                     patterns["patterns"].append({
#                         "name": "Engulfing",
#                         "type": pattern_type,
#                         "date": df.index[idx].strftime('%Y-%m-%d %H:%M:%S'),
#                         "strength": 2
#                     })
        
#         # Сортировка по дате (от новых к старым)
#         patterns["patterns"].sort(key=lambda x: x["date"], reverse=True)
        
#         return patterns
        
#     except Exception as e:
#         logger.error(f"Ошибка обнаружения паттернов: {str(e)}")
#         return {"patterns": []}