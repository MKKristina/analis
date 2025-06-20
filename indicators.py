"""
Модуль технических индикаторов для анализа финансовых данных
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, Dict, List

# Перенос функций из technical_indicators.py

def sma(data: Union[pd.Series, np.ndarray], period: int) -> Union[pd.Series, np.ndarray]:
    """
    Простая скользящая средняя (Simple Moving Average)
    
    Args:
        data: Массив цен закрытия
        period: Период усреднения
        
    Returns:
        Массив значений SMA
    """
    if isinstance(data, pd.Series):
        return data.rolling(window=period).mean()
    else:
        sma_values = np.full_like(data, np.nan, dtype=float)
        for i in range(period - 1, len(data)):
            sma_values[i] = np.mean(data[i - period + 1:i + 1])
        return sma_values


def ema(data: Union[pd.Series, np.ndarray], period: int) -> Union[pd.Series, np.ndarray]:
    """
    Экспоненциальная скользящая средняя (Exponential Moving Average)
    
    Args:
        data: Массив цен закрытия
        period: Период усреднения
        
    Returns:
        Массив значений EMA
    """
    if isinstance(data, pd.Series):
        return data.ewm(span=period, adjust=False).mean()
    else:
        alpha = 2 / (period + 1)
        ema_values = np.full_like(data, np.nan, dtype=float)
        
        # Инициализация первого значения
        if len(data) >= period:
            ema_values[period - 1] = np.mean(data[:period])
        
        # Расчёт остальных значений
        for i in range(period, len(data)):
            ema_values[i] = alpha * data[i] + (1 - alpha) * ema_values[i - 1]
            
        return ema_values


def rsi(data: Union[pd.Series, np.ndarray], period: int = 14) -> Union[pd.Series, np.ndarray]:
    """
    Индекс относительной силы (Relative Strength Index)
    
    Args:
        data: Массив цен закрытия
        period: Период расчета
        
    Returns:
        Массив значений RSI
    """
    if isinstance(data, pd.Series):
        delta = data.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Избегаем деления на ноль
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        rsi_values = 100 - (100 / (1 + rs))
        
        return rsi_values
    else:
        delta = np.diff(data)
        delta = np.insert(delta, 0, 0)
        
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(data, np.nan, dtype=float)
        avg_loss = np.full_like(data, np.nan, dtype=float)
        
        # Инициализация первого значения
        if len(data) >= period:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
        
        # Расчет последующих значений
        for i in range(period + 1, len(data)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i]) / period
        
        rs = np.zeros_like(data, dtype=float)
        # Избегаем деления на ноль
        rs[period:] = avg_gain[period:] / np.where(avg_loss[period:] != 0, avg_loss[period:], np.finfo(float).eps)
        
        rsi_values = 100 - (100 / (1 + rs))
        
        return rsi_values


def macd(data: Union[pd.Series, np.ndarray], fast_period: int = 12, 
         slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Схождение/Расхождение скользящих средних (Moving Average Convergence/Divergence)
    
    Args:
        data: Массив цен закрытия
        fast_period: Период быстрой EMA
        slow_period: Период медленной EMA
        signal_period: Период сигнальной EMA
        
    Returns:
        Кортеж (MACD, сигнальная линия, гистограмма)
    """
    fast_ema_values = ema(data, fast_period)
    slow_ema_values = ema(data, slow_period)
    
    # Линия MACD
    macd_line = fast_ema_values - slow_ema_values
    
    # Сигнальная линия
    if isinstance(macd_line, pd.Series):
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    else:
        signal_line = ema(macd_line, signal_period)
    
    # Гистограмма
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def bollinger_bands(data: Union[pd.Series, np.ndarray], period: int = 20, 
                   std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Полосы Боллинджера (Bollinger Bands)
    
    Args:
        data: Массив цен закрытия
        period: Период SMA для средней линии
        std_dev: Количество стандартных отклонений для верхней и нижней полос
        
    Returns:
        Кортеж (верхняя полоса, средняя полоса, нижняя полоса)
    """
    if isinstance(data, pd.Series):
        middle_band = data.rolling(window=period).mean()
        rolling_std = data.rolling(window=period).std()
        
        upper_band = middle_band + (rolling_std * std_dev)
        lower_band = middle_band - (rolling_std * std_dev)
        
    else:
        middle_band = np.full_like(data, np.nan, dtype=float)
        rolling_std = np.full_like(data, np.nan, dtype=float)
        
        for i in range(period - 1, len(data)):
            window = data[i - period + 1:i + 1]
            middle_band[i] = np.mean(window)
            rolling_std[i] = np.std(window)
        
        upper_band = middle_band + (rolling_std * std_dev)
        lower_band = middle_band - (rolling_std * std_dev)
    
    return upper_band, middle_band, lower_band


def stochastic_oscillator(high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray], 
                         close: Union[pd.Series, np.ndarray], k_period: int = 14, 
                         d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Стохастический осциллятор (Stochastic Oscillator)
    
    Args:
        high: Массив максимальных цен
        low: Массив минимальных цен
        close: Массив цен закрытия
        k_period: Период для линии %K
        d_period: Период для линии %D
        
    Returns:
        Кортеж (линия %K, линия %D)
    """
    if isinstance(high, pd.Series) and isinstance(low, pd.Series) and isinstance(close, pd.Series):
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        # Избегаем деления на ноль
        denominator = highest_high - lowest_low
        denominator = denominator.replace(0, np.finfo(float).eps)
        
        k = 100 * ((close - lowest_low) / denominator)
        d = k.rolling(window=d_period).mean()
        
    else:
        k = np.full_like(close, np.nan, dtype=float)
        
        for i in range(k_period - 1, len(close)):
            window_low = low[i - k_period + 1:i + 1]
            window_high = high[i - k_period + 1:i + 1]
            
            lowest_low = np.min(window_low)
            highest_high = np.max(window_high)
            
            denominator = highest_high - lowest_low
            if denominator != 0:
                k[i] = 100 * ((close[i] - lowest_low) / denominator)
            else:
                k[i] = 50  # Если диапазон равен нулю, используем 50%
        
        # Расчёт %D как SMA от %K
        d = np.full_like(close, np.nan, dtype=float)
        for i in range(k_period + d_period - 2, len(close)):
            d[i] = np.mean(k[i - d_period + 1:i + 1])
    
    return k, d


def atr(high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray], 
       close: Union[pd.Series, np.ndarray], period: int = 14) -> np.ndarray:
    """
    Средний истинный диапазон (Average True Range)
    
    Args:
        high: Массив максимальных цен
        low: Массив минимальных цен
        close: Массив цен закрытия
        period: Период расчета
        
    Returns:
        Массив значений ATR
    """
    if isinstance(high, pd.Series) and isinstance(low, pd.Series) and isinstance(close, pd.Series):
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_values = tr.rolling(window=period).mean()
        
        return atr_values
    else:
        n = len(high)
        tr = np.zeros(n, dtype=float)
        
        # Для первой точки используем только high-low
        tr[0] = high[0] - low[0]
        
        # Расчет True Range для остальных точек
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], 
                        abs(high[i] - close[i-1]), 
                        abs(low[i] - close[i-1]))
        
        # Расчет ATR
        atr_values = np.full_like(close, np.nan, dtype=float)
        
        if n >= period:
            atr_values[period-1] = np.mean(tr[:period])
        
        for i in range(period, n):
            atr_values[i] = (atr_values[i-1] * (period - 1) + tr[i]) / period
        
        return atr_values


def obv(close: Union[pd.Series, np.ndarray], volume: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """
    Балансовый объем (On Balance Volume)
    
    Args:
        close: Массив цен закрытия
        volume: Массив объемов
        
    Returns:
        Массив значений OBV
    """
    if isinstance(close, pd.Series) and isinstance(volume, pd.Series):
        price_change = close.diff()
        
        obv_values = pd.Series(index=close.index, dtype=float)
        obv_values.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if price_change.iloc[i] > 0:
                obv_values.iloc[i] = obv_values.iloc[i-1] + volume.iloc[i]
            elif price_change.iloc[i] < 0:
                obv_values.iloc[i] = obv_values.iloc[i-1] - volume.iloc[i]
            else:
                obv_values.iloc[i] = obv_values.iloc[i-1]
                
        return obv_values
    else:
        n = len(close)
        obv_values = np.zeros(n, dtype=float)
        obv_values[0] = volume[0]
        
        for i in range(1, n):
            if close[i] > close[i-1]:
                obv_values[i] = obv_values[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv_values[i] = obv_values[i-1] - volume[i]
            else:
                obv_values[i] = obv_values[i-1]
                
        return obv_values


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Рассчитывает все основные технические индикаторы для данного датафрейма
    
    Args:
        df: DataFrame с данными OHLCV
        
    Returns:
        DataFrame с добавленными индикаторами
    """
    # Проверка наличия необходимых колонок
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Проверяем имена колонок с учетом регистра
    df_columns = [col.lower() for col in df.columns]
    missing_columns = []
    
    for col in required_columns:
        if col not in df_columns:
            missing_columns.append(col)
    
    if missing_columns:
        raise ValueError(f"В DataFrame отсутствуют необходимые колонки: {', '.join(missing_columns)}")
    
    # Создаем копию датафрейма для результатов
    result = df.copy()
    
    # Приводим имена колонок к нижнему регистру для единообразия
    result.columns = [col.lower() for col in result.columns]
    
    # Рассчитываем скользящие средние (SMA) разных периодов
    result['sma_20'] = sma(result['close'], 20)
    result['sma_50'] = sma(result['close'], 50)
    result['sma_200'] = sma(result['close'], 200)
    
    # Рассчитываем экспоненциальные скользящие средние (EMA)
    result['ema_20'] = ema(result['close'], 20)
    result['ema_50'] = ema(result['close'], 50)
    
    # Рассчитываем RSI
    result['rsi_14'] = rsi(result['close'], 14)
    
    # Рассчитываем MACD
    macd_line, signal_line, histogram = macd(result['close'])
    result['macd'] = macd_line
    result['macd_signal'] = signal_line
    result['macd_hist'] = histogram
    
    # Рассчитываем полосы Боллинджера
    upper_band, middle_band, lower_band = bollinger_bands(result['close'])
    result['bb_upper'] = upper_band
    result['bb_middle'] = middle_band
    result['bb_lower'] = lower_band
    
    # Рассчитываем стохастический осциллятор
    k, d = stochastic_oscillator(result['high'], result['low'], result['close'])
    result['stoch_k'] = k
    result['stoch_d'] = d
    
    # Рассчитываем ATR
    result['atr_14'] = atr(result['high'], result['low'], result['close'])
    
    # Рассчитываем OBV
    result['obv'] = obv(result['close'], result['volume'])
    
    return result