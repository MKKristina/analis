"""
Модуль для расчета технических индикаторов.
Самостоятельная реализация без pandas_ta.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, Dict, List


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
        
        # Расчет True Range
        # Для первой точки используем только high-low
        tr[0] = high[0] - low[0]
        
        # Для остальных точек находим максимум из трех вариантов
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], 
                        abs(high[i] - close[i-1]), 
                        abs(low[i] - close[i-1]))
        
        # Расчет ATR
        atr_values = np.full_like(close, np.nan, dtype=float)
        
        # Инициализация первого значения ATR
        if n >= period:
            atr_values[period-1] = np.mean(tr[:period])
        
        # Расчет последующих значений по формуле Wilder
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


def adx(high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray], 
       close: Union[pd.Series, np.ndarray], period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Индекс направленного движения (Average Directional Index)
    
    Args:
        high: Массив максимальных цен
        low: Массив минимальных цен
        close: Массив цен закрытия
        period: Период расчета
        
    Returns:
        Кортеж (ADX, +DI, -DI)
    """
    if isinstance(high, pd.Series) and isinstance(low, pd.Series) and isinstance(close, pd.Series):
        # Создаем DataFrame для хранения результатов
        df = pd.DataFrame({
            'high': high,
            'low': low,
            'close': close
        })
        
        # Рассчитываем True Range
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = (df['high'] - df['close'].shift(1)).abs()
        df['tr3'] = (df['low'] - df['close'].shift(1)).abs()
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Рассчитываем +DM и -DM
        df['high_diff'] = df['high'].diff()
        df['low_diff'] = df['low'].diff()
        
        # +DM: Если current high - previous high > previous low - current low и > 0
        df['plus_dm'] = np.where(
            (df['high_diff'] > df['low_diff'].abs()) & (df['high_diff'] > 0),
            df['high_diff'],
            0
        )
        
        # -DM: Если previous low - current low > current high - previous high и > 0
        df['minus_dm'] = np.where(
            (df['low_diff'].abs() > df['high_diff']) & (df['low_diff'] < 0),
            df['low_diff'].abs(),
            0
        )
        
        # Рассчитываем скользящие средние для TR, +DM, -DM
        df['tr_ma'] = df['tr'].rolling(window=period).mean()
        df['plus_dm_ma'] = df['plus_dm'].rolling(window=period).mean()
        df['minus_dm_ma'] = df['minus_dm'].rolling(window=period).mean()
        
        # Рассчитываем +DI и -DI
        df['plus_di'] = 100 * df['plus_dm_ma'] / df['tr_ma']
        df['minus_di'] = 100 * df['minus_dm_ma'] / df['tr_ma']
        
        # Рассчитываем DX и ADX
        df['dx'] = 100 * (df['plus_di'] - df['minus_di']).abs() / (df['plus_di'] + df['minus_di'])
        df['adx'] = df['dx'].rolling(window=period).mean()
        
        return df['adx'].values, df['plus_di'].values, df['minus_di'].values
    else:
        n = len(high)
        tr = np.zeros(n, dtype=float)
        plus_dm = np.zeros(n, dtype=float)
        minus_dm = np.zeros(n, dtype=float)
        
        # Рассчитываем TR, +DM, -DM
        tr[0] = high[0] - low[0]
        
        for i in range(1, n):
            # True Range
            tr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i-1]),
                        abs(low[i] - close[i-1]))
            
            # +DM и -DM
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            
            if high_diff > 0 and high_diff > low_diff:
                plus_dm[i] = high_diff
            else:
                plus_dm[i] = 0
                
            if low_diff > 0 and low_diff > high_diff:
                minus_dm[i] = low_diff
            else:
                minus_dm[i] = 0
        
        # Сглаживаем TR, +DM, -DM
        tr_ma = np.full_like(tr, np.nan, dtype=float)
        plus_dm_ma = np.full_like(plus_dm, np.nan, dtype=float)
        minus_dm_ma = np.full_like(minus_dm, np.nan, dtype=float)
        
        # Инициализируем первые значения скользящих средних
        if n >= period:
            tr_ma[period-1] = np.mean(tr[:period])
            plus_dm_ma[period-1] = np.mean(plus_dm[:period])
            minus_dm_ma[period-1] = np.mean(minus_dm[:period])
            
            # Рассчитываем скользящие средние по формуле Wilder
            for i in range(period, n):
                tr_ma[i] = (tr_ma[i-1] * (period - 1) + tr[i]) / period
                plus_dm_ma[i] = (plus_dm_ma[i-1] * (period - 1) + plus_dm[i]) / period
                minus_dm_ma[i] = (minus_dm_ma[i-1] * (period - 1) + minus_dm[i]) / period
        
        # Рассчитываем +DI и -DI
        plus_di = np.zeros(n, dtype=float)
        minus_di = np.zeros(n, dtype=float)
        
        for i in range(period, n):
            if tr_ma[i] > 0:
                plus_di[i] = 100 * plus_dm_ma[i] / tr_ma[i]
                minus_di[i] = 100 * minus_dm_ma[i] / tr_ma[i]
        
        # Рассчитываем DX
        dx = np.zeros(n, dtype=float)
        for i in range(period, n):
            if (plus_di[i] + minus_di[i]) > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        # Рассчитываем ADX
        adx = np.zeros(n, dtype=float)
        
        # Первое значение ADX - среднее из первых period значений DX
        if n >= 2*period:
            adx[2*period-1] = np.mean(dx[period:2*period])
            
            # Последующие значения ADX
            for i in range(2*period, n):
                adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        
        return adx, plus_di, minus_di


def cci(high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray], 
       close: Union[pd.Series, np.ndarray], period: int = 20, constant: float = 0.015) -> np.ndarray:
    """
    Индекс товарного канала (Commodity Channel Index)
    
    Args:
        high: Массив максимальных цен
        low: Массив минимальных цен
        close: Массив цен закрытия
        period: Период расчета
        constant: Константа (обычно 0.015)
        
    Returns:
        Массив значений CCI
    """
    if isinstance(high, pd.Series) and isinstance(low, pd.Series) and isinstance(close, pd.Series):
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        
        # Вычисляем среднее отклонение
        mad = pd.Series(index=typical_price.index, dtype=float)
        for i in range(period - 1, len(typical_price)):
            mad.iloc[i] = abs(typical_price.iloc[i-period+1:i+1] - sma_tp.iloc[i]).mean()
        
        # Вычисляем CCI
        cci_values = (typical_price - sma_tp) / (constant * mad)
        
        return cci_values
    else:
        n = len(high)
        typical_price = (high + low + close) / 3
        
        # Среднее типичной цены
        sma_tp = np.full_like(typical_price, np.nan, dtype=float)
        for i in range(period - 1, n):
            sma_tp[i] = np.mean(typical_price[i-period+1:i+1])
        
        # Среднее абсолютное отклонение
        mad = np.full_like(typical_price, np.nan, dtype=float)
        for i in range(period - 1, n):
            mad[i] = np.mean(np.abs(typical_price[i-period+1:i+1] - sma_tp[i]))
        
        # CCI
        cci_values = np.full_like(typical_price, np.nan, dtype=float)
        for i in range(period - 1, n):
            if mad[i] > 0:
                cci_values[i] = (typical_price[i] - sma_tp[i]) / (constant * mad[i])
        
        return cci_values


def wma(data: Union[pd.Series, np.ndarray], period: int) -> Union[pd.Series, np.ndarray]:
    """
    Взвешенная скользящая средняя (Weighted Moving Average)
    
    Args:
        data: Массив цен закрытия
        period: Период взвешенной скользящей средней
        
    Returns:
        Массив значений WMA
    """
    weights = np.arange(1, period + 1)
    sum_weights = np.sum(weights)
    
    if isinstance(data, pd.Series):
        wma_values = data.rolling(window=period).apply(
            lambda x: np.sum(x * weights[-len(x):]) / np.sum(weights[-len(x):]),
            raw=True
        )
        return wma_values
    else:
        n = len(data)
        wma_values = np.full_like(data, np.nan, dtype=float)
        
        for i in range(period - 1, n):
            wma_values[i] = np.sum(data[i-period+1:i+1] * weights) / sum_weights
            
        return wma_values


def hull_ma(data: Union[pd.Series, np.ndarray], period: int) -> Union[pd.Series, np.ndarray]:
    """
    Скользящая средняя Халла (Hull Moving Average)
    
    Args:
        data: Массив цен закрытия
        period: Период HMA
        
    Returns:
        Массив значений Hull MA
    """
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # Рассчитываем WMA с периодом n
    wma_n = wma(data, period)
    
    # Рассчитываем WMA с периодом n/2
    wma_half = wma(data, half_period)
    
    # 2*WMA(n/2) - WMA(n)
    raw_hma = 2 * wma_half - wma_n
    
    # Финальный WMA с периодом sqrt(n)
    hma = wma(raw_hma, sqrt_period)
    
    return hma


def pivot_points(high: Union[pd.Series, np.ndarray], 
                low: Union[pd.Series, np.ndarray], 
                close: Union[pd.Series, np.ndarray], 
                method: str = 'standard') -> Dict[str, float]:
    """
    Расчет точек разворота (Pivot Points)
    
    Args:
        high: Максимальная цена предыдущего периода
        low: Минимальная цена предыдущего периода
        close: Цена закрытия предыдущего периода
        method: Метод расчета ('standard', 'fibonacci', 'woodie', 'camarilla', 'demark')
        
    Returns:
        Словарь с точками разворота
    """
    # Используем последние значения для расчета
    if isinstance(high, pd.Series):
        h = high.iloc[-1]
        l = low.iloc[-1]
        c = close.iloc[-1]
    else:
        h = high[-1]
        l = low[-1]
        c = close[-1]
    
    result = {}
    
    if method == 'standard':
        # Стандартный метод
        p = (h + l + c) / 3  # Pivot point
        r1 = 2 * p - l       # Resistance 1
        s1 = 2 * p - h       # Support 1
        r2 = p + (h - l)     # Resistance 2
        s2 = p - (h - l)     # Support 2
        r3 = h + 2 * (p - l) # Resistance 3
        s3 = l - 2 * (h - p) # Support 3
        
        result = {'P': p, 'R1': r1, 'S1': s1, 'R2': r2, 'S2': s2, 'R3': r3, 'S3': s3}
        
    elif method == 'fibonacci':
        # Метод Фибоначчи
        p = (h + l + c) / 3
        r1 = p + 0.382 * (h - l)
        s1 = p - 0.382 * (h - l)
        r2 = p + 0.618 * (h - l)
        s2 = p - 0.618 * (h - l)
        r3 = p + 1.000 * (h - l)
        s3 = p - 1.000 * (h - l)
        
        result = {'P': p, 'R1': r1, 'S1': s1, 'R2': r2, 'S2': s2, 'R3': r3, 'S3': s3}
        
    elif method == 'woodie':
        # Метод Вуди
        p = (h + l + 2 * c) / 4
        r1 = 2 * p - l
        s1 = 2 * p - h
        r2 = p + (h - l)
        s2 = p - (h - l)
        
        result = {'P': p, 'R1': r1, 'S1': s1, 'R2': r2, 'S2': s2}
        
    elif method == 'camarilla':
        # Метод Камарилла
        p = (h + l + c) / 3
        r1 = c + (h - l) * 1.1 / 12
        s1 = c - (h - l) * 1.1 / 12
        r2 = c + (h - l) * 1.1 / 6
        s2 = c - (h - l) * 1.1 / 6
        r3 = c + (h - l) * 1.1 / 4
        s3 = c - (h - l) * 1.1 / 4
        r4 = c + (h - l) * 1.1 / 2
        s4 = c - (h - l) * 1.1 / 2
        
        result = {'P': p, 'R1': r1, 'S1': s1, 'R2': r2, 'S2': s2, 
                  'R3': r3, 'S3': s3, 'R4': r4, 'S4': s4}
        
    elif method == 'demark':
        # Метод Демарка
        if c < c:  # Сегодняшнее закрытие ниже вчерашнего
            x = h + 2 * l + c
        elif c > c:  # Сегодняшнее закрытие выше вчерашнего
            x = 2 * h + l + c
        else:  # Сегодняшнее закрытие равно вчерашнему
            x = h + l + 2 * c
            
        p = x / 4
        r1 = x / 2 - l
        s1 = x / 2 - h
        
        result = {'P': p, 'R1': r1, 'S1': s1}
    
    return result


def ichimoku(high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray], 
            conversion_period: int = 9, base_period: int = 26, 
            lagging_span2_period: int = 52, displacement: int = 26) -> Dict[str, np.ndarray]:
    """
    Ичимоку Кинко Хё (Ichimoku Kinko Hyo)
    
    Args:
        high: Массив максимальных цен
        low: Массив минимальных цен
        conversion_period: Период линии Tenkan-sen (Conversion Line)
        base_period: Период линии Kijun-sen (Base Line)
        lagging_span2_period: Период линии Senkou Span B (Leading Span B)
        displacement: Смещение для расчета Senkou Span A и B
        
    Returns:
        Словарь с компонентами Ичимоку
    """
    if isinstance(high, pd.Series) and isinstance(low, pd.Series):
        # Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 for the past 9 periods
        tenkan_sen = (high.rolling(window=conversion_period).max() + 
                     low.rolling(window=conversion_period).min()) / 2
        
        # Kijun-sen (Base Line): (highest high + lowest low) / 2 for the past 26 periods
        kijun_sen = (high.rolling(window=base_period).max() + 
                    low.rolling(window=base_period).min()) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, plotted 26 periods ahead
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
        
        # Senkou Span B (Leading Span B): (highest high + lowest low) / 2 for the past 52 periods, plotted 26 periods ahead
        senkou_span_b = ((high.rolling(window=lagging_span2_period).max() + 
                         low.rolling(window=lagging_span2_period).min()) / 2).shift(displacement)
        
        # Chikou Span (Lagging Span): Close price, plotted 26 periods behind
        chikou_span = high.shift(-displacement)
        
        return {
            'tenkan_sen': tenkan_sen.values,
            'kijun_sen': kijun_sen.values,
            'senkou_span_a': senkou_span_a.values,
            'senkou_span_b': senkou_span_b.values,
            'chikou_span': chikou_span.values
        }
    else:
        n = len(high)
        tenkan_sen = np.full(n, np.nan, dtype=float)
        kijun_sen = np.full(n, np.nan, dtype=float)
        senkou_span_a = np.full(n, np.nan, dtype=float)
        senkou_span_b = np.full(n, np.nan, dtype=float)
        chikou_span = np.full(n, np.nan, dtype=float)
        
        # Tenkan-sen
        for i in range(conversion_period - 1, n):
            tenkan_sen[i] = (np.max(high[i-conversion_period+1:i+1]) + 
                            np.min(low[i-conversion_period+1:i+1])) / 2
        
        # Kijun-sen
        for i in range(base_period - 1, n):
            kijun_sen[i] = (np.max(high[i-base_period+1:i+1]) + 
                           np.min(low[i-base_period+1:i+1])) / 2
        
        # Senkou Span A
        for i in range(base_period - 1, n):
            if i + displacement < n:
                senkou_span_a[i + displacement] = (tenkan_sen[i] + kijun_sen[i]) / 2
        
        # Senkou Span B
        for i in range(lagging_span2_period - 1, n):
            if i + displacement < n:
                senkou_span_b[i + displacement] = (np.max(high[i-lagging_span2_period+1:i+1]) + 
                                                 np.min(low[i-lagging_span2_period+1:i+1])) / 2
        
        # Chikou Span
        for i in range(n):
            if i - displacement >= 0:
                chikou_span[i - displacement] = high[i]
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }


def donchian_channel(high: Union[pd.Series, np.ndarray], 
                    low: Union[pd.Series, np.ndarray], 
                    period: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Канал Дончиана (Donchian Channel)
    
    Args:
        high: Массив максимальных цен
        low: Массив минимальных цен
        period: Период канала
        
    Returns:
        Кортеж (верхняя линия, средняя линия, нижняя линия)
    """
    if isinstance(high, pd.Series) and isinstance(low, pd.Series):
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()
        middle = (upper + lower) / 2
        
        return upper.values, middle.values, lower.values
    else:
        n = len(high)
        upper = np.full(n, np.nan, dtype=float)
        lower = np.full(n, np.nan, dtype=float)
        middle = np.full(n, np.nan, dtype=float)
        
        for i in range(period - 1, n):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
            middle[i] = (upper[i] + lower[i]) / 2
            
        return upper, middle, lower


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
    
    # Рассчитываем взвешенную скользящую среднюю (WMA)
    result['wma_20'] = wma(result['close'], 20)
    
    # Рассчитываем скользящую среднюю Халла (HMA)
    result['hma_20'] = hull_ma(result['close'], 20)
    
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
    
    # Рассчитываем ADX, +DI, -DI
    adx_values, plus_di, minus_di = adx(result['high'], result['low'], result['close'])
    result['adx'] = adx_values
    result['plus_di'] = plus_di
    result['minus_di'] = minus_di
    
    # Рассчитываем CCI
    result['cci'] = cci(result['high'], result['low'], result['close'])
    
    # Канал Дончиана
    upper, middle, lower = donchian_channel(result['high'], result['low'])
    result['donchian_upper'] = upper
    result['donchian_middle'] = middle
    result['donchian_lower'] = lower
    
    return result