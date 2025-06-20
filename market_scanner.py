"""
Модуль сканирования рынка для поиска торговых возможностей
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple

from .indicators import calculate_indicators, detect_patterns, detect_support_resistance

# Настройка логирования
logger = logging.getLogger(__name__)

class MarketScanner:
    """
    Класс для сканирования рынка и поиска торговых возможностей
    """
    
    def __init__(self, db_connector=None, technical_analyzer=None):
        """
        Инициализация сканера рынка
        
        Args:
            db_connector: Объект для подключения к базе данных
            technical_analyzer: Объект технического анализатора
        """
        self.db_connector = db_connector
        self.technical_analyzer = technical_analyzer
        logger.info("Сканер рынка инициализирован")
    
    def find_breakouts(self, symbols: List[str], interval: str = '60') -> List[Dict]:
        """
        Поиск пробоев уровней
        
        Args:
            symbols: Список символов
            interval: Интервал свечей
            
        Returns:
            Список обнаруженных пробоев
        """
        results = []
        
        for symbol in symbols:
            try:
                df = self.technical_analyzer.load_kline_data(symbol, interval, limit=100)
                
                if df.empty:
                    continue
                
                # Вычисляем технические индикаторы
                df = calculate_indicators(df)
                
                # Находим уровни поддержки и сопротивления
                levels = detect_support_resistance(df)
                
                current_price = df['close'].iloc[-1]
                prev_close = df['close'].iloc[-2]
                
                # Проверяем пробой уровня сопротивления
                for resistance in levels['resistance']:
                    if prev_close < resistance <= current_price:
                        results.append({
                            "symbol": symbol,
                            "type": "resistance_breakout",
                            "price": float(current_price),
                            "level": float(resistance),
                            "interval": interval,
                            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                        break
                
                # Проверяем пробой уровня поддержки
                for support in levels['support']:
                    if prev_close > support >= current_price:
                        results.append({
                            "symbol": symbol,
                            "type": "support_breakdown",
                            "price": float(current_price),
                            "level": float(support),
                            "interval": interval,
                            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                        break
                
            except Exception as e:
                logger.error(f"Ошибка при поиске пробоев для {symbol}: {str(e)}")
        
        return results
    
    def find_divergences(self, symbols: List[str], interval: str = '60') -> List[Dict]:
        """
        Поиск дивергенций между ценой и индикаторами
        
        Args:
            symbols: Список символов
            interval: Интервал свечей
            
        Returns:
            Список обнаруженных дивергенций
        """
        results = []
        
        for symbol in symbols:
            try:
                df = self.technical_analyzer.load_kline_data(symbol, interval, limit=100)
                
                if df.empty:
                    continue
                
                # Вычисляем технические индикаторы
                df = calculate_indicators(df)
                
                # Проверяем наличие достаточного количества данных
                if len(df) < 10:
                    continue
                
                # Находим локальные экстремумы цены
                price_highs = []
                price_lows = []
                
                for i in range(2, len(df) - 2):
                    if df['close'].iloc[i] > df['close'].iloc[i-1] and df['close'].iloc[i] > df['close'].iloc[i-2] and \
                       df['close'].iloc[i] > df['close'].iloc[i+1] and df['close'].iloc[i] > df['close'].iloc[i+2]:
                        price_highs.append((i, df['close'].iloc[i]))
                    
                    if df['close'].iloc[i] < df['close'].iloc[i-1] and df['close'].iloc[i] < df['close'].iloc[i-2] and \
                       df['close'].iloc[i] < df['close'].iloc[i+1] and df['close'].iloc[i] < df['close'].iloc[i+2]:
                        price_lows.append((i, df['close'].iloc[i]))
                
                # Проверяем дивергенцию с RSI
                if 'rsi_14' in df.columns and len(price_highs) >= 2 and len(price_lows) >= 2:
                    # Медвежья дивергенция: цена растет, RSI падает
                    if price_highs[-1][1] > price_highs[-2][1] and \
                       df['rsi_14'].iloc[price_highs[-1][0]] < df['rsi_14'].iloc[price_highs[-2][0]]:
                        results.append({
                            "symbol": symbol,
                            "type": "bearish_divergence",
                            "indicator": "RSI",
                            "price": float(df['close'].iloc[-1]),
                            "interval": interval,
                            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                    
                    # Бычья дивергенция: цена падает, RSI растет
                    if price_lows[-1][1] < price_lows[-2][1] and \
                       df['rsi_14'].iloc[price_lows[-1][0]] > df['rsi_14'].iloc[price_lows[-2][0]]:
                        results.append({
                            "symbol": symbol,
                            "type": "bullish_divergence",
                            "indicator": "RSI",
                            "price": float(df['close'].iloc[-1]),
                            "interval": interval,
                            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                
                # Проверяем дивергенцию с MACD
                if 'macd' in df.columns and 'macd_signal' in df.columns and len(price_highs) >= 2 and len(price_lows) >= 2:
                    # Медвежья дивергенция: цена растет, MACD падает
                    if price_highs[-1][1] > price_highs[-2][1] and \
                       df['macd'].iloc[price_highs[-1][0]] < df['macd'].iloc[price_highs[-2][0]]:
                        results.append({
                            "symbol": symbol,
                            "type": "bearish_divergence",
                            "indicator": "MACD",
                            "price": float(df['close'].iloc[-1]),
                            "interval": interval,
                            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                    
                    # Бычья дивергенция: цена падает, MACD растет
                    if price_lows[-1][1] < price_lows[-2][1] and \
                       df['macd'].iloc[price_lows[-1][0]] > df['macd'].iloc[price_lows[-2][0]]:
                        results.append({
                            "symbol": symbol,
                            "type": "bullish_divergence",
                            "indicator": "MACD",
                            "price": float(df['close'].iloc[-1]),
                            "interval": interval,
                            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
            
            except Exception as e:
                logger.error(f"Ошибка при поиске дивергенций для {symbol}: {str(e)}")
        
        return results
    
    def find_volume_spikes(self, symbols: List[str], interval: str = '60', threshold: float = 3.0) -> List[Dict]:
        """
        Поиск всплесков объема
        
        Args:
            symbols: Список символов
            interval: Интервал свечей
            threshold: Порог для определения всплеска (кратность превышения среднего объема)
            
        Returns:
            Список обнаруженных всплесков объема
        """
        results = []
        
        for symbol in symbols:
            try:
                df = self.technical_analyzer.load_kline_data(symbol, interval, limit=30)
                
                if df.empty:
                    continue
                
                # Вычисляем средний объем за последние 20 свечей (исключая последнюю)
                avg_volume = df['volume'].iloc[-21:-1].mean()
                current_volume = df['volume'].iloc[-1]
                
                # Проверка на всплеск объема
                if current_volume > avg_volume * threshold:
                    # Определяем направление движения
                    direction = "up" if df['close'].iloc[-1] > df['open'].iloc[-1] else "down"
                    
                    results.append({
                        "symbol": symbol,
                        "type": "volume_spike",
                        "direction": direction,
                        "price": float(df['close'].iloc[-1]),
                        "volume": float(current_volume),
                        "avg_volume": float(avg_volume),
                        "volume_ratio": float(current_volume / avg_volume),
                        "interval": interval,
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
            
            except Exception as e:
                logger.error(f"Ошибка при поиске всплесков объема для {symbol}: {str(e)}")
        
        return results
    
    def find_pattern_formations(self, symbols: List[str], interval: str = '60') -> List[Dict]:
        """
        Поиск свечных паттернов
        
        Args:
            symbols: Список символов
            interval: Интервал свечей
            
        Returns:
            Список обнаруженных паттернов
        """
        results = []
        
        for symbol in symbols:
            try:
                df = self.technical_analyzer.load_kline_data(symbol, interval, limit=30)
                
                if df.empty:
                    continue
                
                # Находим паттерны
                patterns = detect_patterns(df)
                
                for pattern in patterns['patterns']:
                    # Добавляем только недавние паттерны (последние 3 свечи)
                    pattern_date = datetime.strptime(pattern['date'], '%Y-%m-%d %H:%M:%S')
                    if (datetime.now() - pattern_date) < timedelta(days=1):
                        results.append({
                            "symbol": symbol,
                            "type": "pattern",
                            "pattern_name": pattern['name'],
                            "pattern_type": pattern['type'],
                            "price": float(df['close'].iloc[-1]),
                            "interval": interval,
                            "timestamp": pattern['date']
                        })
            
            except Exception as e:
                logger.error(f"Ошибка при поиске паттернов для {symbol}: {str(e)}")
        
        return results
    
    def scan_market(self, symbols: List[str], intervals: List[str] = None) -> Dict:
        """
        Комплексное сканирование рынка
        
        Args:
            symbols: Список символов
            intervals: Список интервалов для анализа
            
        Returns:
            Словарь с результатами сканирования
        """
        if intervals is None:
            intervals = ['60', '240', 'D']
        
        results = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "signals": []
        }
        
        for interval in intervals:
            # Поиск пробоев уровней
            breakouts = self.find_breakouts(symbols, interval)
            results["signals"].extend(breakouts)
            
            # Поиск дивергенций
            divergences = self.find_divergences(symbols, interval)
            results["signals"].extend(divergences)
            
            # Поиск всплесков объема
            volume_spikes = self.find_volume_spikes(symbols, interval)
            results["signals"].extend(volume_spikes)
            
            # Поиск свечных паттернов
            patterns = self.find_pattern_formations(symbols, interval)
            results["signals"].extend(patterns)
        
        # Сортировка сигналов по символам
        results["signals"].sort(key=lambda x: x["symbol"])
        
        # Группировка сигналов по символам
        symbols_signals = {}
        for signal in results["signals"]:
            symbol = signal["symbol"]
            if symbol not in symbols_signals:
                symbols_signals[symbol] = []
            
            symbols_signals[symbol].append(signal)
        
        results["symbols_signals"] = symbols_signals
        results["signals_count"] = len(results["signals"])
        
        return results