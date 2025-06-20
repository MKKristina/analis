"""
Главный модуль анализа данных
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Union, Tuple, Optional
import os
import json

# Импортируем модули для технического анализа
from analysis.technical.indicators import calculate_all_indicators
from analysis.technical.patterns import (
    detect_candle_patterns, 
    detect_chart_patterns, 
    identify_supports_resistances
)

# Импортируем модули для машинного обучения
from analysis.ml.features import extract_basic_features, extract_technical_features
from analysis.ml.preprocessing import clean_data, normalize_features
from analysis.ml.models.price_prediction import PricePredictionModel

logger = logging.getLogger(__name__)

class Analyzer:
    """
    Главный класс для анализа финансовых данных
    """
    
    def __init__(self, db_connector=None, api_connector=None, asset_selector=None,
               use_gpu=False, config_path=None):
        """
        Инициализация аналитического модуля
        
        Args:
            db_connector: Коннектор к базе данных
            api_connector: Коннектор к API биржи
            asset_selector: Объект для выбора активов
            use_gpu: Использовать ли GPU для вычислений
            config_path: Путь к файлу конфигурации
        """
        self.db_connector = db_connector
        self.api_connector = api_connector
        self.asset_selector = asset_selector
        self.use_gpu = use_gpu
        self.config = self._load_config(config_path)
        
        # Инициализируем модели
        self._initialize_models()
        
        logger.info("Аналитический модуль инициализирован")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """
        Загружает конфигурацию из файла
        
        Args:
            config_path: Путь к файлу конфигурации
            
        Returns:
            Словарь с конфигурацией
        """
        default_config = {
            "technical_analysis": {
                "enabled": True,
                "indicators": ["sma", "ema", "rsi", "macd", "bollinger_bands", "stochastic"],
                "patterns": ["candle", "chart", "support_resistance"]
            },
            "ml_analysis": {
                "enabled": True,
                "model_type": "random_forest",
                "prediction_horizon": 5,
                "feature_extraction": {
                    "windows": [5, 14, 30],
                    "use_technical_indicators": True
                }
            }
        }
        
        if not config_path or not os.path.exists(config_path):
            logger.info("Файл конфигурации не указан или не существует. Используется конфигурация по умолчанию.")
            return default_config
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"Конфигурация загружена из {config_path}")
                return config
        except Exception as e:
            logger.error(f"Ошибка при загрузке конфигурации: {str(e)}. Используется конфигурация по умолчанию.")
            return default_config
    
    def _initialize_models(self):
        """
        Инициализирует модели машинного обучения
        """
        if self.config["ml_analysis"]["enabled"]:
            # Инициализируем модель для прогнозирования цен
            self.price_predictor = PricePredictionModel(
                model_type=self.config["ml_analysis"]["model_type"]
            )
            logger.info(f"Модель прогнозирования инициализирована, устройство: {'gpu' if self.use_gpu else 'cpu'}")
    
    def analyze_symbol(self, symbol: str, timeframe: str) -> Dict:
        """
        Проводит комплексный анализ символа
        
        Args:
            symbol: Символ для анализа
            timeframe: Таймфрейм для анализа
            
        Returns:
            Словарь с результатами анализа
        """
        logger.info(f"Анализ символа {symbol} на таймфрейме {timeframe}")
        
        # Получаем данные
        df = self._get_data(symbol, timeframe)
        if df is None or len(df) == 0:
            logger.warning(f"Нет данных для {symbol} ({timeframe})")
            return {"error": "Нет данных для анализа"}
        
        # Результаты анализа
        results = {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Технический анализ
        if self.config["technical_analysis"]["enabled"]:
            tech_results = self._perform_technical_analysis(df)
            results["technical_analysis"] = tech_results
        
        # Машинное обучение для прогнозирования
        if self.config["ml_analysis"]["enabled"]:
            ml_results = self._perform_ml_analysis(df)
            results["ml_analysis"] = ml_results
        
        return results
    
    def _get_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Получает данные для анализа из базы данных или API
        
        Args:
            symbol: Символ для анализа
            timeframe: Таймфрейм для анализа
            
        Returns:
            DataFrame с данными или None при ошибке
        """
        try:
            if self.db_connector:
                # Пытаемся получить данные из БД
                query = f"""
                    SELECT * FROM klines_{symbol.lower()}_{timeframe.lower()}
                    ORDER BY timestamp ASC
                    LIMIT 200
                """
                
                df = self.db_connector.execute_query(query, fetch=True)
                
                if df is not None and len(df) > 0:
                    logger.info(f"Получено {len(df)} записей из БД для {symbol} ({timeframe})")
                    return df
            
            if self.api_connector:
                # Если данных нет в БД или их недостаточно, получаем из API
                logger.info(f"Получение данных из API для {symbol} ({timeframe})")
                
                # Предполагаем, что у api_connector есть метод get_klines
                klines = self.api_connector.get_klines(symbol, timeframe, limit=200)
                
                if klines:
                    df = pd.DataFrame(klines)
                    logger.info(f"Получено {len(df)} записей из API для {symbol} ({timeframe})")
                    return df
            
            logger.warning(f"Не удалось получить данные для {symbol} ({timeframe})")
            return None
            
        except Exception as e:
            logger.error(f"Ошибка при получении данных для {symbol} ({timeframe}): {str(e)}")
            return None
    
    def _perform_technical_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Выполняет технический анализ данных
        
        Args:
            df: DataFrame с данными OHLCV
            
        Returns:
            Словарь с результатами технического анализа
        """
        try:
            # Приводим имена колонок к нижнему регистру для единообразия
            df.columns = [col.lower() for col in df.columns]
            
            # Рассчитываем технические индикаторы
            df_with_indicators = calculate_all_indicators(df)
            
            # Определяем тип тренда на основе скользящих средних
            if 'sma_20' in df_with_indicators.columns and 'sma_50' in df_with_indicators.columns:
                current_trend = "uptrend" if df_with_indicators['sma_20'].iloc[-1] > df_with_indicators['sma_50'].iloc[-1] else "downtrend"
            else:
                current_trend = "unknown"
            
            # Определяем переукупленность/перепроданность на основе RSI
            if 'rsi_14' in df_with_indicators.columns:
                rsi_value = df_with_indicators['rsi_14'].iloc[-1]
                if rsi_value > 70:
                    rsi_signal = "overbought"
                elif rsi_value < 30:
                    rsi_signal = "oversold"
                else:
                    rsi_signal = "neutral"
            else:
                rsi_signal = "unknown"
            
            # Обнаруживаем паттерны свечей
            candle_patterns = detect_candle_patterns(df)
            
            # Выбираем активные паттерны (последние 3 свечи)
            active_candle_patterns = {}
            for pattern, values in candle_patterns.items():
                if any(values.iloc[-3:] != 0):
                    active_candle_patterns[pattern] = values.iloc[-3:].to_list()
            
            # Обнаруживаем графические паттерны
            chart_patterns = detect_chart_patterns(df)
            
            # Находим уровни поддержки и сопротивления
            support_resistance = identify_supports_resistances(df)
            
            # Определяем текущую волатильность
            if 'atr_14' in df_with_indicators.columns:
                volatility = float(df_with_indicators['atr_14'].iloc[-1] / df_with_indicators['close'].iloc[-1] * 100)
            else:
                volatility = None
            
            # Формируем сигналы для торговли
            # Пример простых правил:
            if rsi_signal == "oversold" and current_trend == "uptrend":
                trading_signal = "buy"
            elif rsi_signal == "overbought" and current_trend == "downtrend":
                trading_signal = "sell"
            else:
                trading_signal = "hold"
            
            # Собираем результаты анализа
            tech_analysis_results = {
                "trend": current_trend,
                "rsi_signal": rsi_signal,
                "active_candle_patterns": active_candle_patterns,
                "chart_patterns": chart_patterns,
                "support_levels": support_resistance["support"],
                "resistance_levels": support_resistance["resistance"],
                "volatility_percent": volatility,
                "trading_signal": trading_signal,
                "last_price": float(df['close'].iloc[-1])
            }
            
            return tech_analysis_results
            
        except Exception as e:
            logger.error(f"Ошибка при выполнении технического анализа: {str(e)}")
            return {"error": str(e)}
    
    def _perform_ml_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Выполняет анализ данных с помощью машинного обучения
        
        Args:
            df: DataFrame с данными OHLCV
            
        Returns:
            Словарь с результатами анализа
        """
        try:
            # Очищаем данные
            clean_df = clean_data(df)
            
            # Рассчитываем технические индикаторы, если их еще нет
            if 'rsi_14' not in clean_df.columns:
                clean_df = calculate_all_indicators(clean_df)
            
            # Извлекаем признаки для машинного обучения
            window_sizes = self.config["ml_analysis"]["feature_extraction"]["windows"]
            basic_features = extract_basic_features(clean_df, window_sizes=window_sizes)
            
            technical_features = extract_technical_features(
                clean_df, 
                with_indicators=True
            )
            
            # Объединяем все признаки
            all_features = pd.concat([basic_features, technical_features], axis=1)
            
            # Нормализуем признаки
            normalized_features, _ = normalize_features(all_features, method='standard')
            
            # Получаем прогноз на основе модели
            # Предполагается, что модель уже обучена
            horizon = self.config["ml_analysis"]["prediction_horizon"]
            
            # Здесь должен быть код для прогнозирования
            # Но так как модель еще не обучена, возвращаем заглушку
            
            forecast = {
                "horizon": horizon,
                "forecast_values": [float(df['close'].iloc[-1]) * (1 + np.random.normal(0, 0.01)) for _ in range(horizon)],
                "confidence": 0.75,  # Заглушка для уровня уверенности
                "forecast_direction": "up" if np.random.random() > 0.5 else "down"  # Заглушка для направления прогноза
            }
            
            return forecast
            
        except Exception as e:
            logger.error(f"Ошибка при выполнении ML анализа: {str(e)}")
            return {"error": str(e)}
    
    def analyze_market(self, symbols: List[str] = None, timeframe: str = "D") -> Dict:
        """
        Анализирует рынок по списку символов
        
        Args:
            symbols: Список символов для анализа (None - выбрать автоматически)
            timeframe: Таймфрейм для анализа
            
        Returns:
            Словарь с результатами анализа рынка
        """
        # Если символы не указаны, выбираем автоматически
        if symbols is None:
            if self.asset_selector:
                symbols = self.asset_selector.get_top_symbols(criteria='local')
                logger.info(f"Выбрано {len(symbols)} символов для анализа рынка")
            else:
                symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
                logger.warning("Asset Selector не инициализирован. Используется список символов по умолчанию.")
        
        market_results = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "timeframe": timeframe,
            "symbols_analyzed": len(symbols),
            "symbol_results": {}
        }
        
        # Анализируем каждый символ
        for symbol in symbols:
            try:
                symbol_result = self.analyze_symbol(symbol, timeframe)
                market_results["symbol_results"][symbol] = symbol_result
                logger.info(f"Завершен анализ {symbol}")
            except Exception as e:
                logger.error(f"Ошибка при анализе {symbol}: {str(e)}")
                market_results["symbol_results"][symbol] = {"error": str(e)}
        
        # Агрегируем результаты
        buy_signals = [symbol for symbol, result in market_results["symbol_results"].items() 
                    if "technical_analysis" in result and 
                    result["technical_analysis"].get("trading_signal") == "buy"]
        
        sell_signals = [symbol for symbol, result in market_results["symbol_results"].items() 
                     if "technical_analysis" in result and 
                     result["technical_analysis"].get("trading_signal") == "sell"]
        
        market_results["summary"] = {
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "market_sentiment": "bullish" if len(buy_signals) > len(sell_signals) else "bearish"
        }
        
        return market_results