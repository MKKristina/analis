"""
Аналитический блок приложения Graffvenv.

Модуль отвечает за анализ криптовалютных данных, технические индикаторы
и машинное обучение для принятия торговых решений.
"""

from .analyzer import AnalyticalModule, TechnicalAnalyzer, MLAnalyzer
from .indicators import calculate_indicators
from .market_scanner import MarketScanner
from .news_analyzer import NewsAnalyzer
from .ml.prediction import PredictionModel

__all__ = [
    'AnalyticalModule',
    'TechnicalAnalyzer',
    'MLAnalyzer',
    'calculate_indicators',
    'MarketScanner',
    'NewsAnalyzer',
    'PredictionModel'
]