"""
Модуль для запуска анализа и интеграции с другими компонентами
"""

import argparse
import logging
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Optional

# Добавляем корневую директорию проекта в sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Импорты из других модулей проекта
from config.db_config import DBConnector
from api.api_connector import BybitAPIConnector
from analysis.analyzer import AnalyticalModule
from asset_selection.selector import AssetSelector

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analysis_runner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def save_analysis_to_file(analysis_result, filename):
    """
    Сохранение результатов анализа в JSON файл
    
    Args:
        analysis_result: Результаты анализа
        filename: Имя файла для сохранения
    """
    try:
        # Преобразование numpy типов в стандартные типы Python
        def convert_numpy_types(obj):
            import numpy as np
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return convert_numpy_types(obj.tolist())
            else:
                return obj
        
        # Конвертируем все numpy типы
        analysis_result = convert_numpy_types(analysis_result)
        
        # Сохраняем в файл
        with open(filename, 'w') as f:
            json.dump(analysis_result, f, indent=2)
            
        logger.info(f"Результаты анализа сохранены в файл {filename}")
    except Exception as e:
        logger.error(f"Ошибка сохранения результатов анализа: {str(e)}")

def run_analysis(symbols: Optional[List[str]] = None, 
                intervals: Optional[List[str]] = None,
                include_ml_prediction: bool = True,
                include_news: bool = True,
                save_results: bool = True,
                output_file: Optional[str] = None):
    """
    Запуск полного анализа рынка
    
    Args:
        symbols: Список символов для анализа
        intervals: Список интервалов для анализа
        include_ml_prediction: Включать ли ML-прогнозы
        include_news: Включать ли анализ новостей
        save_results: Сохранять ли результаты в файл
        output_file: Имя файла для сохранения результатов
    
    Returns:
        Результаты анализа
    """
    try:
        # Создаем необходимые объекты
        db_connector = DBConnector()
        api_connector = BybitAPIConnector()
        
        # Если символы не переданы, получаем их из модуля отбора активов
        if symbols is None:
            asset_selector = AssetSelector(db_connector, api_connector)
            symbols = asset_selector.get_top_symbols(limit=15)
        
        # Настройка интервалов
        if intervals is None:
            intervals = ['5', '15', '30', '60', '240', 'D', 'W', 'M']
        
        # Создание и запуск аналитического модуля
        analytical_module = AnalyticalModule(api_connector, db_connector, use_gpu=True)
        
        logger.info(f"Запуск анализа для {len(symbols)} символов, с интервалами {intervals}")
        logger.info(f"Включать ML-прогнозы: {include_ml_prediction}, включать анализ новостей: {include_news}")
        
        # Проведение анализа
        analysis_result = analytical_module.analyze_market(
            symbols=symbols,
            intervals=intervals,
            include_ml_prediction=include_ml_prediction,
            include_news_sentiment=include_news
        )
        
        # Сохранение результатов
        if save_results:
            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"analysis_result_{timestamp}.json"
            
            save_analysis_to_file(analysis_result, output_file)
        
        return analysis_result
    
    except Exception as e:
        logger.error(f"Ошибка запуска анализа: {str(e)}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Парсер аргументов командной строки
    parser = argparse.ArgumentParser(description='Запуск анализа рынка криптовалют')
    parser.add_argument('--symbols', nargs='+', help='Список символов для анализа')
    parser.add_argument('--intervals', nargs='+', help='Список интервалов для анализа')
    parser.add_argument('--no-ml', action='store_true', help='Отключить ML-прогнозы')
    parser.add_argument('--no-news', action='store_true', help='Отключить анализ новостей')
    parser.add_argument('--no-save', action='store_true', help='Не сохранять результаты в файл')
    parser.add_argument('--output', help='Имя файла для сохранения результатов')
    
    args = parser.parse_args()
    
    # Запуск анализа с переданными параметрами
    run_analysis(
        symbols=args.symbols,
        intervals=args.intervals,
        include_ml_prediction=not args.no_ml,
        include_news=not args.no_news,
        save_results=not args.no_save,
        output_file=args.output
    )