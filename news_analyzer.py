"""
Модуль анализа новостей о криптовалютах
"""

import os
import pandas as pd
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Настройка логирования
logger = logging.getLogger(__name__)

class NewsAnalyzer:
    """
    Класс для сбора и анализа новостей о криптовалютах
    """
    
    def __init__(self, db_connector=None):
        """
        Инициализация анализатора новостей
        
        Args:
            db_connector: Объект для подключения к базе данных
        """
        self.db_connector = db_connector
        
        # Инициализация анализатора сентимента
        try:
            nltk.download('vader_lexicon', quiet=True)
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.error(f"Ошибка инициализации анализатора сентимента: {str(e)}")
            self.sentiment_analyzer = None
        
        # API ключи
        self.cryptopanic_api_key = os.getenv('CRYPTOPANIC_API_KEY')
        self.cryptocompare_api_key = os.getenv('CRYPTOCOMPARE_API_KEY')
        
        logger.info("Анализатор новостей инициализирован")
    
    def create_tables(self):
        """Создание необходимых таблиц в базе данных"""
        if not self.db_connector:
            logger.error("Отсутствует подключение к базе данных")
            return
            
        try:
            # Таблица для новостей
            self.db_connector.execute_query("""
                CREATE TABLE IF NOT EXISTS crypto_news (
                    id SERIAL PRIMARY KEY,
                    source VARCHAR(100),
                    title TEXT,
                    url TEXT UNIQUE,
                    published_at TIMESTAMP,
                    retrieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    content TEXT,
                    symbols TEXT[],
                    sentiment_score NUMERIC,
                    positive NUMERIC,
                    negative NUMERIC,
                    neutral NUMERIC
                )
            """)
            
            # Таблица для агрегированных данных сентимента по символам
            self.db_connector.execute_query("""
                CREATE TABLE IF NOT EXISTS crypto_sentiment (
                    symbol VARCHAR(20),
                    date DATE,
                    sentiment_score NUMERIC,
                    news_count INTEGER,
                    PRIMARY KEY (symbol, date)
                )
            """)
            
            logger.info("Таблицы для новостей успешно созданы")
        except Exception as e:
            logger.error(f"Ошибка создания таблиц: {e}")
    
    def fetch_cryptopanic_news(self, currencies: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """
        Получение новостей с CryptoPanic API
        
        Args:
            currencies: Строка с символами криптовалют через запятую (например, "BTC,ETH,SOL")
            limit: Максимальное количество новостей для получения
            
        Returns:
            Список словарей с новостями
        """
        if not self.cryptopanic_api_key:
            logger.warning("API ключ CryptoPanic не найден")
            return []
        
        try:
            url = "https://cryptopanic.com/api/v1/posts/"
            params = {
                "auth_token": self.cryptopanic_api_key,
                "kind": "news",
                "filter": "hot",
                "limit": limit
            }
            
            if currencies:
                params["currencies"] = currencies
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                news_list = data.get("results", [])
                logger.info(f"Получено {len(news_list)} новостей с CryptoPanic")
                return news_list
            else:
                logger.error(f"Ошибка API CryptoPanic: {response.status_code}, {response.text}")
                return []
        except Exception as e:
            logger.error(f"Ошибка получения новостей с CryptoPanic: {str(e)}")
            return []
    
    def fetch_cryptocompare_news(self, limit: int = 50) -> List[Dict]:
        """
        Получение новостей с CryptoCompare API
        
        Args:
            limit: Максимальное количество новостей для получения
            
        Returns:
            Список словарей с новостями
        """
        if not self.cryptocompare_api_key:
            logger.warning("API ключ CryptoCompare не найден")
            return []
        
        try:
            url = "https://min-api.cryptocompare.com/data/v2/news/"
            params = {
                "api_key": self.cryptocompare_api_key,
                "sortOrder": "popular",
                "limit": limit
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                news_list = data.get("Data", [])
                logger.info(f"Получено {len(news_list)} новостей с CryptoCompare")
                return news_list
            else:
                logger.error(f"Ошибка API CryptoCompare: {response.status_code}, {response.text}")
                return []
        except Exception as e:
            logger.error(f"Ошибка получения новостей с CryptoCompare: {str(e)}")
            return []
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Анализ сентимента текста
        
        Args:
            text: Текст для анализа
            
        Returns:
            Словарь с оценками сентимента
        """
        if not self.sentiment_analyzer or not text:
            return {"compound": 0, "pos": 0, "neu": 0, "neg": 0}
        
        try:
            sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
            return sentiment_scores
        except Exception as e:
            logger.error(f"Ошибка анализа сентимента: {str(e)}")
            return {"compound": 0, "pos": 0, "neu": 0, "neg": 0}
    
    def extract_symbols_from_text(self, text: str, title: str = "") -> List[str]:
        """
        Извлечение упоминаний криптовалют из текста
        
        Args:
            text: Основной текст
            title: Заголовок
            
        Returns:
            Список символов криптовалют
        """
        # Комбинируем заголовок и текст для анализа
        full_text = f"{title} {text}" if title else text
        
        # Список популярных криптовалют для поиска
        common_symbols = {
            "BTC": ["bitcoin", "btc"],
            "ETH": ["ethereum", "eth"],
            "SOL": ["solana", "sol"],
            "BNB": ["binance coin", "bnb"],
            "XRP": ["ripple", "xrp"],
            "ADA": ["cardano", "ada"],
            "AVAX": ["avalanche", "avax"],
            "DOT": ["polkadot", "dot"],
            "MATIC": ["polygon", "matic"],
            "DOGE": ["dogecoin", "doge"],
            "SHIB": ["shiba", "shib"]
        }
        
        found_symbols = []
        full_text_lower = full_text.lower()
        
        for symbol, keywords in common_symbols.items():
            if any(keyword in full_text_lower for keyword in keywords):
                found_symbols.append(symbol)
        
        return found_symbols
    
    def save_news_to_db(self, news: List[Dict], source: str):
        """
        Сохранение новостей в базу данных
        
        Args:
            news: Список новостей
            source: Источник новостей ('cryptopanic' или 'cryptocompare')
        """
        if not self.db_connector:
            logger.error("Отсутствует подключение к базе данных")
            return
            
        if not news:
            logger.info(f"Нет новостей для сохранения из источника {source}")
            return
        
        try:
            news_count = 0
            for item in news:
                try:
                    if source == 'cryptopanic':
                        title = item.get('title', '')
                        url = item.get('url', '')
                        published_at = item.get('published_at', '')
                        content = item.get('body', '')
                        
                        # Попытка извлечь символы из заголовка и контента
                        currencies = []
                        if 'currencies' in item and item['currencies']:
                            currencies = [c['code'] for c in item['currencies']]
                        else:
                            currencies = self.extract_symbols_from_text(content, title)
                    
                    elif source == 'cryptocompare':
                        title = item.get('title', '')
                        url = item.get('url', '')
                        published_at = datetime.fromtimestamp(item.get('published_on', 0)).isoformat()
                        content = item.get('body', '')
                        
                        # Извлекаем символы
                        categories = item.get('categories', '')
                        currencies = self.extract_symbols_from_text(content, title + " " + categories)
                    
                    else:
                        continue
                    
                    # Анализ сентимента
                    sentiment = self.analyze_sentiment(title + " " + content)
                    
                    # Проверка существования новости по URL
                    query = "SELECT id FROM crypto_news WHERE url = %s"
                    existing = self.db_connector.execute_query(query, (url,), fetch_one=True)
                    
                    if not existing:
                        # Вставка новой новости
                        query = """
                            INSERT INTO crypto_news 
                            (source, title, url, published_at, content, symbols, sentiment_score, positive, negative, neutral)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        self.db_connector.execute_query(
                            query,
                            (
                                source, title, url, published_at, content, currencies,
                                sentiment['compound'], sentiment['pos'], sentiment['neg'], sentiment['neu']
                            )
                        )
                        news_count += 1
                
                except Exception as e:
                    logger.error(f"Ошибка обработки новости: {str(e)}")
            
            logger.info(f"Сохранено {news_count} новых новостей из источника {source}")
        
        except Exception as e:
            logger.error(f"Ошибка сохранения новостей в БД: {str(e)}")
    
    def update_sentiment_aggregations(self):
        """Обновление агрегированных данных о сентименте по символам"""
        if not self.db_connector:
            logger.error("Отсутствует подключение к базе данных")
            return
            
        try:
            # Удаляем старые агрегации за последние 7 дней
            seven_days_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            query = "DELETE FROM crypto_sentiment WHERE date >= %s"
            self.db_connector.execute_query(query, (seven_days_ago,))
            
            # Создаем новые агрегации
            query = """
                INSERT INTO crypto_sentiment (symbol, date, sentiment_score, news_count)
                SELECT 
                    unnest(symbols) as symbol,
                    DATE(published_at) as date,
                    AVG(sentiment_score) as sentiment_score,
                    COUNT(*) as news_count
                FROM crypto_news
                WHERE published_at >= %s
                GROUP BY unnest(symbols), DATE(published_at)
            """
            self.db_connector.execute_query(query, (seven_days_ago,))
            
            logger.info("Агрегированные данные о сентименте успешно обновлены")
        
        except Exception as e:
            logger.error(f"Ошибка обновления агрегированных данных: {str(e)}")
    
    def get_symbol_sentiment(self, symbol: str, days: int = 7) -> Dict:
        """
        Получение данных о сентименте для конкретного символа
        
        Args:
            symbol: Символ криптовалюты
            days: Количество дней для анализа
            
        Returns:
            Словарь с данными о сентименте
        """
        if not self.db_connector:
            return {"status": "error", "message": "Отсутствует подключение к базе данных"}
            
        try:
            # Получаем агрегированные данные за указанный период
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            query = """
                SELECT 
                    date,
                    sentiment_score,
                    news_count
                FROM crypto_sentiment
                WHERE symbol = %s AND date >= %s
                ORDER BY date ASC
            """
            
            rows = self.db_connector.execute_query(query, (symbol, from_date))
            
            if not rows:
                return {
                    "status": "success",
                    "symbol": symbol,
                    "days": days,
                    "avg_sentiment": 0,
                    "news_count": 0,
                    "sentiment_trend": "neutral",
                    "daily_data": []
                }
            
            daily_data = []
            total_sentiment = 0
            total_news = 0
            
            for row in rows:
                daily_data.append({
                    "date": row[0].strftime('%Y-%m-%d'),
                    "sentiment": float(row[1]),
                    "news_count": row[2]
                })
                total_sentiment += float(row[1]) * row[2]
                total_news += row[2]
            
            # Средний сентимент
            avg_sentiment = total_sentiment / total_news if total_news > 0 else 0
            
            # Определение тренда сентимента
            sentiment_trend = "neutral"
            if len(daily_data) > 1:
                first_half = daily_data[:len(daily_data)//2]
                second_half = daily_data[len(daily_data)//2:]
                
                first_half_avg = sum(d["sentiment"] * d["news_count"] for d in first_half) / sum(d["news_count"] for d in first_half) if sum(d["news_count"] for d in first_half) > 0 else 0
                second_half_avg = sum(d["sentiment"] * d["news_count"] for d in second_half) / sum(d["news_count"] for d in second_half) if sum(d["news_count"] for d in second_half) > 0 else 0
                
                if second_half_avg > first_half_avg + 0.1:
                    sentiment_trend = "improving"
                elif second_half_avg < first_half_avg - 0.1:
                    sentiment_trend = "deteriorating"
            
            return {
                "status": "success",
                "symbol": symbol,
                "days": days,
                "avg_sentiment": float(avg_sentiment),
                "news_count": total_news,
                "sentiment_trend": sentiment_trend,
                "daily_data": daily_data
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения данных о сентименте: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def collect_and_analyze_news(self):
        """Сбор и анализ новостей из всех источников"""
        try:
            # Создаем необходимые таблицы, если они еще не существуют
            self.create_tables()
            
            # Сбор новостей из CryptoPanic
            cryptopanic_news = self.fetch_cryptopanic_news(limit=100)
            self.save_news_to_db(cryptopanic_news, 'cryptopanic')
            
            # Сбор новостей из CryptoCompare
            cryptocompare_news = self.fetch_cryptocompare_news(limit=100)
            self.save_news_to_db(cryptocompare_news, 'cryptocompare')
            
            # Обновление агрегированных данных
            self.update_sentiment_aggregations()
            
            return {
                "status": "success",
                "cryptopanic_count": len(cryptopanic_news),
                "cryptocompare_count": len(cryptocompare_news),
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Ошибка сбора и анализа новостей: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_latest_news(self, symbol: str = None, limit: int = 10) -> List[Dict]:
        """
        Получение последних новостей
        
        Args:
            symbol: Символ криптовалюты (если None, будут возвращены все новости)
            limit: Максимальное количество новостей
            
        Returns:
            Список новостей
        """
        if not self.db_connector:
            logger.error("Отсутствует подключение к базе данных")
            return []
            
        try:
            query_params = []
            query = """
                SELECT 
                    id, source, title, url, published_at, symbols, sentiment_score
                FROM crypto_news
            """
            
            if symbol:
                query += " WHERE %s = ANY(symbols)"
                query_params.append(symbol)
            
            query += " ORDER BY published_at DESC LIMIT %s"
            query_params.append(limit)
            
            rows = self.db_connector.execute_query(query, tuple(query_params))
            
            news_list = []
            for row in rows:
                news_list.append({
                    "id": row[0],
                    "source": row[1],
                    "title": row[2],
                    "url": row[3],
                    "published_at": row[4].strftime('%Y-%m-%d %H:%M:%S'),
                    "symbols": row[5],
                    "sentiment_score": float(row[6])
                })
            
            return news_list
            
        except Exception as e:
            logger.error(f"Ошибка получения последних новостей: {str(e)}")
            return []