import hashlib

from moexalgo import Ticker
from config import MOEX_API_KEY, TICKERS, NEWS_API_KEY, COMPANY_NAMES
import pandas as pd
import json
from datetime import datetime, timedelta
import os
from newsapi import NewsApiClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
from typing import List, Dict
import feedparser
import requests
# ===================== Инициализация =====================
newsapi = NewsApiClient(api_key=NEWS_API_KEY)
Ticker.TOKEN = MOEX_API_KEY

# Загрузка модели и токенизатора
tokenizer = AutoTokenizer.from_pretrained("blanchefort/rubert-base-cased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("blanchefort/rubert-base-cased-sentiment")

# Даты
today = datetime.today().strftime('%Y-%m-%d')
thirty_days_ago = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')

# Пути
os.makedirs('data', exist_ok=True)
os.makedirs('news', exist_ok=True)
HISTORICAL_NEWS_PATH = 'news/Lenta_20_23.csv'
SENTIMENT_CACHE_PATH = 'news/sentiment_cache.json'
DATASET_CACHE_PATH = lambda ticker: f"data/{ticker}_ml_ready.csv"


# ===================== Функции =====================
def load_sentiment_cache():
    """Загрузка кэша тональности"""
    if os.path.exists(SENTIMENT_CACHE_PATH):
        with open(SENTIMENT_CACHE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_sentiment_cache(cache):
    """Сохранение кэша тональности"""
    with open(SENTIMENT_CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=4, ensure_ascii=False)

def analyze_sentiment(text: str) -> str:
    """Анализ тональности текста с использованием RuBERT"""
    if not text or len(text.strip()) == 0:
        return "neutral"
    sentiment_cache = load_sentiment_cache()
    import hashlib
    cache_key = hashlib.sha256(text.encode()).hexdigest()
    if cache_key in sentiment_cache:
        return sentiment_cache[cache_key]

    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        label = torch.argmax(probs).item()
        sentiment_labels = ['negative', 'neutral', 'positive']
        sentiment = sentiment_labels[label]
        sentiment_cache[cache_key] = sentiment
        save_sentiment_cache(sentiment_cache)
        return sentiment
    except Exception as e:
        print(f"❌ Ошибка при анализе тональности: {e}")
        return "neutral"

def fetch_news(ticker_name: str) -> list:
    """Получение свежих новостей по тикеру (последние 30 дней)"""
    try:
        all_articles = newsapi.get_everything(
            q=f"{ticker_name} OR {COMPANY_NAMES.get(ticker_name, ticker_name)}",
            from_param=thirty_days_ago,
            to=today,
            language='ru',
            sort_by='publishedAt',
            page_size=100
        )
        if all_articles['status'] == 'error':
            print(f"⚠️ NewsAPI error: {all_articles['message']}")
            if 'rate limit' in all_articles['message'].lower():
                print("⏳ Ожидание сброса лимита...")
                time.sleep(60)
            return []

        sentiment_cache = load_sentiment_cache()
        articles = []
        for article in all_articles.get('articles', []):
            title = article.get('title', '')
            description = article.get('description', '')
            content = f"{title}. {description}" if description else title
            published_at = article.get('publishedAt', '')
            if not published_at:
                continue

            news_date = published_at[:10]
            if news_date < thirty_days_ago or news_date > today:
                continue

            cache_key = hashlib.sha256(content.encode()).hexdigest()
            sentiment = sentiment_cache.get(cache_key, analyze_sentiment(content))
            if cache_key not in sentiment_cache:
                sentiment_cache[cache_key] = sentiment
                save_sentiment_cache(sentiment_cache)

            articles.append({
                'title': title,
                'description': description,
                'publishedAt': news_date,
                'sentiment': sentiment
            })
        print(f"📰 Получено {len(articles)} новостей для {ticker_name}")
        return articles
    except Exception as e:
        print(f"❌ Ошибка при получении новостей для {ticker_name}: {str(e)}")
        return []

def load_historical_news(ticker_name: str) -> list:
    """Загрузка исторических новостей из CSV"""
    try:
        df = pd.read_csv(HISTORICAL_NEWS_PATH, on_bad_lines='skip')
        required_cols = ['title', 'text', 'pubdate']
        if not all(col in df.columns for col in required_cols):
            raise KeyError(f"В CSV-файле отсутствуют необходимые колонки: {required_cols}")

        df['pubdate'] = pd.to_datetime(df['pubdate'], unit='s').dt.date
        today_date = datetime.strptime(today, '%Y-%m-%d').date()
        df = df[df['pubdate'] <= today_date]

        sentiment_cache = load_sentiment_cache()
        keywords = [ticker_name, COMPANY_NAMES.get(ticker_name, ticker_name)]
        filtered_news = []

        for _, row in df.iterrows():
            title = row['title']
            text = row.get('text', '')
            content = f"{title}. {text}" if text else title
            news_date = row['pubdate']
            valid_keywords = [kw for kw in keywords if isinstance(kw, str)]

            if any(kw.lower() in content.lower() for kw in valid_keywords):
                cache_key = hashlib.sha256(content.encode()).hexdigest()
                sentiment = sentiment_cache.get(cache_key, analyze_sentiment(content))
                if cache_key not in sentiment_cache:
                    sentiment_cache[cache_key] = sentiment
                    save_sentiment_cache(sentiment_cache)

                filtered_news.append({
                    'title': title,
                    'description': text[:200] if text else '',
                    'publishedAt': news_date,
                    'sentiment': sentiment
                })
        print(f"📰 Загружено {len(filtered_news)} исторических новостей для {ticker_name}")
        return filtered_news
    except Exception as e:
        print(f"❌ Ошибка при загрузке исторических новостей: {str(e)}")
        return []

def combine_news(historical_news: list, recent_news: list) -> list:
    """Объединение исторических и свежих новостей"""
    return historical_news + recent_news

# ===================== Функции для ценовых данных =====================
def load_cached_dataset(ticker_name: str) -> pd.DataFrame:
    """Загрузка кэшированного датасета"""
    dataset_path = DATASET_CACHE_PATH(ticker_name)
    if os.path.exists(dataset_path):
        print(f"🧠 Используем кэшированный датасет для {ticker_name}")
        df = pd.read_csv(dataset_path)
        df['date'] = pd.to_datetime(df['date']).dt.date
        cols_to_drop = [col for col in df.columns if col.endswith('_x') or col.endswith('_y')]
        if cols_to_drop:
            print(f"🧹 Удаляем старые колонки: {cols_to_drop}")
        #             df = df.drop(columns=cols_to_drop)
        return df
    return pd.DataFrame()


def get_last_date(df: pd.DataFrame) -> str:
    """Получение последней даты из кэшированного датасета"""
    if not df.empty:
        return df['date'].max().strftime('%Y-%m-%d')
    return '2020-01-01'


def update_price_data(df_cached: pd.DataFrame, ticker_name: str, last_date: str) -> pd.DataFrame:
    """Обновление ценовых данных с помощью цикла, пока не дойдём до сегодня"""
    ticker = Ticker(ticker_name)
    current_start = last_date
    all_new_candles = pd.DataFrame()
    max_iterations = 100
    iteration = 0

    while current_start < today and iteration < max_iterations:
        try:
            # Запрашиваем данные
            candles = ticker.candles(start=current_start, end=today)
            if candles.empty:
                print(f"❌ Нет новых данных после {current_start}")
                # Двигаем дату вперёд на 1 день, чтобы не застрять
                current_start_dt = datetime.strptime(current_start, '%Y-%m-%d') + timedelta(days=1)
                current_start = current_start_dt.strftime('%Y-%m-%d')
                iteration += 1
                continue

            # Обработка свечей
            candles['begin'] = pd.to_datetime(candles['begin'])
            candles['date'] = candles['begin'].dt.date
            candles = candles.drop_duplicates(subset='date', keep='last')
            candles = candles[['date', 'open', 'high', 'low', 'close', 'volume']]
            candles.columns = ['date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']

            # Добавляем новые данные
            all_new_candles = pd.concat([all_new_candles, candles], ignore_index=True)

            # Определяем последнюю дату в полученных данных
            last_fetched_date = candles['date'].max()

            # Следующая дата — на день после последней полученной
            next_start_dt = last_fetched_date + timedelta(days=1)
            current_start = next_start_dt.strftime('%Y-%m-%d')

            # Если достигли сегодня — выходим
            if last_fetched_date >= datetime.strptime(today, '%Y-%m-%d').date():
                break

            iteration += 1

        except Exception as e:
            print(f"❌ Ошибка при запросе данных для {ticker_name} с {current_start}: {e}")
            break

    if all_new_candles.empty:
        print(f"❌ Не удалось получить новые ценовые данные для {ticker_name}")
        return df_cached

    # Очистка
    all_new_candles.drop_duplicates(subset='date', keep='last', inplace=True)
    all_new_candles.sort_values('date', inplace=True)

    print(f"📈 Получено {len(all_new_candles)} новых свечей для {ticker_name}")

    # Объединение с кэшем
    if not df_cached.empty:
        last_date_dt = datetime.strptime(last_date, '%Y-%m-%d').date()
        df_filtered = df_cached[df_cached['date'] <= last_date_dt]
        updated_df = pd.concat([df_filtered, all_new_candles], ignore_index=True)
    else:
        updated_df = all_new_candles.copy()

    # Финальная очистка
    updated_df['date'] = pd.to_datetime(updated_df['date']).dt.date
    updated_df.drop_duplicates(subset='date', keep='last', inplace=True)
    updated_df.sort_values('date', inplace=True)

    return updated_df


# ===================== Функции для фичей =====================
def add_lagged_features(df: pd.DataFrame, lag: int = 5) -> pd.DataFrame:
    df = df.copy()
    for i in range(1, lag + 1):
        df[f'close_lag_{i}'] = df['close_price'].shift(i)
    # Не удаляем строки, если данных мало
    if len(df) > 20:  # Хватает для rolling и lag
        df.dropna(inplace=True)
    else:
        df.fillna(0, inplace=True)  # Заполняем нулями
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Добавление технических индикаторов"""
    df['SMA_5'] = df['close_price'].rolling(window=5).mean()
    df['SMA_10'] = df['close_price'].rolling(window=10).mean()
    df['RSI'] = compute_rsi(df['close_price'], window=14)
    df.dropna(inplace=True)
    return df


def compute_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """Вычисление RSI"""
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Создание целевой переменной: доходность за 15 дней в %"""
    df['target_return'] = (df['close_price'].shift(-15) - df['close_price']) / df['close_price'] * 100
    df.dropna(subset=['target_return'], inplace=True)
    return df


def save_to_csv(df: pd.DataFrame, ticker_name: str) -> None:
    """Сохранение данных в CSV"""
    df.to_csv(DATASET_CACHE_PATH(ticker_name), index=False)
    print(f"✅ Данные для {ticker_name} обновлены и сохранены в {DATASET_CACHE_PATH(ticker_name)}")

# Заголовок, чтобы не выглядеть как бот
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36'
}

def fetch_rbc_news(ticker_name: str) -> List[Dict]:
    """Парсит RSS РБК с использованием requests"""
    feed_url = "https://rssexport.rbc.ru/rbcnews/news/30/full.rss"
    return _parse_rss_with_requests(feed_url, ticker_name, "РБК")

def fetch_interfax_news(ticker_name: str) -> List[Dict]:
    """Парсит RSS Интерфакса с использованием requests"""
    feed_url = "https://www.interfax.ru/rss.asp"
    return _parse_rss_with_requests(feed_url, ticker_name, "Интерфакс")


def _parse_rss_with_requests(feed_url: str, ticker_name: str, source: str) -> List[Dict]:
    """Парсинг RSS через requests + feedparser с обработкой ошибок"""
    feed_url = feed_url.strip()

    try:
        response = requests.get(feed_url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding

        feed = feedparser.parse(response.content)

        if feed.bozo:
            print(f"⚠️ Предупреждение при парсинге {source}: {feed.bozo_exception}")

        articles = []
        # Получаем название компании - может быть строкой или списком
        company_name = COMPANY_NAMES.get(ticker_name, ticker_name)

        # Формируем список ключевых слов
        keywords = [ticker_name.lower()]

        # Обрабатываем разные типы company_name
        if isinstance(company_name, str):
            # Если это строка, добавляем варианты
            keywords.append(company_name.lower())
            keywords.append(f"акции {company_name}".lower())
            keywords.append(f"{company_name} курс".lower())
            keywords.append(f"{company_name} дивиденды".lower())
        elif isinstance(company_name, list):
            # Если это список, обрабатываем каждый элемент
            for name in company_name:
                if isinstance(name, str):
                    keywords.append(name.lower())
                    keywords.append(f"акции {name}".lower())
                    keywords.append(f"{name} курс".lower())
                    keywords.append(f"{name} дивиденды".lower())
        # # Для GAZP добавляем "газпром" как ключевое слово
        # if ticker_name == "GAZP":
        #     keywords.append("газпром")
        #     keywords.append("акции газпром")
        #     keywords.append("газпром курс")
        #     keywords.append("газпром дивиденды")

        seen_titles = set()

        def to_string(value):
            if value is None:
                return ''
            if isinstance(value, str):
                return value
            if isinstance(value, (list, tuple)):
                return ' '.join(to_string(item) for item in value)
            if isinstance(value, dict):
                return ' '.join(to_string(v) for v in value.values())
            return str(value)

        for entry in feed.entries:
            title = to_string(entry.get('title', ''))
            summary = to_string(entry.get('summary', '') or entry.get('description', ''))
            published = to_string(entry.get('published', ''))
            link = to_string(entry.get('link', ''))

            if not title.strip():
                continue

            if title in seen_titles:
                continue

            # Приводим к нижнему регистру для сравнения
            content_lower = (title + ' ' + summary).lower()

            # Проверяем наличие любого из ключевых слов
            if any(kw in content_lower for kw in keywords):
                sentiment = analyze_sentiment(title + " " + summary)
                articles.append({
                    'title': title,
                    'description': summary[:200],
                    'publishedAt': published,
                    'url': link,
                    'sentiment': sentiment,
                    'source': source
                })
                seen_titles.add(title)

        print(f"📰 Получено {len(articles)} новостей из {source} для {ticker_name}")
        return articles

    except Exception as e:
        print(f"❌ Непредвиденная ошибка при парсинге {source}: {e}")
        import traceback
        traceback.print_exc()
        return []

# ===================== Основной процесс =====================
for ticker_name in TICKERS:
    print(f"\n🔄 Обновляем данные для {ticker_name}...")

    # 1. Загрузка кэшированного датасета
    cached_df = load_cached_dataset(ticker_name)
    last_date = get_last_date(cached_df)
    print(f"⏳ Загружаем данные с {last_date} по {today}")

    # 2. Обновление ценовых данных
    price_df = update_price_data(cached_df, ticker_name, last_date)

    # 3. Обновление новостей
    historical_news = load_historical_news(ticker_name)

    # 🔥 Заменяем NewsAPI на RSS-парсеры
    print(f"📡 Получаем свежие новости для {ticker_name} из РБК и Интерфакса...")
    rbc_news = fetch_rbc_news(ticker_name)
    interfax_news = fetch_interfax_news(ticker_name)

    # Объединяем все свежие новости
    recent_news = rbc_news + interfax_news

    # Объединяем с историческими
    combined_news = combine_news(historical_news, recent_news)

    # 4. Объединение данных: объединяем ценовые данные с новостями
    if not combined_news:
        print("📰 Нет новостей для объединения, используем только ценовые данные")
        merged_df = price_df.copy()
        # Явно добавляем колонки с правильным типом
        merged_df['sentiment_avg'] = 0.0
        merged_df['positive_news'] = 0.0
        merged_df['neutral_news'] = 0.0
        merged_df['negative_news'] = 0.0
    else:
        # Создаём DataFrame из новостей
        news_df = pd.DataFrame(combined_news)

        # Обработка временных зон
        news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'], errors='coerce', utc=True)
        news_df['publishedAt'] = news_df['publishedAt'].dt.tz_localize(None)
        news_df['date'] = news_df['publishedAt'].dt.date

        # Преобразуем тональность в числовую метрику
        sentiment_map = {'negative': -1, 'neutral': 0, 'positive': 1}
        news_df['sentiment_num'] = news_df['sentiment'].map(sentiment_map)

        # Агрегируем по датам
        daily_sentiment = news_df.groupby('date').agg(
            sentiment_avg=('sentiment_num', 'mean'),
            positive_news=('sentiment', lambda x: (x == 'positive').sum()),
            neutral_news=('sentiment', lambda x: (x == 'neutral').sum()),
            negative_news=('sentiment', lambda x: (x == 'negative').sum())
        ).reset_index()

        # Объединяем с ценовыми данными
        merged_df = pd.merge(price_df, daily_sentiment, on='date', how='left')

        # 🔑 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: явное преобразование типов
        for col in ['sentiment_avg', 'positive_news', 'neutral_news', 'negative_news']:
            if col in merged_df.columns:
                # Сначала заполняем NaN
                merged_df[col] = merged_df[col].fillna(0)
                # Затем явно преобразуем в float
                merged_df[col] = merged_df[col].astype(float)

        print(f"📊 Новостные признаки добавлены для {len(merged_df)} строк")

    # 🔥 Добавляем целевую переменную: доходность за 15 дней в %
    merged_df['target_return'] = (merged_df['close_price'].shift(-15) - merged_df['close_price']) / merged_df[
        'close_price'] * 100

    # 5. Фильтрация новых строк (только данные после последней даты в кэше)
    try:
        last_date_dt = datetime.strptime(last_date, '%Y-%m-%d').date()
        new_data_df = merged_df[merged_df['date'] > last_date_dt].copy()
        print(f"🆕 Отобрано {len(new_data_df)} новых строк после {last_date_dt}")
    except Exception as e:
        print(f"⚠️ Не удалось определить last_date, используем все данные: {e}")
        new_data_df = merged_df.copy()

    # Если нет новых данных — пропускаем
    if new_data_df.empty:
        print(f"✅ Нет новых данных для {ticker_name}, пропускаем...")
        continue

    # 6. Добавление признаков
    new_data_df = add_lagged_features(new_data_df)
    new_data_df = add_technical_indicators(new_data_df)
    # Удаляем create_target, так как target уже добавлен выше
    # new_data_df = create_target(new_data_df)  # ❌ Удаляем эту строку

    if new_data_df.empty:
        print(f"❌ После добавления признаков данные для {ticker_name} пусты")
        continue

    # 7. Сохранение обновлённого датасета
    final_df = pd.concat([cached_df, new_data_df], ignore_index=True)
    final_df['date'] = pd.to_datetime(final_df['date']).dt.date
    final_df.drop_duplicates(subset='date', keep='last', inplace=True)
    final_df.sort_values('date', inplace=True)

    save_to_csv(final_df, ticker_name)

    # Дополнительная проверка: вывод статистики по новостным признакам
    print(f"🔍 Статистика новостных признаков для {ticker_name}:")
    print(final_df[['sentiment_avg', 'positive_news', 'neutral_news', 'negative_news']].describe())

print("\n🎉 Все данные успешно обновлены!")