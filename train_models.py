# train_models.py
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
from datetime import datetime
from config import TICKERS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
import shap  # Добавлен для интерпретации
import math
# Создание папок
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("results/explanations", exist_ok=True)

# ===================== Настройки =====================
TARGET = 'target_return'  # Теперь прогнозируем изменение в %
FEATURES = [
               'close_price', 'open_price', 'high_price', 'low_price', 'volume',
               'rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower',
               'sentiment_avg', 'positive_news', 'neutral_news', 'negative_news'
           ] + [f'close_lag_{i}' for i in range(1, 6)]

# Стандартные веса (используются только как fallback)
BASELINE_WEIGHTS = {
    'lgb': 0.3,
    'xgb': 0.5,
    'lstm': 0.2
}

LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 42
}

LSTM_SEQ_LEN = 10
LSTM_EPOCHS = 100
LSTM_BATCH_SIZE = 32
LSTM_LR = 0.001


class LSTMModel(nn.Module):
    """
    LSTM модель для прогнозирования временных рядов.
    Поддерживает dropout для регуляризации (только при num_layers > 1).
    """

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM слой с dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0  # PyTorch требует num_layers > 1 для dropout
        )
        # Полносвязный слой для выхода
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Прямой проход.
        :param x: тензор формы (batch_size, seq_len, input_size)
        :return: прогноз (batch_size, 1)
        """
        # Выход LSTM: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        # Берём только последний временной шаг
        last_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        # Прогноз
        out = self.fc(last_out)  # (batch_size, 1)
        return out


# ===================== Вспомогательные функции =====================
def load_data(ticker):
    """Загрузка данных из CSV"""
    path = f"data/{ticker}_ml_ready.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        df.drop_duplicates(subset='date', keep='last', inplace=True)
        print(f"🧠 Загружены данные для {ticker} (строк: {len(df)})")
        return df
    return None


def create_target(df):
    """Создаём целевую переменную: лог-доходность за 15 дней"""
    df = df.copy()
    # Проверяем, что данные не содержат нулевых или отрицательных цен
    if (df['close_price'] <= 0).any():
        print("⚠️ Предупреждение: Обнаружены нулевые или отрицательные цены")
        df['close_price'] = df['close_price'].clip(lower=0.01)

    df['target_log_return'] = np.log(df['close_price'].shift(-15) / df['close_price'])
    return df


def safe_rsi(ser):
    """Безопасный расчёт RSI с обработкой крайних случаев"""
    if len(ser) < 2:
        return 50
    changes = np.diff(ser)
    up = changes[changes > 0].sum()
    down = -changes[changes < 0].sum()
    if down == 0:
        return 100
    if up == 0:
        return 0
    rs = up / down
    return 100 - (100 / (1 + rs))


def prepare_features(df):
    """Подготовка всех признаков: лаги, индикаторы, цель
    ВАЖНО: Все индикаторы смещены на 1 шаг назад (shift(1)), чтобы избежать утечки данных.
    """
    df = df.copy()
    df.sort_values('date', inplace=True)
    df = df.reset_index(drop=True)
    # 1. Лаги цен
    for i in range(1, 6):
        df[f'close_lag_{i}'] = df['close_price'].shift(i)
    # 2. Технические индикаторы — все со смещением
    df['rsi'] = df['close_price'].rolling(14).apply(safe_rsi, raw=True).shift(1)
    df['ma20'] = df['close_price'].rolling(20).mean().shift(1)
    df['ma50'] = df['close_price'].rolling(50).mean().shift(1)
    df['macd'] = (df['ma20'] - df['ma50']).shift(1)
    df['macd_signal'] = df['macd'].rolling(9).mean().shift(1)
    df['bollinger_std'] = df['close_price'].rolling(20).std().shift(1)
    df['bollinger_upper'] = (df['ma20'] + 2 * df['bollinger_std']).shift(1)
    df['bollinger_lower'] = (df['ma20'] - 2 * df['bollinger_std']).shift(1)
    # 3. Целевая переменная
    df = create_target(df)
    # 4. Удаление NaN
    feature_cols = [col for col in FEATURES if col in df.columns]
    required_cols = feature_cols + [TARGET]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Не хватает колонок: {missing}")
    df.dropna(subset=required_cols, inplace=True)
    X = df[feature_cols].copy()
    y = df[TARGET].copy()
    dates = df['date'].copy()
    return X, y, dates


def save_scaler(scaler, ticker):
    """Сохранение StandardScaler"""
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, f'models/{ticker}_scaler.pkl')


def create_sequences(data, seq_length):
    """Создание последовательностей для LSTM"""
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)


def calculate_adaptive_weights(ticker, lgb_dir_acc, xgb_dir_acc, lstm_dir_acc, alpha=0.7):
    """
    Рассчитывает адаптивные веса на основе точности по направлению (directional accuracy)

    :param alpha: коэффициент сглаживания (0.7 означает 70% нового веса, 30% предыдущего)
    """
    # Проверяем, есть ли сохраненные предыдущие веса
    weights_path = f'models/{ticker}_ensemble_weights.json'
    prev_weights = None

    if os.path.exists(weights_path):
        try:
            with open(weights_path, 'r') as f:
                prev_weights = json.load(f)
        except:
            pass

    # Нормализуем текущие метрики (добавляем небольшое число, чтобы избежать 0)
    valid_models = []
    weights = []

    if not np.isnan(lgb_dir_acc):
        valid_models.append('lgb')
        weights.append(lgb_dir_acc)
    if not np.isnan(xgb_dir_acc):
        valid_models.append('xgb')
        weights.append(xgb_dir_acc)
    if not np.isnan(lstm_dir_acc):
        valid_models.append('lstm')
        weights.append(lstm_dir_acc)

    # Если нет валидных моделей, возвращаем стандартные веса
    if not valid_models:
        print(f"⚠️ Все модели имеют nan значения для {ticker}, используем стандартные веса")
        return BASELINE_WEIGHTS.copy()

    # Нормализуем веса
    total = sum(weights) + 1e-6
    current_weights = {model: weight / total for model, weight in zip(valid_models, weights)}

    # Если есть предыдущие веса, применяем экспоненциальное сглаживание
    adaptive_weights = BASELINE_WEIGHTS.copy()  # Начинаем со стандартных весов

    if prev_weights:
        for model in adaptive_weights:
            if model in current_weights:
                adaptive_weights[model] = alpha * current_weights[model] + (1 - alpha) * prev_weights.get(model,
                                                                                                          BASELINE_WEIGHTS[
                                                                                                              model])
            else:
                # Сохраняем предыдущий вес, если модель недоступна сейчас
                adaptive_weights[model] = prev_weights.get(model, BASELINE_WEIGHTS[model])
    else:
        for model in adaptive_weights:
            if model in current_weights:
                adaptive_weights[model] = current_weights[model]

    # Нормализуем, чтобы сумма весов = 1
    total_weights = sum(adaptive_weights.values())
    if total_weights > 0:
        for model in adaptive_weights:
            adaptive_weights[model] /= total_weights

    # Сохраняем для следующего раза
    with open(weights_path, 'w') as f:
        json.dump(adaptive_weights, f, indent=4)

    return adaptive_weights


def load_ensemble_weights(ticker):
    """Загружает адаптивные веса или возвращает стандартные"""
    weights_path = f'models/{ticker}_ensemble_weights.json'
    if os.path.exists(weights_path):
        try:
            with open(weights_path, 'r') as f:
                return json.load(f)
        except:
            pass
    # Возвращаем стандартные веса как fallback
    return BASELINE_WEIGHTS.copy()


def explain_prediction(ticker, model, scaler, X_full, days_ahead=0):
    """Объясняет прогноз с помощью SHAP значений"""
    try:
        # Подготовка данных
        X_scaled = scaler.transform(X_full)
        last_instance = X_scaled[-1].reshape(1, -1)

        # Создаем explainer (для деревьев)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(last_instance)

        # Сохраняем объяснение
        try:
            shap.initjs()  # Может вызывать ошибку, если IPython не установлен
            force_plot = shap.force_plot(
                explainer.expected_value,
                shap_values[0],
                features=last_instance[0],
                feature_names=X_full.columns.tolist(),
                show=False
            )

            # Сохраняем как HTML
            os.makedirs(f"results/explanations", exist_ok=True)
            shap.save_html(f"results/explanations/{ticker}_shap_{days_ahead}.html", force_plot)
        except Exception as e:
            print(f"⚠️ Предупреждение: Не удалось сгенерировать HTML-объяснение для {ticker}: {e}")

        # Возвращаем самые важные признаки
        feature_importance = pd.Series(shap_values[0], index=X_full.columns)
        top_features = feature_importance.abs().sort_values(ascending=False).head(5).index.tolist()

        return {
            'top_features': top_features,
            'explanation_path': f"results/explanations/{ticker}_shap_{days_ahead}.html" if os.path.exists(
                f"results/explanations/{ticker}_shap_{days_ahead}.html") else None
        }
    except Exception as e:
        print(f"⚠️ Ошибка при генерации SHAP объяснения для {ticker}: {e}")
        return {
            'top_features': [],
            'explanation_path': None
        }


def find_similar_patterns(ticker, X_full, n_similar=3):
    """Находит исторические периоды с похожими признаками"""
    try:
        # Нормализуем данные
        X_scaled = StandardScaler().fit_transform(X_full[FEATURES])

        # Берем последние данные как запрос
        query = X_scaled[-1]

        # Вычисляем расстояния
        distances = np.linalg.norm(X_scaled[:-15] - query, axis=1)  # исключаем последние 15 дней

        # Находим ближайшие аналоги
        similar_indices = np.argsort(distances)[:n_similar]

        # Анализируем, что происходило после этих моментов
        similar_periods = []
        for idx in similar_indices:
            # Исправлено: проверяем, что индекс в пределах диапазона
            if idx + 15 >= len(X_full):
                continue

            future_return = np.log(X_full['close_price'].iloc[idx + 15] / X_full['close_price'].iloc[idx])

            # Исправлено: получаем дату правильно
            date_value = X_full.index[idx]
            if isinstance(date_value, (int, np.integer)):
                # Если индекс числовой, пытаемся получить дату из исходного DataFrame
                if 'date' in X_full.columns:
                    date_str = X_full['date'].iloc[idx]
                else:
                    date_str = f"Дата {idx}"
            else:
                date_str = date_value

            similar_periods.append({
                'date': date_str,
                'similarity': 1 / (1 + distances[idx]),
                'future_return': future_return,
                'trend': 'up' if future_return > 0 else 'down'
            })

        return similar_periods
    except Exception as e:
        print(f"⚠️ Ошибка при поиске исторических аналогов для {ticker}: {e}")
        return []


def train_or_finetune_lstm(ticker, X_train_seq, y_train, X_val_seq, y_val, input_size, num_epochs=100):
    """Дообучение LSTM модели"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = f'models/{ticker}_lstm.pth'
    model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.2).to(device)

    # Загрузка весов (если есть)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"🔁 Загружена LSTM модель для {ticker}, дообучаем...")
        except Exception as e:
            print(f"⚠️ Не удалось загрузить модель для {ticker}: {e}. Создаём новую.")
    else:
        print(f"🆕 Создана новая LSTM модель для {ticker}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Проверяем, что данные не пустые
    if len(X_train_seq) == 0 or len(X_val_seq) == 0:
        print(f"⚠️ Недостаточно данных для обучения LSTM для {ticker}")
        return model, device

    # Конвертируем в тензоры
    X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val_seq).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device).unsqueeze(1)

    # Проверяем, что размерности совпадают
    if X_train_tensor.shape[0] != y_train_tensor.shape[0]:
        min_len = min(X_train_tensor.shape[0], y_train_tensor.shape[0])
        X_train_tensor = X_train_tensor[:min_len]
        y_train_tensor = y_train_tensor[:min_len]
        print(f"⚠️ Размерности X_train и y_train не совпадают для {ticker}. Обрезаем до {min_len} образцов.")

    if X_val_tensor.shape[0] != y_val_tensor.shape[0]:
        min_len = min(X_val_tensor.shape[0], y_val_tensor.shape[0])
        X_val_tensor = X_val_tensor[:min_len]
        y_val_tensor = y_val_tensor[:min_len]
        print(f"⚠️ Размерности X_val и y_val не совпадают для {ticker}. Обрезаем до {min_len} образцов.")

    best_loss = float('inf')
    patience = 30
    trigger = 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train_tensor)
        loss = criterion(y_pred, y_train_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor)
            val_loss = criterion(val_pred, y_val_tensor).item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Val Loss: {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            trigger = 0
        else:
            trigger += 1
            if trigger >= patience:
                print(f"🛑 Остановка обучения на эпохе {epoch} из-за early stopping")
                break

    # Сохраняем ТОЛЬКО веса модели
    torch.save(model.state_dict(), model_path)
    print(f"💾 Модель LSTM для {ticker} сохранена")
    return model, device


def predict_next_15_days(ticker, model_lgb, model_xgb, model_lstm, device, scaler, X_full):
    """Прогноз на 15 дней вперёд с адаптивными весами и доверительными интервалами"""
    model_lstm.eval()

    # Загружаем адаптивные веса
    ensemble_weights = load_ensemble_weights(ticker)

    # Масштабируем X_full с помощью scaler
    current_X = scaler.transform(X_full)
    feature_names = X_full.columns.tolist()

    # Сохраняем исходную последнюю цену
    initial_price = X_full['close_price'].iloc[-1]

    # Получаем прогноз 15-дневной лог-доходности
    flat_input = current_X[-1].reshape(1, -1)
    assert flat_input.shape[1] == len(FEATURES), f"Ожидалось {len(FEATURES)} признаков, получено {flat_input.shape[1]}"

    # Получаем предсказания
    try:
        lgb_pred = model_lgb.predict(flat_input)[0]
    except:
        lgb_pred = 0
        print(f"⚠️ Ошибка при предсказании LightGBM для {ticker}")

    try:
        xgb_pred = model_xgb.predict(flat_input)[0]
    except:
        xgb_pred = 0
        print(f"⚠️ Ошибка при предсказании XGBoost для {ticker}")

    # Подготовка последовательности для LSTM
    if len(current_X) >= LSTM_SEQ_LEN:
        seq = current_X[-LSTM_SEQ_LEN:].reshape(1, LSTM_SEQ_LEN, -1)
        seq_tensor = torch.FloatTensor(seq).to(device)
        with torch.no_grad():
            try:
                lstm_pred = model_lstm(seq_tensor).cpu().numpy().flatten()[0]
            except Exception as e:
                print(f"⚠️ Ошибка при предсказании LSTM для {ticker}: {e}")
                lstm_pred = 0
    else:
        lstm_pred = 0
        print(f"⚠️ Недостаточно данных для LSTM последовательности для {ticker}")

    # Используем адаптивные веса, учитывая возможные ошибки
    valid_models = []
    weighted_sum = 0
    total_weight = 0

    if not np.isnan(lgb_pred) and not np.isinf(lgb_pred):
        valid_models.append('lgb')
        weighted_sum += ensemble_weights.get('lgb', BASELINE_WEIGHTS['lgb']) * lgb_pred
        total_weight += ensemble_weights.get('lgb', BASELINE_WEIGHTS['lgb'])

    if not np.isnan(xgb_pred) and not np.isinf(xgb_pred):
        valid_models.append('xgb')
        weighted_sum += ensemble_weights.get('xgb', BASELINE_WEIGHTS['xgb']) * xgb_pred
        total_weight += ensemble_weights.get('xgb', BASELINE_WEIGHTS['xgb'])

    if not np.isnan(lstm_pred) and not np.isinf(lstm_pred):
        valid_models.append('lstm')
        weighted_sum += ensemble_weights.get('lstm', BASELINE_WEIGHTS['lstm']) * lstm_pred
        total_weight += ensemble_weights.get('lstm', BASELINE_WEIGHTS['lstm'])

    # Если есть валидные модели, используем их
    if total_weight > 0:
        ensemble_pred = weighted_sum / total_weight
    else:
        # Если все модели дали ошибку, используем среднее значение
        ensemble_pred = 0
        print(f"⚠️ Все модели дали ошибку для {ticker}, используем нейтральный прогноз")

    # Оценка волатильности для доверительных интервалов
    daily_returns = X_full['close_price'].pct_change().dropna()
    if len(daily_returns) > 0:
        volatility = daily_returns.std() * np.sqrt(252)  # годовая волатильность
    else:
        volatility = 0.3  # значение по умолчанию (30%)

    # Рассчитываем разумные пределы на основе волатильности
    max_reasonable_log_return = np.log(1 + 3 * volatility * np.sqrt(15 / 252))
    min_reasonable_log_return = np.log(1 - 3 * volatility * np.sqrt(15 / 252))

    # Проверяем, не выходят ли предсказания за разумные пределы
    if ensemble_pred > max_reasonable_log_return:
        print(
            f"⚠️ Предупреждение: Прогноз для {ticker} ({ensemble_pred:.4f}) превышает разумный предел ({max_reasonable_log_return:.4f}). Корректируем.")
        ensemble_pred = max_reasonable_log_return
    elif ensemble_pred < min_reasonable_log_return:
        print(
            f"⚠️ Предупреждение: Прогноз для {ticker} ({ensemble_pred:.4f}) ниже разумного предела ({min_reasonable_log_return:.4f}). Корректируем.")
        ensemble_pred = min_reasonable_log_return

    # Рассчитываем доверительные интервалы для 15-дневного прогноза
    ci_range = volatility * np.sqrt(15 / 252) * 1.645  # 90% ДИ для 15 дней
    final_price = initial_price * np.exp(ensemble_pred)
    ci_lower = initial_price * np.exp(ensemble_pred - ci_range)
    ci_upper = initial_price * np.exp(ensemble_pred + ci_range)

    # Генерация плавного перехода для графика
    price_forecast = np.linspace(initial_price, final_price, 15)
    ci_lower_values = np.linspace(initial_price, ci_lower, 15)
    ci_upper_values = np.linspace(initial_price, ci_upper, 15)

    # Рассчитываем ожидаемую доходность
    expected_return = (final_price / initial_price - 1) * 100
    expected_return_range = (
        (ci_lower / initial_price - 1) * 100,
        (ci_upper / initial_price - 1) * 100
    )

    # Расчёт вероятности роста
    def calc_growth_prob(model_pred):
        # Используем волатильность для определения вероятности
        z_score = model_pred / (volatility * np.sqrt(15 / 252))
        # Приближенная вероятность через стандартное нормальное распределение
        return max(0, min(100, (0.5 + 0.5 * math.erf(z_score / math.sqrt(2))) * 100))

    lgb_growth_prob = calc_growth_prob(lgb_pred) if 'lgb' in valid_models else 50.0
    xgb_growth_prob = calc_growth_prob(xgb_pred) if 'xgb' in valid_models else 50.0
    lstm_growth_prob = calc_growth_prob(lstm_pred) if 'lstm' in valid_models else 50.0

    # Ансамблевая вероятность
    ensemble_growth_prob = calc_growth_prob(ensemble_pred)

    # Возвращаем результаты
    return {
        'price_forecast': price_forecast,
        'current_price': initial_price,
        'ci_lower': ci_lower_values,
        'ci_upper': ci_upper_values,
        'expected_return': expected_return,
        'expected_return_range': expected_return_range,
        'growth_probs': {
            'lgb': lgb_growth_prob,
            'xgb': xgb_growth_prob,
            'lstm': lstm_growth_prob,
            'ensemble': ensemble_growth_prob
        }
    }


# ===================== Обучение LightGBM =====================
def train_or_finetune_lgb(ticker, X_train, y_train, X_val, y_val):
    model_path = f"models/{ticker}_lgb.txt"
    init_model = model_path if os.path.exists(model_path) else None
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    print(f"📈 {'Дообучаем' if init_model else 'Обучаем'} LightGBM для {ticker}...")
    model = lgb.train(
        LGB_PARAMS,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        init_model=init_model,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    model.save_model(model_path)
    return model


# ===================== Обучение XGBoost =====================
def train_or_finetune_xgb(ticker, X_train, y_train, X_val, y_val):
    model_path = f"models/{ticker}_xgb.pkl"
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.9,
        'random_state': 42
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    print(f"📈 {'Дообучаем' if os.path.exists(model_path) else 'Обучаем'} XGBoost для {ticker}...")
    if os.path.exists(model_path):
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=1000,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=100,
            xgb_model=model_path
        )
    else:
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=1000,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=100
        )
    booster.save_model(model_path)
    model = xgb.XGBRegressor(**params)
    model._Booster = booster
    return model


# ===================== Скалер и метаданные =====================
def load_scaler(ticker):
    path = f"models/{ticker}_scaler.pkl"
    if os.path.exists(path):
        scaler = joblib.load(path)
        print(f"🔁 Загружен скалер для {ticker}")
        return scaler
    return None


def save_scaler(scaler, ticker):
    joblib.dump(scaler, f"models/{ticker}_scaler.pkl")
    print(f"💾 Сохранен скалер для {ticker}")


def save_meta(ticker, last_date, data_length):
    """Сохранение метаданных модели"""
    os.makedirs('models', exist_ok=True)
    meta_path = f'models/{ticker}_meta.json'
    # Преобразуем Timestamp в строку
    if isinstance(last_date, pd.Timestamp):
        last_date_str = last_date.strftime('%Y-%m-%d %H:%M:%S')
    else:
        last_date_str = str(last_date)
    meta = {
        'ticker': ticker,
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'last_date_in_data': last_date_str,
        'data_length': data_length,
        'model_version': '1.1'  # обновленная версия
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
    print(f"💾 Метаданные для {ticker} сохранены")


def main():
    all_predictions = []
    for ticker in TICKERS:
        print(f"🚀 Обработка тикера: {ticker}")
        # 1. Загрузка данных
        df = load_data(ticker)
        if df is None or df.empty or len(df) < 50:
            continue
        # 2. Подготовка признаков
        X, y, dates = prepare_features(df)
        if X.empty or len(X) < 50:
            continue
        # 3. Разделение на train / val / test
        split1 = int(len(X) * 0.7)
        split2 = int(len(X) * 0.85)
        X_train, X_val, X_test = X[:split1], X[split1:split2], X[split2:]
        y_train, y_val, y_test = y[:split1], y[split1:split2], y[split2:]
        # 4. Стандартизация — с сохранением DataFrame для feature names
        scaler = StandardScaler()
        X_full_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        X_train_scaled = X_full_scaled.iloc[:split1]
        X_val_scaled = X_full_scaled.iloc[split1:split2]
        X_test_scaled = X_full_scaled.iloc[split2:]
        save_scaler(scaler, ticker)
        save_meta(ticker, dates.iloc[-1], len(X))
        # 5. Обучение LightGBM и XGBoost
        model_lgb = train_or_finetune_lgb(ticker, X_train_scaled.values, y_train, X_val_scaled.values, y_val)
        model_xgb = train_or_finetune_xgb(ticker, X_train_scaled.values, y_train, X_val_scaled.values, y_val)
        # 6. LSTM: подготовка последовательностей
        X_seq = create_sequences(X_full_scaled.values, LSTM_SEQ_LEN)
        train_end = split1 - LSTM_SEQ_LEN
        val_end = split2 - LSTM_SEQ_LEN
        X_train_seq = X_seq[:train_end]
        X_val_seq = X_seq[train_end:val_end]
        y_train_seq = y.iloc[LSTM_SEQ_LEN:split1].values
        y_val_seq = y.iloc[split1:split2].values
        model_lstm, device = train_or_finetune_lstm(
            ticker, X_train_seq, y_train_seq, X_val_seq, y_val_seq, input_size=X.shape[1]
        )
        # 7. Оценка качества на тесте
        dir_acc_lgb = dir_acc_xgb = dir_acc_lstm = baseline_dir_acc = np.nan

        try:
            # LightGBM
            y_pred_lgb = model_lgb.predict(X_test_scaled.values)
            rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
            mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
            dir_acc_lgb = ((y_pred_lgb > 0) == (y_test > 0)).mean()
            print(f"📊 {ticker} | LGB: RMSE={rmse_lgb:.3f}, MAE={mae_lgb:.3f}, DirAcc={dir_acc_lgb:.3f}")

            # XGBoost
            y_pred_xgb = model_xgb.predict(X_test_scaled.values)
            rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
            mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
            dir_acc_xgb = ((y_pred_xgb > 0) == (y_test > 0)).mean()
            print(f"📊 {ticker} | XGB: RMSE={rmse_xgb:.3f}, MAE={mae_xgb:.3f}, DirAcc={dir_acc_xgb:.3f}")

            # LSTM
            if len(X_seq) > 0 and len(y_test) >= LSTM_SEQ_LEN:
                X_test_seq = create_sequences(X_test_scaled.values, LSTM_SEQ_LEN)
                y_test_lstm = y_test.iloc[LSTM_SEQ_LEN:].values

                if len(X_test_seq) == len(y_test_lstm):
                    X_test_seq_tensor = torch.FloatTensor(X_test_seq).to(device)
                    model_lstm.eval()
                    with torch.no_grad():
                        y_pred_lstm = model_lstm(X_test_seq_tensor).cpu().numpy().flatten()
                    rmse_lstm = np.sqrt(mean_squared_error(y_test_lstm, y_pred_lstm))
                    mae_lstm = mean_absolute_error(y_test_lstm, y_pred_lstm)
                    dir_acc_lstm = ((y_pred_lstm > 0) == (y_test_lstm > 0)).mean()
                    print(f"📊 {ticker} | LSTM: RMSE={rmse_lstm:.3f}, MAE={mae_lstm:.3f}, DirAcc={dir_acc_lstm:.3f}")
                else:
                    print(
                        f"⚠️ Несоответствие размерностей для LSTM в {ticker}: {len(X_test_seq)} vs {len(y_test_lstm)}")
            else:
                print(f"⚠️ Недостаточно данных для LSTM теста в {ticker}")

            # Бейзлайн: простой тренд (дневная доходность)
            baseline_preds = X['close_price'].pct_change().shift(-1)
            baseline_preds = baseline_preds.iloc[split2:-15]  # выравнивание с y_test
            y_test_aligned = y_test.iloc[:-15]
            if len(baseline_preds) == len(y_test_aligned):
                baseline_dir_acc = ((baseline_preds > 0) == (y_test_aligned > 0)).mean()
                print(f"📊 {ticker} | Baseline: DirAcc={baseline_dir_acc:.3f}")
            else:
                print(f"⚠️ Несоответствие размерностей для бейзлайна в {ticker}")
        except Exception as e:
            print(f"⚠️ Не удалось оценить качество на тесте для {ticker}: {e}")

        # 7.5. Расчет адаптивных весов
        try:
            adaptive_weights = calculate_adaptive_weights(
                ticker,
                dir_acc_lgb,
                dir_acc_xgb,
                dir_acc_lstm
            )
            print(f"⚖️ Адаптивные веса для {ticker}: LGB={adaptive_weights['lgb']:.2f}, "
                  f"XGB={adaptive_weights['xgb']:.2f}, LSTM={adaptive_weights['lstm']:.2f}")
        except Exception as e:
            print(f"⚠️ Не удалось рассчитать адаптивные веса для {ticker}: {e}, используются стандартные")

        # 8. Прогноз на 15 дней
        try:
            growth_probs = predict_next_15_days(ticker, model_lgb, model_xgb, model_lstm, device, scaler, X)

            # Генерация SHAP объяснения
            explanation = {'top_features': []}
            try:
                explanation = explain_prediction(ticker, model_xgb, scaler, X)
                if explanation['top_features']:
                    print(f"🔍 {ticker}: Прогноз обусловлен в первую очередь: {', '.join(explanation['top_features'])}")
            except Exception as e:
                print(f"⚠️ Не удалось сгенерировать SHAP объяснение для {ticker}: {e}")

            # Поиск исторических аналогов
            similar_patterns = []
            try:
                similar_patterns = find_similar_patterns(ticker, X)
                if similar_patterns:
                    print(f"🔍 Исторические аналоги для {ticker}:")
                    for i, pattern in enumerate(similar_patterns, 1):
                        trend = "↑" if pattern['trend'] == 'up' else "↓"
                        # Исправлено: безопасное форматирование даты
                        date_str = pattern['date'].strftime('%Y-%m-%d') if hasattr(pattern['date'],
                                                                                   'strftime') else str(pattern['date'])
                        print(
                            f"   {i}. {date_str} (сходство: {pattern['similarity']:.2f}) → {trend} {pattern['future_return']:.2%}")
            except Exception as e:
                print(f"⚠️ Не удалось найти исторические аналоги для {ticker}: {e}")

            all_predictions.append({
                'ticker': ticker,
                'lgb_growth_prob': growth_probs['growth_probs']['lgb'],
                'xgb_growth_prob': growth_probs['growth_probs']['xgb'],
                'lstm_growth_prob': growth_probs['growth_probs']['lstm'],
                'ensemble_growth_prob': growth_probs['growth_probs']['ensemble'],
                'current_price': growth_probs['current_price'],
                'predictions': growth_probs['price_forecast'],
                'ci_lower': growth_probs['ci_lower'],
                'ci_upper': growth_probs['ci_upper'],
                'expected_return': growth_probs['expected_return'],
                'expected_return_range': growth_probs['expected_return_range'],
                'test_dir_acc_lgb': dir_acc_lgb,
                'test_dir_acc_xgb': dir_acc_xgb,
                'test_dir_acc_lstm': dir_acc_lstm,
                'baseline_dir_acc': baseline_dir_acc,
                'shap_explanation': explanation['top_features'],
                'historical_analogues': similar_patterns
            })
        except Exception as e:
            print(f"❌ Ошибка при прогнозировании для {ticker}: {e}")
            import traceback
            traceback.print_exc()
            continue
    # 9. Вывод результатов
    if all_predictions:
        results_df = pd.DataFrame(all_predictions)

        # Сортируем по ожидаемой доходности
        results_df_sorted = results_df.sort_values(by='expected_return', ascending=False).reset_index(drop=True)

        print("\n" + "📊 АНАЛИЗ ПРОГНОЗОВ ПО ВСЕМ АКЦИЯМ")
        print("=" * 110)

        # Фильтруем акции с положительной ожидаемой доходностью
        positive_return = results_df_sorted[results_df_sorted['expected_return'] > 0]

        if positive_return.empty:
            top_3 = results_df_sorted.head(3)
            print("📉 Нет акций с положительной ожидаемой доходностью.")
            print("🛡️  ТОП-3 с наименьшими ожидаемыми потерями:")
        else:
            top_3 = positive_return.head(3)
            print("🏆 ТОП-3 АКЦИЙ К ПОКУПКЕ НА БЛИЖАЙШИЕ 15 ДНЕЙ")
            print("📌 Основано на ожидаемой доходности: чем выше %, тем больше потенциальная прибыль")

        print("-" * 110)
        print(
            f"{'Место':<6} {'Тикер':<8} {'LGB':<6} {'XGB':<6} {'LSTM':<6} {'Вероятность':<12} {'Ожид. доходность':<18}")
        print("-" * 110)

        for idx, (_, row) in enumerate(top_3.iterrows(), 1):
            # Форматируем ожидаемую доходность с доверительным интервалом
            lower, upper = row['expected_return_range']
            return_str = f"{row['expected_return']:.2f}% [{lower:.2f}% - {upper:.2f}%]"

            print(f"{idx:<6} {row['ticker']:<8} "
                  f"{row['lgb_growth_prob']:>5.1f}% "
                  f"{row['xgb_growth_prob']:>5.1f}% "
                  f"{row['lstm_growth_prob']:>5.1f}% "
                  # ИСПРАВЛЕНО: используем ensemble_growth_prob вместо growth_probs['ensemble']
                  f"{row['ensemble_growth_prob']:>10.1f}% "
                  f"{return_str:<18}")

        print("-" * 110)

        # Полный рейтинг
        print("\n📋 ПОЛНЫЙ РЕЙТИНГ ВСЕХ АКЦИЙ (от наиболее перспективной к наименее)")
        print("💡 Ожидаемая доходность — прогнозируемое изменение цены через 15 дней")
        print("📊 Вероятность — доля дней с положительной доходностью в прогнозе")
        print("-" * 110)
        print(
            f"{'Место':<6} {'Тикер':<8} {'Ожид. доходность':<18} {'Вероятность':<12} {'LGB':<6} {'XGB':<6} {'LSTM':<6}")
        print("-" * 110)

        for idx, (_, row) in enumerate(results_df_sorted.iterrows(), 1):
            # Форматируем ожидаемую доходность с доверительным интервалом
            lower, upper = row['expected_return_range']
            return_str = f"{row['expected_return']:.2f}% [{lower:.2f}% - {upper:.2f}%]"

            print(f"{idx:<6} {row['ticker']:<8} "
                  f"{return_str:<18} "
                  # ИСПРАВЛЕНО: используем ensemble_growth_prob вместо growth_probs['ensemble']
                  f"{row['ensemble_growth_prob']:>10.1f}% "
                  f"{row['lgb_growth_prob']:>5.1f}% "
                  f"{row['xgb_growth_prob']:>5.1f}% "
                  f"{row['lstm_growth_prob']:>5.1f}%")

        # Добавим информацию об интерпретации
        print("\n🔍 ДЕТАЛИ ПРОГНОЗА ДЛЯ ТОП-3 АКЦИЙ:")
        print("-" * 110)

        for idx, (_, row) in enumerate(top_3.iterrows(), 1):
            print(f"{idx}. {row['ticker']}")
            if row['shap_explanation']:
                print(f"   💡 Основные драйверы прогноза: {', '.join(row['shap_explanation'])}")

            if isinstance(row['historical_analogues'], list) and len(row['historical_analogues']) > 0:
                print(f"   🔍 Исторические аналоги:")
                for i, pattern in enumerate(row['historical_analogues'][:2], 1):
                    trend = "↑" if pattern['trend'] == 'up' else "↓"
                    date_str = pattern['date'].strftime('%Y-%m-%d') if hasattr(pattern['date'], 'strftime') else str(
                        pattern['date'])
                    print(
                        f"      {i}. {date_str} (сходство: {pattern['similarity']:.2f}) → {trend} {pattern['future_return']:.2%}")

            # Выводим ожидаемую доходность с доверительным интервалом
            lower, upper = row['expected_return_range']
            print(f"   📈 Прогнозируемая доходность: {row['expected_return']:.2f}%")
            print(f"   📊 90% доверительный интервал: [{lower:.2f}% - {upper:.2f}%]")
            print(f"   💰 Цена: {row['current_price']:.2f} → {row['predictions'][-1]:.2f}")
            print("-" * 110)

        # Сохранение
        results_df_sorted.to_csv('results/all_predictions.csv', index=False)
        top_3.to_csv('results/top_stocks.csv', index=False)
        print("\n📊 Все результаты сохранены: all_predictions.csv, top_stocks.csv")
    else:
        print("❌ Не удалось рассчитать прогнозы ни для одной акции.")
    print("🎉 Обработка завершена!")


if __name__ == "__main__":
    main()