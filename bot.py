import os
import logging
import pandas as pd
import numpy as np
import nest_asyncio
import matplotlib.pyplot as plt
import io
import math
from datetime import datetime
from telegram import Update, ForceReply, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from config import TG_BOT_TOKEN
from config import TICKERS  # Добавляем импорт тикеров

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Применение nest_asyncio
nest_asyncio.apply()

# Пути к файлам
RESULTS_PATH = 'results/all_predictions.csv'
TOP_STOCKS_PATH = 'results/top_stocks.csv'
BACKTEST_SUMMARY = 'backtest_results/backtest_summary.txt'
BACKTEST_CHART = 'backtest_results/comprehensive_backtest_analysis.png'

# Приветственное сообщение
WELCOME_MESSAGE = """
📈 *Добро пожаловать в бота прогнозирования акций!*

Я использую ансамбль из трех моделей (LightGBM, XGBoost и LSTM) для прогнозирования движения акций на 15 дней вперед.

Нажмите кнопку ниже или введите команду для:
• Получения ТОП-3 акций
• Просмотра прогноза по конкретной акции
• Ознакомления с эффективностью стратегии
"""

# Кнопки
BUTTON_TOP = "📈 Показать ТОП-3 акций"
BUTTON_HELP = "❓ Справка"
BUTTON_STRATEGY = "📊 Эффективность стратегии"
BUTTON_STOCKS = "🔍 Прогноз по акции"
BUTTON_RECOMMEND = "💡 Персональные рекомендации"

# Состояния для диалога
CHOOSING_TICKER = "CHOOSING_TICKER"


# Клавиатура с основными кнопками
def get_main_keyboard():
    buttons = [
        [KeyboardButton(BUTTON_TOP), KeyboardButton(BUTTON_STRATEGY)],
        [KeyboardButton(BUTTON_STOCKS), KeyboardButton(BUTTON_RECOMMEND)],
        [KeyboardButton(BUTTON_HELP)]
    ]
    return ReplyKeyboardMarkup(buttons, resize_keyboard=True, one_time_keyboard=False)


# Клавиатура с тикерами
def get_ticker_keyboard():
    # Группируем тикеры по 3 в строку
    ticker_buttons = [KeyboardButton(ticker) for ticker in TICKERS]
    keyboard = [ticker_buttons[i:i + 3] for i in range(0, len(ticker_buttons), 3)]
    keyboard.append([KeyboardButton("🔙 Назад")])
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)


# Команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.clear()  # Очищаем состояние пользователя

    if update.message:
        await update.message.reply_text(
            WELCOME_MESSAGE,
            parse_mode='Markdown',
            reply_markup=get_main_keyboard()
        )
    else:
        await update.callback_query.message.reply_text(
            WELCOME_MESSAGE,
            parse_mode='Markdown',
            reply_markup=get_main_keyboard()
        )


# Команда /help
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = """
📌 *Доступные команды:*

📈 /top - Показать ТОП-3 акций
🔍 /stock [ТИКЕР] - Прогноз по конкретной акции (например, /stock GAZP)
📊 /strategy - Эффективность торговой стратегии
💡 /recommend - Персональные рекомендации
❓ /help - Эта справка

*Как использовать:*
• Нажмите кнопку внизу для быстрого доступа
• Или введите команду вручную
• Для просмотра прогноза по акции введите её тикер

*Примечание:*
Данные обновляются ежедневно после 18:00 по МСК
"""
    if update.message:
        await update.message.reply_text(help_text, parse_mode='Markdown', reply_markup=get_main_keyboard())
    else:
        await update.callback_query.message.reply_text(help_text, parse_mode='Markdown',
                                                       reply_markup=get_main_keyboard())


# Обработчик ТОП-3 акций
async def send_top_stocks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        if not os.path.exists(TOP_STOCKS_PATH):
            await update.message.reply_text("❌ Файл с результатами не найден. Убедитесь, что модели обучены.")
            return

        df = pd.read_csv(TOP_STOCKS_PATH)
        if df.empty:
            await update.message.reply_text("❌ Нет данных для отображения ТОП-3 акций.")
            return

        # Сортируем по ожидаемой доходности
        df = df.sort_values(by='expected_return', ascending=False).head(3)

        message = "🏆 *ТОП-3 АКЦИЙ К ПОКУПКЕ НА 15 ДНЕЙ*\n\n"
        message += "📌 *Ранжирование по ожидаемой доходности (от высокой к низкой)*\n\n"

        for idx, row in df.iterrows():
            # Используем отдельные колонки для границ интервала
            try:
                # Пытаемся получить границы интервала из отдельных колонок
                lower_bound = float(row['expected_return_lower'])
                upper_bound = float(row['expected_return_upper'])
            except:
                try:
                    # Пытаемся распарсить кортеж из expected_return_range
                    expected_range = str(row['expected_return_range']).replace('(', '').replace(')', '')
                    lower_bound_str, upper_bound_str = expected_range.split(',')
                    lower_bound = float(lower_bound_str.strip())
                    upper_bound = float(upper_bound_str.strip())
                except:
                    # Если все методы не сработали, используем стандартные значения
                    lower_bound = max(0, float(row['expected_return']) * 0.5)
                    upper_bound = float(row['expected_return']) * 1.5

            expected_return = float(row['expected_return'])

            message += f"📈 *{row['ticker']}*\n"
            message += f"• Ожидаемая доходность: *{expected_return:.2f}%*\n"
            message += f"• Доверительный интервал: [{lower_bound:.2f}% - {upper_bound:.2f}%]\n"
            message += f"• Вероятность роста: *{row['ensemble_growth_prob']:.1f}%*\n"

            # Проверяем наличие составного скоринга
            if 'composite_score' in row and not pd.isna(row['composite_score']) and row['composite_score'] != 0:
                message += f"• Составной скор: *{row['composite_score']:.1f}*\n\n"
            else:
                # Если составного скоринга нет, вычисляем его
                composite_score = calculate_composite_score(expected_return, row['ensemble_growth_prob'])
                message += f"• Составной скор: *{composite_score:.1f}*\n\n"

        message += "💡 *Совет:* Чем выше составной скор и ожидаемая доходность, тем привлекательнее позиция для покупки."

        if update.message:
            await update.message.reply_text(message, parse_mode='Markdown', reply_markup=get_main_keyboard())
        else:
            await update.callback_query.message.reply_text(message, parse_mode='Markdown',
                                                           reply_markup=get_main_keyboard())

    except Exception as e:
        error_msg = f"❌ Ошибка при обработке данных: {str(e)}"
        if update.message:
            await update.message.reply_text(error_msg, reply_markup=get_main_keyboard())
        else:
            await update.callback_query.message.reply_text(error_msg, reply_markup=get_main_keyboard())


# Обработчик эффективности стратегии
async def send_strategy_performance(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        if not os.path.exists(BACKTEST_SUMMARY):
            await update.message.reply_text(
                "📊 Данные о эффективности стратегии недоступны.\n"
                "Запустите бэктест с помощью команды `python backtest.py`",
                reply_markup=get_main_keyboard()
            )
            return

        # Читаем данные из файла
        with open(BACKTEST_SUMMARY, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Формируем сообщение
        message = "📊 *ЭФФЕКТИВНОСТЬ ТОРГОВОЙ СТРАТЕГИИ*\n\n"
        for line in lines[2:]:  # Пропускаем заголовок
            if ':' in line:
                key, value = line.strip().split(':', 1)
                message += f"• *{key.strip()}*: {value.strip()}\n"

        message += "\n📌 *Интерпретация ключевых метрик:*\n"
        message += "• *Win Rate*: >55% - высокая точность, 45-55% - умеренная, <45% - низкая\n"
        message += "• *Profit Factor*: >1.5 - высокая прибыльность, >1.0 - умеренная, <1.0 - низкая\n"
        message += "• *Sharpe Ratio*: >1.0 - высокая эффективность, >0.5 - умеренная, <0.5 - низкая\n\n"
        message += "💡 *Совет:* Идеальная стратегия должна иметь высокие значения по всем метрикам."

        # Отправляем текстовое сообщение
        if update.message:
            await update.message.reply_text(message, parse_mode='Markdown', reply_markup=get_main_keyboard())
        else:
            await update.callback_query.message.reply_text(message, parse_mode='Markdown',
                                                           reply_markup=get_main_keyboard())

        # Отправляем график, если он существует
        if os.path.exists(BACKTEST_CHART):
            with open(BACKTEST_CHART, 'rb') as photo:
                if update.message:
                    await update.message.reply_photo(photo, caption="📈 График эффективности стратегии")
                else:
                    await update.callback_query.message.reply_photo(photo, caption="📈 График эффективности стратегии")

    except Exception as e:
        error_msg = f"❌ Ошибка при обработке данных: {str(e)}"
        if update.message:
            await update.message.reply_text(error_msg, reply_markup=get_main_keyboard())
        else:
            await update.callback_query.message.reply_text(error_msg, reply_markup=get_main_keyboard())


# Обработчик выбора акции
async def choose_stock(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data['state'] = CHOOSING_TICKER
    await update.message.reply_text(
        "🔍 *Введите тикер акции* (например, GAZP, SBER, VTBR)\n"
        "Или выберите из списка ниже:",
        parse_mode='Markdown',
        reply_markup=get_ticker_keyboard()
    )


# Функция для расчета составного скоринга
def calculate_composite_score(expected_return, ensemble_growth_prob):
    """
    Рассчитываем составной скор на основе вероятности роста и ожидаемой доходности
    Скор нормализован в диапазоне 0-100
    """
    # Нормализуем вероятность роста (уже в %)
    growth_prob_score = ensemble_growth_prob / 100.0

    # Нормализуем ожидаемую доходность (приводим к шкале 0-1)
    # Предполагаем, что доходность выше 10% - это отлично
    expected_return_score = min(expected_return / 10.0, 1.0)
    expected_return_score = max(expected_return_score, 0.0)

    # Взвешенное среднее
    composite_score = (0.7 * growth_prob_score + 0.3 * expected_return_score) * 100

    return composite_score


# Обработчик прогноза по конкретной акции
async def send_stock_forecast(update: Update, context: ContextTypes.DEFAULT_TYPE, ticker=None) -> None:
    if not ticker:
        ticker = update.message.text

    # Проверяем, не запросили ли возврат в главное меню
    if ticker == "🔙 НАЗАД":
        context.user_data.clear()
        await start(update, context)
        return

    try:
        if not os.path.exists(RESULTS_PATH):
            await update.message.reply_text("❌ Файл с результатами не найден. Убедитесь, что модели обучены.")
            return

        df = pd.read_csv(RESULTS_PATH)
        stock_data = df[df['ticker'] == ticker]

        if stock_data.empty:
            await update.message.reply_text(
                f"❌ Данные для акции {ticker} не найдены.\n"
                "Пожалуйста, введите корректный тикер.",
                reply_markup=get_ticker_keyboard()
            )
            return

        row = stock_data.iloc[0]

        # Извлекаем данные
        current_price = float(row['current_price'])
        expected_return = float(row['expected_return'])

        # Используем отдельные колонки для границ интервала
        try:
            # Пытаемся получить границы интервала из отдельных колонок
            lower_bound = float(row['expected_return_lower'])
            upper_bound = float(row['expected_return_upper'])
        except:
            try:
                # Пытаемся распарсить кортеж из expected_return_range
                expected_range = str(row['expected_return_range']).replace('(', '').replace(')', '')
                lower_bound_str, upper_bound_str = expected_range.split(',')
                lower_bound = float(lower_bound_str.strip())
                upper_bound = float(upper_bound_str.strip())
            except:
                # Если все методы не сработали, используем стандартные значения
                lower_bound = max(0, expected_return * 0.5)
                upper_bound = expected_return * 1.5

        final_price = current_price * (1 + expected_return / 100)

        # Формируем сообщение
        message = f"📊 *ПРОГНОЗ ПО АКЦИИ {ticker}*\n\n"
        message += f"💰 *Текущая цена*: {current_price:.2f} руб.\n"
        message += f"📈 *Ожидаемая цена через 15 дней*: {final_price:.2f} руб.\n"
        message += f"🎯 *Ожидаемая доходность*: *{expected_return:.2f}%*\n"
        message += f"📏 *90% доверительный интервал*: [{lower_bound:.2f}% - {upper_bound:.2f}%]\n"
        message += f"✅ *Вероятность роста*: *{row['ensemble_growth_prob']:.1f}%*\n"

        # Проверяем наличие составного скоринга
        if 'composite_score' in row and not pd.isna(row['composite_score']) and row['composite_score'] != 0:
            message += f"⭐ *Составной скор*: *{row['composite_score']:.1f}*\n\n"
        else:
            # Если составного скоринга нет, вычисляем его
            composite_score = calculate_composite_score(expected_return, row['ensemble_growth_prob'])
            message += f"⭐ *Составной скор*: *{composite_score:.1f}*\n\n"

        # Добавляем информацию о моделях
        message += "🧠 *Прогнозы отдельных моделей:*\n"

        # Проверяем и используем веса моделей, если они есть
        try:
            lgb_weight = float(row['lgb_weight'])
        except:
            lgb_weight = 0.33

        try:
            xgb_weight = float(row['xgb_weight'])
        except:
            xgb_weight = 0.33

        try:
            lstm_weight = float(row['lstm_weight'])
        except:
            lstm_weight = 0.33

        message += f"• LightGBM: {row['lgb_growth_prob']:.1f}% (вес: {lgb_weight:.2f})\n"
        message += f"• XGBoost: {row['xgb_growth_prob']:.1f}% (вес: {xgb_weight:.2f})\n"
        message += f"• LSTM: {row['lstm_growth_prob']:.1f}% (вес: {lstm_weight:.2f})\n\n"

        # Добавляем рекомендацию
        if expected_return > 2.0 and row['ensemble_growth_prob'] > 60:
            message += "✅ *РЕКОМЕНДАЦИЯ*: СИЛЬНЫЙ ПОКУПАТЬ\n"
            message += "Акция имеет высокую ожидаемую доходность и уверенность в прогнозе."
        elif expected_return > 0.5 and row['ensemble_growth_prob'] > 50:
            message += "⚠️ *РЕКОМЕНДАЦИЯ*: УМЕРЕННЫЙ ПОКУПАТЬ\n"
            message += "Акция имеет положительную ожидаемую доходность, но с умеренной уверенностью."
        else:
            message += "❌ *РЕКОМЕНДАЦИЯ*: НЕ РЕКОМЕНДУЕТСЯ К ПОКУПКЕ\n"
            message += "Акция имеет низкую или отрицательную ожидаемую доходность."

        # Отправляем сообщение
        if update.message:
            await update.message.reply_text(message, parse_mode='Markdown', reply_markup=get_main_keyboard())
        else:
            await update.callback_query.message.reply_text(message, parse_mode='Markdown',
                                                           reply_markup=get_main_keyboard())

        # Добавляем информацию об интерпретации, если она есть
        if 'shap_explanation' in row and row['shap_explanation'] and row['shap_explanation'] != '[]':
            try:
                # Убираем квадратные скобки и кавычки
                features = str(row['shap_explanation']).strip('[]').replace("'", "").split(',')
                features = [f.strip() for f in features if f.strip()]

                if features:
                    interpretation = "\n\n🔍 *ОСНОВНЫЕ ДРАЙВЕРЫ ПРОГНОЗА:*\n"
                    for i, feature in enumerate(features[:5], 1):
                        interpretation += f"{i}. {feature}\n"

                    interpretation += "\n💡 *Как это интерпретировать:*\n"
                    interpretation += "Эти факторы оказали наибольшее влияние на прогноз модели."

                    if update.message:
                        await update.message.reply_text(interpretation, parse_mode='Markdown')
                    else:
                        await update.callback_query.message.reply_text(interpretation, parse_mode='Markdown')
            except Exception as e:
                print(f"Ошибка при обработке SHAP объяснения: {e}")
                pass

        # Добавляем информацию об исторических аналогах, если она есть
        if 'historical_analogues' in row and row['historical_analogues'] and row['historical_analogues'] != '[]':
            try:
                # Парсим исторические аналоги
                analogues_str = str(row['historical_analogues'])

                # Пытаемся исправить строку, если она содержит np.float64
                analogues_str = analogues_str.replace('np.float64(', '').replace(')', '')

                analogues = eval(analogues_str)

                if analogues:
                    history_msg = "\n\n🕰️ *ИСТОРИЧЕСКИЕ АНАЛОГИ:*\n"
                    for i, pattern in enumerate(analogues[:3], 1):
                        trend = "📈" if pattern['trend'] == 'up' else "📉"
                        similarity = pattern['similarity']
                        future_return = pattern['future_return']
                        date_str = pattern['date']

                        history_msg += f"{i}. {date_str} (сходство: {similarity:.2f}) {trend} {future_return:.2%}\n"

                    history_msg += "\n💡 *Как это интерпретировать:*\n"
                    history_msg += "Эти периоды в истории показывают похожие рыночные условия."

                    if update.message:
                        await update.message.reply_text(history_msg, parse_mode='Markdown')
                    else:
                        await update.callback_query.message.reply_text(history_msg, parse_mode='Markdown')
            except Exception as e:
                print(f"Ошибка при обработке исторических аналогов: {e}")
                pass

    except Exception as e:
        error_msg = f"❌ Ошибка при обработке данных для {ticker}: {str(e)}"
        if update.message:
            await update.message.reply_text(error_msg, reply_markup=get_main_keyboard())
        else:
            await update.callback_query.message.reply_text(error_msg, reply_markup=get_main_keyboard())


# Обработчик персональных рекомендаций
async def send_personal_recommendations(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        if not os.path.exists(RESULTS_PATH):
            await update.message.reply_text("❌ Файл с результатами не найден.")
            return

        df = pd.read_csv(RESULTS_PATH)

        # Преобразуем ожидаемую доходность в числовой формат
        df['expected_return'] = pd.to_numeric(df['expected_return'], errors='coerce')

        # Преобразуем вероятность роста в числовой формат
        df['ensemble_growth_prob'] = pd.to_numeric(df['ensemble_growth_prob'], errors='coerce')

        # Добавляем составной скор, если его нет
        if 'composite_score' not in df.columns or df['composite_score'].isna().all():
            df['composite_score'] = df.apply(
                lambda x: calculate_composite_score(x['expected_return'], x['ensemble_growth_prob']),
                axis=1
            )

        # Фильтруем акции с высокой ожидаемой доходностью и вероятностью
        recommendations = df[
            (df['expected_return'] > 2.0) &
            (df['ensemble_growth_prob'] > 60)
            ].sort_values(by='composite_score', ascending=False)

        if recommendations.empty:
            # Если нет сильных рекомендаций, ищем умеренные
            recommendations = df[
                (df['expected_return'] > 0.5) &
                (df['ensemble_growth_prob'] > 50)
                ].sort_values(by='composite_score', ascending=False)

        message = "💡 *ПЕРСОНАЛЬНЫЕ РЕКОМЕНДАЦИИ*\n\n"

        if recommendations.empty:
            message += "К сожалению, в данный момент нет акций, рекомендованных к покупке.\n"
            message += "Рекомендуем понаблюдать за рынком и проверить позже."
        else:
            message += "На основе вашего профиля риска мы рекомендуем следующие акции:\n\n"

            for idx, row in recommendations.head(3).iterrows():
                # Используем отдельные колонки для границ интервала
                try:
                    lower_bound = float(row['expected_return_lower'])
                    upper_bound = float(row['expected_return_upper'])
                except:
                    try:
                        # Пытаемся распарсить кортеж из expected_return_range
                        expected_range = str(row['expected_return_range']).replace('(', '').replace(')', '')
                        lower_bound_str, upper_bound_str = expected_range.split(',')
                        lower_bound = float(lower_bound_str.strip())
                        upper_bound = float(upper_bound_str.strip())
                    except:
                        # Если все методы не сработали, используем стандартные значения
                        lower_bound = max(0, float(row['expected_return']) * 0.5)
                        upper_bound = float(row['expected_return']) * 1.5

                expected_return = float(row['expected_return'])
                growth_prob = float(row['ensemble_growth_prob'])

                message += f"📈 *{row['ticker']}*\n"
                message += f"• Ожидаемая доходность: *{expected_return:.2f}%*\n"
                message += f"• Доверительный интервал: [{lower_bound:.2f}% - {upper_bound:.2f}%]\n"
                message += f"• Вероятность роста: *{growth_prob:.1f}%*\n"
                message += f"• Составной скор: *{row['composite_score']:.1f}*\n\n"

            message += "✅ *Причина рекомендации:*\n"
            message += "Эти акции показывают высокую ожидаемую доходность и уверенность в прогнозе.\n\n"
            message += "📌 *Совет:* Для диверсификации рекомендуется распределить капитал между несколькими акциями из списка."

        if update.message:
            await update.message.reply_text(message, parse_mode='Markdown', reply_markup=get_main_keyboard())
        else:
            await update.callback_query.message.reply_text(message, parse_mode='Markdown',
                                                           reply_markup=get_main_keyboard())

    except Exception as e:
        error_msg = f"❌ Ошибка при генерации рекомендаций: {str(e)}"
        if update.message:
            await update.message.reply_text(error_msg, reply_markup=get_main_keyboard())
        else:
            await update.callback_query.message.reply_text(error_msg, reply_markup=get_main_keyboard())


# Основной обработчик текстовых сообщений
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text

    # Проверяем состояние пользователя
    if context.user_data.get('state') == CHOOSING_TICKER:
        await send_stock_forecast(update, context, text)
        context.user_data.clear()  # Сбрасываем состояние
        return

    # Обрабатываем команды - сравниваем без учета регистра
    if text == BUTTON_TOP or text.lower() == "/top" or text.lower().startswith("/top@"):
        await send_top_stocks(update, context)
    elif text == BUTTON_STRATEGY or text.lower() == "/strategy" or text.lower().startswith("/strategy@"):
        await send_strategy_performance(update, context)
    elif text == BUTTON_STOCKS or text.lower() == "/stock" or text.lower().startswith("/stock@"):
        await choose_stock(update, context)
    elif text == BUTTON_RECOMMEND or text.lower() == "/recommend" or text.lower().startswith("/recommend@"):
        await send_personal_recommendations(update, context)
    elif text == BUTTON_HELP or text.lower() == "/help" or text.lower().startswith("/help@"):
        await help_command(update, context)
    else:
        # Проверяем, не является ли текст тикером
        if text.upper() in [t.upper() for t in TICKERS]:
            await send_stock_forecast(update, context, text.upper())
        else:
            await update.message.reply_text(
                "❓ Неизвестная команда. Используйте кнопки ниже или введите тикер акции.",
                reply_markup=get_main_keyboard()
            )


# Основная функция
async def main() -> None:
    application = ApplicationBuilder().token(TG_BOT_TOKEN).build()

    # Регистрация команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("top", send_top_stocks))
    application.add_handler(CommandHandler("strategy", send_strategy_performance))
    application.add_handler(CommandHandler("stock", choose_stock))
    application.add_handler(CommandHandler("recommend", send_personal_recommendations))

    # Общий обработчик текста
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Запуск бота
    print("🚀 Бот запущен и готов к работе!")
    await application.run_polling()


# Запуск
if __name__ == '__main__':
    import asyncio

    asyncio.run(main())