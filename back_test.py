# backtest.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math
from sklearn.metrics import mean_squared_error

# Пути
DATA_DIR = "data"
RESULTS_DIR = "results"
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, "all_predictions.csv")
OUTPUT_DIR = "backtest_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Настройки
LOOKBACK_DAYS = 365  # Анализируем последние 90 дней
HOLD_DAYS = 90  # Держим позицию 15 дней (по аналогии с прогнозом)
BUY_THRESHOLD = 60  # Покупаем, если ensemble_growth_prob > 60%
MIN_EXPECTED_RETURN = 18.0  # Минимальная ожидаемая доходность для покупки (2%)
CONFIDENCE_WEIGHT = 0.7  # Вес уверенности в ансамбле
RETURN_WEIGHT = 0.3  # Вес ожидаемой доходности


def load_and_prepare_data(ticker):
    """Загружаем и подготавливаем данные с признаками"""
    path = os.path.join(DATA_DIR, f"{ticker}_ml_ready.csv")
    if not os.path.exists(path):
        print(f"❌ Нет данных для {ticker}")
        return None

    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df.sort_values('date').reset_index(drop=True)
    return df


def calculate_composite_score(row):
    """
    Рассчитываем составной скор на основе вероятности роста и ожидаемой доходности
    Скор нормализован в диапазоне 0-100
    """
    # Нормализуем вероятность роста (уже в %)
    growth_prob_score = row['ensemble_growth_prob'] / 100.0

    # Нормализуем ожидаемую доходность (приводим к шкале 0-1)
    # Предполагаем, что доходность выше 10% - это отлично
    expected_return_score = min(row['expected_return'] / 10.0, 1.0)
    expected_return_score = max(expected_return_score, 0.0)

    # Взвешенное среднее
    composite_score = (CONFIDENCE_WEIGHT * growth_prob_score +
                       RETURN_WEIGHT * expected_return_score) * 100

    return composite_score


def simulate_trades(historical_data, predictions_df, ticker):
    """Симуляция торговли на основе исторических данных"""
    trades = []
    predictions = predictions_df[predictions_df['ticker'] == ticker]

    if predictions.empty:
        print(f"❌ Нет прогнозов для {ticker}")
        return trades

    pred_row = predictions.iloc[0]

    # Проверяем условия для покупки
    confidence_ok = pred_row['ensemble_growth_prob'] > BUY_THRESHOLD
    return_ok = pred_row['expected_return'] > MIN_EXPECTED_RETURN
    has_valid_data = not (pd.isna(pred_row['expected_return']) or
                          pred_row['expected_return'] == 0)

    # Формируем сигнал на основе составного скоринга
    if confidence_ok and return_ok and has_valid_data:
        signal = "BUY"
        confidence = pred_row['ensemble_growth_prob']
    else:
        signal = "HOLD"
        confidence = pred_row['ensemble_growth_prob']

    # Сортируем данные по дате
    historical_data = historical_data.sort_values('date')

    # Проверяем, есть ли данные для бэктеста
    if len(historical_data) < HOLD_DAYS + 1:
        print(f"⚠️ Недостаточно данных для {ticker} для бэктеста")
        return trades

    # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: используем данные, где есть как минимум HOLD_DAYS впереди
    # Берём запись, которая на HOLD_DAYS раньше последней
    entry_index = -1 - HOLD_DAYS
    entry_date = historical_data.iloc[entry_index]['date']
    entry_price = historical_data.iloc[entry_index]['close_price']

    # Выход через HOLD_DAYS
    exit_date = historical_data.iloc[-1]['date']
    exit_price = historical_data.iloc[-1]['close_price']

    # Рассчитываем доходность
    pnl = (exit_price - entry_price) / entry_price * 100
    trade_status = "COMPLETE"

    # Рассчитываем составной скор
    composite_score = calculate_composite_score(pred_row)

    trades.append({
        'ticker': ticker,
        'signal': signal,
        'entry_date': entry_date,
        'entry_price': entry_price,
        'exit_date': exit_date,
        'exit_price': exit_price,
        'pnl_pct': pnl,
        'hold_days': HOLD_DAYS,
        'confidence': confidence,
        'expected_return': pred_row['expected_return'],
        'composite_score': composite_score,
        'growth_prob': pred_row['ensemble_growth_prob'],
        'trade_status': trade_status
    })

    return trades


def analyze_strategy_performance(trades_df):
    """Детальный анализ эффективности стратегии"""
    if trades_df.empty:
        return {}

    # Фильтруем только завершенные сделки
    completed_trades = trades_df[trades_df['trade_status'] == 'COMPLETE']

    if completed_trades.empty:
        print("⚠️ Нет завершенных сделок для анализа")
        return {
            'total_trades': len(trades_df),
            'completed_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'breakeven_trades': 0,
            'win_rate_pct': 0.0,
            'loss_rate_pct': 0.0,
            'avg_win_pct': 0.0,
            'avg_loss_pct': 0.0,
            'profit_factor': 0.0,
            'avg_return_per_trade_pct': 0.0,
            'std_return_pct': 0.0,
            'max_drawdown_pct': 0.0,
            'expected_return_pct': 0.0,
            'sharpe_ratio': 0.0,
            'incomplete_trades': len(trades_df)
        }

    # Основные метрики
    total_trades = len(trades_df)
    completed_trades_count = len(completed_trades)
    winning_trades = len(completed_trades[completed_trades['pnl_pct'] > 0])
    losing_trades = len(completed_trades[completed_trades['pnl_pct'] < 0])
    breakeven_trades = len(completed_trades[completed_trades['pnl_pct'] == 0])

    win_rate = (winning_trades / completed_trades_count * 100) if completed_trades_count > 0 else 0
    loss_rate = (losing_trades / completed_trades_count * 100) if completed_trades_count > 0 else 0

    # Средние значения
    avg_win = completed_trades[completed_trades['pnl_pct'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
    avg_loss = abs(completed_trades[completed_trades['pnl_pct'] < 0]['pnl_pct'].mean()) if losing_trades > 0 else 0

    # Коэффициент прибыльности
    profit_factor = (avg_win * winning_trades) / (avg_loss * losing_trades) if (
                losing_trades > 0 and avg_loss > 0) else float('inf')

    # Средняя доходность
    avg_pnl = completed_trades['pnl_pct'].mean()
    std_pnl = completed_trades['pnl_pct'].std() if completed_trades_count > 1 else 0

    # Максимальная просадка
    max_drawdown = completed_trades['pnl_pct'].min() if not completed_trades['pnl_pct'].empty else 0

    # Ожидаемая доходность (математическое ожидание)
    expected_return = (win_rate / 100 * avg_win) - (loss_rate / 100 * avg_loss)

    # Шарп-коэффициент
    sharpe_ratio = avg_pnl / std_pnl if std_pnl > 0 else 0

    return {
        'total_trades': total_trades,
        'completed_trades': completed_trades_count,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'breakeven_trades': breakeven_trades,
        'win_rate_pct': round(win_rate, 2),
        'loss_rate_pct': round(loss_rate, 2),
        'avg_win_pct': round(avg_win, 2),
        'avg_loss_pct': round(avg_loss, 2),
        'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else float('inf'),
        'avg_return_per_trade_pct': round(avg_pnl, 2),
        'std_return_pct': round(std_pnl, 2),
        'max_drawdown_pct': round(max_drawdown, 2),
        'expected_return_pct': round(expected_return, 2),
        'sharpe_ratio': round(sharpe_ratio, 2),
        'incomplete_trades': total_trades - completed_trades_count
    }


def generate_detailed_report(trades_df, summary):
    """Генерация подробного отчета"""
    report_path = os.path.join(OUTPUT_DIR, "detailed_backtest_report.txt")

    with open(report_path, "w", encoding='utf-8') as f:
        f.write("ДЕТАЛЬНЫЙ ОТЧЕТ О BACKTEST'Е\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. ОБЩАЯ ИНФОРМАЦИЯ\n")
        f.write("-" * 50 + "\n")
        f.write(f"Период анализа: Последние {LOOKBACK_DAYS} дней\n")
        f.write(f"Горизонт прогноза: {HOLD_DAYS} дней\n")
        f.write(f"Порог уверенности: >{BUY_THRESHOLD}%\n")
        f.write(f"Минимальная ожидаемая доходность: >{MIN_EXPECTED_RETURN}%\n")
        f.write(f"Дата отчета: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("2. СТАТИСТИКА СДЕЛОК\n")
        f.write("-" * 50 + "\n")

        # Добавляем информацию о незавершенных сделках
        if 'incomplete_trades' in summary and summary['incomplete_trades'] > 0:
            f.write(f"• Всего сделок: {summary['total_trades']}\n")
            f.write(f"• Завершенных сделок: {summary['completed_trades']}\n")
            f.write(f"• Незавершенных сделок: {summary['incomplete_trades']}\n\n")
        else:
            for k, v in summary.items():
                f.write(f"{k}: {v}\n")

        f.write("\n3. АНАЛИЗ ЭФФЕКТИВНОСТИ\n")
        f.write("-" * 50 + "\n")

        # Проверяем наличие незавершенных сделок
        if 'incomplete_trades' in summary and summary['incomplete_trades'] == summary['total_trades']:
            f.write("⚠️ Все сделки незавершены. Возможно, недостаточно исторических данных.\n")
            f.write(f"Проверьте, что у вас есть данные за последние {HOLD_DAYS} дней.\n")
        elif summary['win_rate_pct'] > 55:
            f.write("✅ Высокая прибыльность: Win Rate > 55%\n")
        elif summary['win_rate_pct'] > 45:
            f.write("⚠️ Умеренная прибыльность: Win Rate > 45%\n")
        else:
            f.write("❌ Низкая прибыльность: Win Rate < 45%\n")

        if 'profit_factor' in summary and summary['profit_factor'] != float('inf'):
            if summary['profit_factor'] > 1.5:
                f.write("✅ Высокая прибыльность: Profit Factor > 1.5\n")
            elif summary['profit_factor'] > 1.0:
                f.write("⚠️ Умеренная прибыльность: Profit Factor > 1.0\n")
            else:
                f.write("❌ Низкая прибыльность: Profit Factor < 1.0\n")

        if summary['sharpe_ratio'] > 1.0:
            f.write("✅ Высокая эффективность: Sharpe Ratio > 1.0\n")
        elif summary['sharpe_ratio'] > 0.5:
            f.write("⚠️ Умеренная эффективность: Sharpe Ratio > 0.5\n")
        else:
            f.write("❌ Низкая эффективность: Sharpe Ratio < 0.5\n")

        f.write(f"\n4. ТОП-5 ПОЗИЦИЙ ПО СОСТАВНОМУ СКОРУ\n")
        f.write("-" * 50 + "\n")

        # Сортируем только завершенные сделки
        completed_trades = trades_df[trades_df['trade_status'] == 'COMPLETE']
        if not completed_trades.empty:
            top_positions = completed_trades.nlargest(5, 'composite_score')
            f.write(
                f"{'Тикер':<8} {'Сигнал':<6} {'Доходность':<12} {'Уверенность':<12} {'Ож.доходн.':<12} {'Скор':<8}\n")
            f.write("-" * 70 + "\n")

            for _, row in top_positions.iterrows():
                f.write(f"{row['ticker']:<8} {row['signal']:<6} {row['pnl_pct']:>10.2f}% "
                        f"{row['confidence']:>10.2f}% {row['expected_return']:>10.2f}% {row['composite_score']:>6.1f}\n")
        else:
            f.write("Нет завершенных сделок для отображения\n")

        f.write(f"\n5. РАСПРЕДЕЛЕНИЕ СДЕЛОК ПО ТИКЕРАМ\n")
        f.write("-" * 50 + "\n")
        ticker_summary = completed_trades.groupby('ticker').agg({
            'pnl_pct': ['count', 'mean', 'std'],
            'signal': 'first'
        }).round(2)

        for ticker in ticker_summary.index:
            count = ticker_summary.loc[ticker, ('pnl_pct', 'count')]
            avg_return = ticker_summary.loc[ticker, ('pnl_pct', 'mean')]
            signal = ticker_summary.loc[ticker, ('signal', 'first')]
            f.write(f"{ticker}: {count} сделок, средняя доходность {avg_return:.2f}%, сигнал: {signal}\n")


def run_backtest():
    """Запуск backtest'а для всех тикеров"""
    if not os.path.exists(PREDICTIONS_FILE):
        print("❌ Файл прогнозов не найден. Сначала запустите train_models.py")
        return

    predictions_df = pd.read_csv(PREDICTIONS_FILE)

    # Проверяем наличие необходимых колонок
    required_columns = ['expected_return', 'ensemble_growth_prob']
    missing_columns = [col for col in required_columns if col not in predictions_df.columns]

    if missing_columns:
        print(f"❌ Отсутствуют необходимые колонки в файле прогнозов: {missing_columns}")
        print("Пожалуйста, убедитесь, что train_models.py корректно рассчитывает ожидаемую доходность")
        return

    tickers = predictions_df['ticker'].unique()

    all_trades = []

    print("🔍 Запуск backtest'а...")
    for ticker in tickers:
        print(f"🚀 Тестирование стратегии для {ticker}...")

        data = load_and_prepare_data(ticker)
        if data is None:
            continue

        trades = simulate_trades(data, predictions_df, ticker)
        all_trades.extend(trades)

    if not all_trades:
        print("❌ Не удалось сгенерировать сделки.")
        return

    trades_df = pd.DataFrame(all_trades)
    trades_df.to_csv(os.path.join(OUTPUT_DIR, "trades_simulation.csv"), index=False)

    # Анализ эффективности
    summary = analyze_strategy_performance(trades_df)

    if not summary:
        print("❌ Не удалось рассчитать метрики эффективности.")
        return

    # Сохранение и вывод результатов
    summary_file = os.path.join(OUTPUT_DIR, "backtest_summary.txt")
    with open(summary_file, "w", encoding='utf-8') as f:
        f.write("РЕЗУЛЬТАТЫ BACKTEST'А\n")
        f.write("=" * 50 + "\n")

        # Форматируем вывод в зависимости от наличия незавершенных сделок
        if 'incomplete_trades' in summary and summary['incomplete_trades'] == summary['total_trades']:
            f.write("⚠️ Все сделки незавершены. Возможно, недостаточно исторических данных.\n")
            f.write(f"Проверьте, что у вас есть данные за последние {HOLD_DAYS} дней.\n")
        else:
            for k, v in summary.items():
                f.write(f"{k}: {v}\n")

    # Форматированный вывод в консоль
    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТЫ BACKTEST'А")
    print("=" * 60)

    if 'incomplete_trades' in summary and summary['incomplete_trades'] == summary['total_trades']:
        print("⚠️ Все сделки незавершены. Возможно, недостаточно исторических данных.")
        print(f"Проверьте, что у вас есть данные за последние {HOLD_DAYS} дней.")
    else:
        for k, v in summary.items():
            print(f"{k:<35}: {v}")

    print("=" * 60)

    # Графики
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Распределение сделок (только завершенные)
    completed_trades = trades_df[trades_df['trade_status'] == 'COMPLETE']
    if not completed_trades.empty:
        labels = ['Прибыльные', 'Убыточные', 'Безубыточные']
        winning = len(completed_trades[completed_trades['pnl_pct'] > 0])
        losing = len(completed_trades[completed_trades['pnl_pct'] < 0])
        breakeven = len(completed_trades[completed_trades['pnl_pct'] == 0])
        sizes = [winning, losing, breakeven]
        colors = ['green', 'red', 'gray']
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Распределение завершенных сделок')
    else:
        ax1.text(0.5, 0.5, 'Нет завершенных сделок',
                 ha='center', va='center', fontsize=12)
        ax1.set_title('Распределение сделок')

    # 2. Гистограмма доходности
    if not completed_trades.empty and not completed_trades['pnl_pct'].isna().all():
        ax2.hist(completed_trades['pnl_pct'].dropna(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Доходность (%)')
        ax2.set_ylabel('Частота')
        ax2.set_title('Распределение доходности по сделкам')
    else:
        ax2.text(0.5, 0.5, 'Нет данных для построения гистограммы',
                 ha='center', va='center', fontsize=12)
        ax2.set_title('Распределение доходности')

    # 3. Сравнение ожидаемой и реальной доходности
    if not completed_trades.empty:
        valid_trades = completed_trades[completed_trades['pnl_pct'].notna()]
        if not valid_trades.empty:
            ax3.scatter(valid_trades['expected_return'], valid_trades['pnl_pct'], alpha=0.6)
            ax3.plot([valid_trades['expected_return'].min(), valid_trades['expected_return'].max()],
                     [valid_trades['expected_return'].min(), valid_trades['expected_return'].max()],
                     'r--', linewidth=2)
            ax3.set_xlabel('Ожидаемая доходность (%)')
            ax3.set_ylabel('Реальная доходность (%)')
            ax3.set_title('Ожидаемая vs Реальная доходность')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Нет данных для сравнения',
                     ha='center', va='center', fontsize=12)
            ax3.set_title('Ожидаемая vs Реальная доходность')
    else:
        ax3.text(0.5, 0.5, 'Нет завершенных сделок',
                 ha='center', va='center', fontsize=12)
        ax3.set_title('Ожидаемая vs Реальная доходность')

    # 4. Составной скор по тикерам
    if not completed_trades.empty:
        ticker_scores = completed_trades.groupby('ticker')['composite_score'].mean().sort_values(ascending=False)
        if not ticker_scores.empty:
            bars = ax4.bar(ticker_scores.index, ticker_scores.values, color='orange', alpha=0.7)
            ax4.set_xlabel('Тикер')
            ax4.set_ylabel('Составной скор')
            ax4.set_title('Составной скор по тикерам')
            ax4.tick_params(axis='x', rotation=45)

            # Добавляем значения над столбцами
            for bar, score in zip(bars, ticker_scores.values):
                ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                         f'{score:.1f}', ha='center', va='bottom')
        else:
            ax4.text(0.5, 0.5, 'Нет данных',
                     ha='center', va='center', fontsize=12)
            ax4.set_title('Составной скор по тикерам')
    else:
        ax4.text(0.5, 0.5, 'Нет завершенных сделок',
                 ha='center', va='center', fontsize=12)
        ax4.set_title('Составной скор по тикерам')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comprehensive_backtest_analysis.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # Генерация подробного отчета
    generate_detailed_report(trades_df, summary)
    print(f"\n✅ Подробный отчет сохранен в {OUTPUT_DIR}/detailed_backtest_report.txt")
    print(f"✅ Графики сохранены в {OUTPUT_DIR}/")

    return trades_df, summary


if __name__ == "__main__":
    run_backtest()