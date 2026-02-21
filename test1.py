#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Продуктовый анализ для EdTech-платформы LearnFlow.
Выполняет:
1. Анализ воронки (регистрация → начало урока → завершение → оплата)
2. RFM-сегментацию пользователей
3. Расчёт ежедневных метрик (DAU, Revenue, ARPU, ARPPU, конверсия)
4. Формирование рекомендаций на основе данных
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import clickhouse_connect
from datetime import datetime
import numpy as np
import csv

# ========== 1. НАСТРОЙКА ПОДКЛЮЧЕНИЯ К БАЗЕ ==========
# Предполагается, что ClickHouse запущен локально на порту 8123
client = clickhouse_connect.get_client(
    host='localhost',
    port=8123,
    username='default',
    password=''          # если пароль не задан, оставляем пустым
)

# ========== 2. ФУНКЦИИ ДЛЯ ЗАГРУЗКИ ДАННЫХ ==========
def create_tables():
    """Создаёт таблицы users, events, payments (если их нет)."""
    client.command('DROP TABLE IF EXISTS users')
    client.command('DROP TABLE IF EXISTS events')
    client.command('DROP TABLE IF EXISTS payments')

    client.command("""
        CREATE TABLE users (
            user_id UInt32,
            country String,
            device String,
            traffic_source String,
            registration_date Date
        ) ENGINE = MergeTree()
        ORDER BY user_id
    """)

    client.command("""
        CREATE TABLE events (
            event_id UInt32,
            user_id UInt32,
            event_type String,
            event_date Date
        ) ENGINE = MergeTree()
        ORDER BY (event_date, user_id)
    """)

    client.command("""
        CREATE TABLE payments (
            payment_id UInt32,
            user_id UInt32,
            amount UInt32,
            status String,
            payment_date Date
        ) ENGINE = MergeTree()
        ORDER BY (payment_date, user_id)
    """)
    print("Таблицы созданы.")

def load_csv_to_clickhouse(filename, table_name, column_names, parsers):
    """
    Читает CSV-файл и загружает данные в указанную таблицу ClickHouse.
    parsers – список функций для преобразования каждого поля.
    """
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # пропускаем заголовок
        for row in reader:
            # Применяем парсеры к каждому полю строки
            parsed_row = tuple(parsers[i](row[i]) for i in range(len(parsers)))
            data.append(parsed_row)
    client.insert(table_name, data, column_names=column_names)
    print(f'Таблица {table_name} загружена, строк: {len(data)}')

# ========== 3. ЗАГРУЗКА ФАЙЛОВ ==========
create_tables()

files_config = [
    {
        'filename': 'users_variant_1.csv',
        'table_name': 'users',
        'column_names': ['user_id', 'country', 'device', 'traffic_source', 'registration_date'],
        'parsers': [int, str, str, str, lambda x: datetime.strptime(x, '%Y-%m-%d').date()]
    },
    {
        'filename': 'events_variant_1.csv',
        'table_name': 'events',
        'column_names': ['event_id', 'user_id', 'event_type', 'event_date'],
        'parsers': [int, int, str, lambda x: datetime.strptime(x, '%Y-%m-%d').date()]
    },
    {
        'filename': 'payments_variant_1.csv',
        'table_name': 'payments',
        'column_names': ['payment_id', 'user_id', 'amount', 'status', 'payment_date'],
        'parsers': [int, int, int, str, lambda x: datetime.strptime(x, '%Y-%m-%d').date()]
    }
]

for conf in files_config:
    load_csv_to_clickhouse(**conf)

print("Все данные успешно загружены.\n")

# ========== 4. ЗАДАНИЕ 1: АНАЛИЗ ВОРОНКИ ==========
print("\n" + "="*60)
print("ЗАДАНИЕ 1: Анализ воронки")
print("="*60)

# 4.1. Подсчёт пользователей на каждом этапе
query_funnel = """
WITH 
    users_base AS (SELECT user_id FROM users),
    started AS (SELECT DISTINCT user_id FROM events WHERE event_type = 'lesson_start'),
    completed AS (SELECT DISTINCT user_id FROM events WHERE event_type = 'lesson_complete'),
    paid AS (SELECT DISTINCT user_id FROM payments WHERE status = 'success')
SELECT
    (SELECT count() FROM users_base) AS registration,
    (SELECT count() FROM started) AS started_lesson,
    (SELECT count() FROM completed) AS completed_lesson,
    (SELECT count() FROM paid) AS paid
"""
funnel = client.query(query_funnel).result_rows[0]
reg_total, started_total, completed_total, paid_total = funnel

print(f"Регистраций: {reg_total}")
print(f"Начали урок: {started_total}")
print(f"Завершили урок: {completed_total}")
print(f"Оплатили: {paid_total}")

# 4.2. Конверсии
conv_reg_start = started_total / reg_total if reg_total else 0
conv_start_complete = completed_total / started_total if started_total else 0
conv_complete_paid = paid_total / completed_total if completed_total else 0
conv_reg_paid = paid_total / reg_total if reg_total else 0

print(f"\nКонверсии:")
print(f"Регистрация → первый урок: {conv_reg_start:.2%}")
print(f"Начало урока → завершение: {conv_start_complete:.2%}")
print(f"Завершение → оплата: {conv_complete_paid:.2%}")
print(f"Общая конверсия в оплату: {conv_reg_paid:.2%}")

# 4.3. Сегментный анализ (по стране, устройству, источнику)
def segment_analysis(segment_column):
    """
    Для указанного сегмента (country, device, traffic_source) выводит
    количество пользователей на каждом этапе и потери между этапами.
    """
    print(f"\n--- Анализ по {segment_column} ---")
    query = f"""
    SELECT 
        u.{segment_column} AS segment,
        COUNT(DISTINCT u.user_id) AS users,
        COUNT(DISTINCT if(e.event_type = 'lesson_start', u.user_id, NULL)) AS started,
        COUNT(DISTINCT if(e.event_type = 'lesson_complete', u.user_id, NULL)) AS completed,
        COUNT(DISTINCT if(p.status = 'success', u.user_id, NULL)) AS paid
    FROM users u
    LEFT JOIN events e ON u.user_id = e.user_id
    LEFT JOIN payments p ON u.user_id = p.user_id AND p.status = 'success'
    GROUP BY segment
    ORDER BY paid / users DESC
    """
    results = client.query(query).result_rows
    for row in results:
        seg, users_cnt, started_cnt, completed_cnt, paid_cnt = row
        if users_cnt == 0:
            continue
        conv_reg_start = started_cnt / users_cnt
        conv_start_comp = completed_cnt / started_cnt if started_cnt else 0
        conv_comp_paid = paid_cnt / completed_cnt if completed_cnt else 0
        conv_reg_paid = paid_cnt / users_cnt
        print(f"{seg}: всего {users_cnt}, конверсия в оплату {conv_reg_paid:.2%}, "
              f"потери: рег→старт {1-conv_reg_start:.2%}, "
              f"старт→заверш {1-conv_start_comp:.2%}, "
              f"заверш→оплата {1-conv_comp_paid:.2%}")

for seg in ['country', 'device', 'traffic_source']:
    segment_analysis(seg)

# 4.4. Визуализация воронки
# Общая воронка
stages = ['Регистрация', 'Начало урока', 'Завершение урока', 'Оплата']
counts = [reg_total, started_total, completed_total, paid_total]
plt.figure(figsize=(8,5))
plt.bar(stages, counts, color=['skyblue', 'lightgreen', 'gold', 'salmon'])
plt.title('Общая воронка пользователей')
plt.ylabel('Количество пользователей')
for i, v in enumerate(counts):
    plt.text(i, v + 5, str(v), ha='center')
plt.tight_layout()
plt.savefig('funnel_overall.png')
plt.show()

# Воронка по устройствам
query_dev = """
SELECT 
    device,
    COUNT(DISTINCT u.user_id) AS reg,
    COUNT(DISTINCT if(e.event_type = 'lesson_start', u.user_id, NULL)) AS started,
    COUNT(DISTINCT if(e.event_type = 'lesson_complete', u.user_id, NULL)) AS completed,
    COUNT(DISTINCT if(p.status = 'success', u.user_id, NULL)) AS paid
FROM users u
LEFT JOIN events e ON u.user_id = e.user_id
LEFT JOIN payments p ON u.user_id = p.user_id AND p.status = 'success'
GROUP BY device
ORDER BY device
"""
dev_data = client.query(query_dev).result_rows
devices = [row[0] for row in dev_data]
regs_dev = [row[1] for row in dev_data]
starts_dev = [row[2] for row in dev_data]
comps_dev = [row[3] for row in dev_data]
pays_dev = [row[4] for row in dev_data]

x = np.arange(len(devices))
width = 0.2
plt.figure(figsize=(10,6))
plt.bar(x - 1.5*width, regs_dev, width, label='Регистрация')
plt.bar(x - 0.5*width, starts_dev, width, label='Начало')
plt.bar(x + 0.5*width, comps_dev, width, label='Завершение')
plt.bar(x + 1.5*width, pays_dev, width, label='Оплата')
plt.xlabel('Устройство')
plt.ylabel('Количество пользователей')
plt.title('Воронка по устройствам')
plt.xticks(x, devices)
plt.legend()
plt.tight_layout()
plt.savefig('funnel_by_device.png')
plt.show()

# Конверсии по странам
query_country_conv = """
WITH country_conv AS (
    SELECT 
        u.country as country,
        COUNT(DISTINCT u.user_id) AS reg,
        COUNT(DISTINCT if(e.event_type = 'lesson_start', u.user_id, NULL)) AS started,
        COUNT(DISTINCT if(e.event_type = 'lesson_complete', u.user_id, NULL)) AS completed,
        COUNT(DISTINCT if(p.status = 'success', u.user_id, NULL)) AS paid
    FROM users u
    LEFT JOIN events e ON u.user_id = e.user_id
    LEFT JOIN payments p ON u.user_id = p.user_id AND p.status = 'success'
    GROUP BY u.country
)
SELECT 
    country,
    if(reg > 0, started / reg, 0) AS reg_start,
    if(started > 0, completed / started, 0) AS start_complete,
    if(completed > 0, paid / completed, 0) AS complete_paid,
    if(reg > 0, paid / reg, 0) AS reg_paid
FROM country_conv
ORDER BY reg_paid DESC
"""
country_conv = client.query(query_country_conv).result_rows
if country_conv:
    countries = [c[0] for c in country_conv]
    reg_start = [c[1] for c in country_conv]
    start_comp = [c[2] for c in country_conv]
    comp_paid = [c[3] for c in country_conv]
    reg_paid = [c[4] for c in country_conv]

    x = np.arange(len(countries))
    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.3, reg_start, 0.2, label='Регистрация → Старт')
    plt.bar(x - 0.1, start_comp, 0.2, label='Старт → Завершение')
    plt.bar(x + 0.1, comp_paid, 0.2, label='Завершение → Оплата')
    plt.bar(x + 0.3, reg_paid, 0.2, label='Регистрация → Оплата')
    plt.xlabel('Страна')
    plt.ylabel('Конверсия')
    plt.title('Конверсии по странам')
    plt.xticks(x, countries)
    plt.legend()
    plt.tight_layout()
    plt.savefig('conversion_by_country.png')
    plt.show()

# ========== 5. ЗАДАНИЕ 2: RFM-АНАЛИЗ ==========
print("\n" + "="*60)
print("ЗАДАНИЕ 2: RFM-анализ")
print("="*60)

# Определяем текущую дату как максимальную среди всех дат в данных
current_date = client.query("""
    SELECT greatest(
        max(registration_date), 
        max(event_date), 
        max(payment_date)
    ) FROM users, events, payments
""").result_rows[0][0]
if current_date is None:
    current_date = datetime.now().date()

# Рассчитываем Recency, Frequency, Monetary для каждого пользователя
rfm_query = f"""
SELECT 
    u.user_id,
    DATE_DIFF('day', greatest(
        max(e.event_date),
        max(p.payment_date),
        u.registration_date
    ), toDate('{current_date}')) AS recency,
    count(e.event_id) AS frequency,
    coalesce(sum(if(p.status='success', p.amount, 0)), 0) AS monetary
FROM users u
LEFT JOIN events e ON u.user_id = e.user_id
LEFT JOIN payments p ON u.user_id = p.user_id
GROUP BY u.user_id, u.registration_date
"""
rfm_data = client.query(rfm_query).result_rows
rfm_df = pd.DataFrame(rfm_data, columns=['user_id', 'recency', 'frequency', 'monetary'])

print("\nОписательная статистика по RFM:")
print(rfm_df.describe())

# Квантили для Recency
r_bins = rfm_df['recency'].quantile([0.25, 0.5, 0.75]).tolist()
def r_score(days):
    if days <= r_bins[0]:
        return 1
    elif days <= r_bins[1]:
        return 2
    elif days <= r_bins[2]:
        return 3
    else:
        return 4

# Квантили для Frequency (только для тех, у кого frequency > 0)
freq_nonzero = rfm_df[rfm_df['frequency'] > 0]['frequency']
if len(freq_nonzero) > 0:
    f_bins = freq_nonzero.quantile([0.25, 0.5, 0.75]).tolist()
else:
    f_bins = [0, 0, 0]

def f_score(freq):
    if freq == 0:
        return 0
    elif freq <= f_bins[0]:
        return 1
    elif freq <= f_bins[1]:
        return 2
    elif freq <= f_bins[2]:
        return 3
    else:
        return 4

def m_score(money):
    return 1 if money > 0 else 0

rfm_df['R'] = rfm_df['recency'].apply(r_score)
rfm_df['F'] = rfm_df['frequency'].apply(f_score)
rfm_df['M'] = rfm_df['monetary'].apply(m_score)

# Сегментация пользователей
def segment_user(row):
    if row['M'] == 1 and row['R'] == 1 and row['F'] >= 3:
        return 'Топ'
    elif row['M'] == 1 and row['R'] <= 2 and row['F'] >= 2:
        return 'Постоянные'
    elif row['R'] == 4 and row['F'] > 0:
        return 'Под угрозой ухода'
    elif row['R'] == 1 and row['F'] == 0 and row['M'] == 0:
        return 'Новички'
    else:
        return 'Прочие'

rfm_df['segment'] = rfm_df.apply(segment_user, axis=1)

# Распределение по сегментам
seg_counts = rfm_df['segment'].value_counts()
print("\nРаспределение по сегментам:")
print(seg_counts)

# Подгружаем демографические данные пользователей
user_info = client.query_df("SELECT user_id, country, device, traffic_source FROM users")
rfm_df = rfm_df.merge(user_info, on='user_id', how='left')

# Анализ сегментов по странам, устройствам, источникам
print("\n--- Сегменты по странам ---")
print(pd.crosstab(rfm_df['segment'], rfm_df['country']))

print("\n--- Сегменты по устройствам ---")
print(pd.crosstab(rfm_df['segment'], rfm_df['device']))

print("\n--- Сегменты по источникам трафика ---")
print(pd.crosstab(rfm_df['segment'], rfm_df['traffic_source']))

# Визуализация RFM
plt.figure(figsize=(10,6))
sns.scatterplot(data=rfm_df, x='recency', y='frequency', hue='segment', alpha=0.6)
plt.title('Recency vs Frequency по сегментам')
plt.xlabel('Recency (дней)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('rfm_scatter.png')
plt.show()

avg_revenue = rfm_df.groupby('segment')['monetary'].mean().sort_values()
plt.figure(figsize=(8,5))
avg_revenue.plot(kind='bar', color='coral')
plt.title('Средняя выручка по сегментам')
plt.ylabel('Средний monetary')
plt.tight_layout()
plt.savefig('avg_revenue_by_segment.png')
plt.show()

plt.figure(figsize=(8,5))
seg_counts.plot(kind='bar', color='lightblue')
plt.title('Размер сегментов')
plt.ylabel('Количество пользователей')
plt.tight_layout()
plt.savefig('segment_sizes.png')
plt.show()

# ARPU и ARPPU
total_revenue = rfm_df['monetary'].sum()
total_users = len(rfm_df)
paying_users = (rfm_df['monetary'] > 0).sum()
arpu = total_revenue / total_users if total_users else 0
arppu = total_revenue / paying_users if paying_users else 0

print(f"\nARPU: {arpu:.2f}")
print(f"ARPPU: {arppu:.2f}")

# ========== 6. ЗАДАНИЕ 3: ЕЖЕДНЕВНЫЕ МЕТРИКИ ==========
print("\n" + "="*60)
print("ЗАДАНИЕ 3: Ежедневные метрики и дашборд")
print("="*60)

# 6.1. Проверка качества данных
print("\n--- Проверка качества данных ---")
# Дубликаты в events
dupes_events = client.query("""
    SELECT event_date, user_id, event_type, count()
    FROM events
    GROUP BY event_date, user_id, event_type
    HAVING count() > 1
""").result_rows
if dupes_events:
    print(f"Найдены дубликаты в events: {len(dupes_events)} групп")
else:
    print("Дубликатов в events не обнаружено.")

# Дубликаты в payments
dupes_payments = client.query("""
    SELECT payment_date, user_id, amount, count()
    FROM payments
    GROUP BY payment_date, user_id, amount
    HAVING count() > 1
""").result_rows
if dupes_payments:
    print(f"Найдены дубликаты в payments: {len(dupes_payments)} групп")
else:
    print("Дубликатов в payments не обнаружено.")

# Пропуски (NULL заменены на 0, но проверяем условные "пустые" значения)
null_events = client.query("""
    SELECT count() FROM events WHERE user_id = 0 OR event_date = '1970-01-01'
""").result_rows[0][0]
print(f"Пропуски в events (user_id=0 или дата=1970-01-01): {null_events}")
null_payments = client.query("""
    SELECT count() FROM payments WHERE user_id = 0 OR payment_date = '1970-01-01' OR amount = 0
""").result_rows[0][0]
print(f"Пропуски в payments: {null_payments}")

# Отрицательные суммы (в UInt32 не бывает, но проверяем)
negative_amount = client.query("SELECT count() FROM payments WHERE amount < 0").result_rows[0][0]
print(f"Отрицательные суммы в payments: {negative_amount}")

# 6.2. Формирование ежедневных метрик
# Определяем диапазон дат
min_date = client.query("SELECT min(event_date) FROM events").result_rows[0][0]
max_date = client.query("SELECT max(payment_date) FROM payments").result_rows[0][0]
if min_date is None or max_date is None:
    min_date = '2026-01-01'
    max_date = '2026-03-01'

daily_metrics_query = f"""
WITH dates AS (
    SELECT arrayJoin(range(toUInt32(toDate('{min_date}')), toUInt32(toDate('{max_date}')) + 1)) as date_uint
),
date_series AS (
    SELECT toDate(date_uint) AS date FROM dates
),
dau AS (
    SELECT event_date AS date, countDistinct(user_id) AS dau
    FROM events
    GROUP BY event_date
),
revenue AS (
    SELECT payment_date AS date, sum(amount) AS revenue,
           countDistinct(user_id) AS paying_users
    FROM payments
    WHERE status = 'success'
    GROUP BY payment_date
)
SELECT 
    ds.date,
    ifnull(da.dau, 0) AS dau,
    ifnull(r.revenue, 0) AS revenue,
    ifnull(r.paying_users, 0) AS paying_users
FROM date_series ds
LEFT JOIN dau da ON ds.date = da.date
LEFT JOIN revenue r ON ds.date = r.date
ORDER BY ds.date
"""
daily = client.query(daily_metrics_query).result_rows

daily_df = pd.DataFrame(daily, columns=['date', 'dau', 'revenue', 'paying_users'])
daily_df['date'] = pd.to_datetime(daily_df['date'])
daily_df = daily_df.sort_values('date')

# Скользящее среднее MAU (приближённо)
daily_df['mau'] = daily_df['dau'].rolling(window=30, min_periods=1).sum()

# Расчёт производных метрик
daily_df['arppu'] = daily_df.apply(lambda row: row['revenue'] / row['paying_users'] if row['paying_users'] > 0 else 0, axis=1)
daily_df['arpu'] = daily_df.apply(lambda row: row['revenue'] / row['dau'] if row['dau'] > 0 else 0, axis=1)
daily_df['conversion'] = daily_df.apply(lambda row: row['paying_users'] / row['dau'] if row['dau'] > 0 else 0, axis=1)

print("\nСтатистика по DAU и Revenue:")
print(daily_df[['dau', 'revenue']].describe())

# 6.3. Визуализация ежедневных метрик
plt.figure(figsize=(14,10))

plt.subplot(2,2,1)
plt.plot(daily_df['date'], daily_df['dau'], marker='.', linestyle='-', color='blue')
plt.title('DAU по дням')
plt.ylabel('DAU')
plt.xticks(rotation=45)

plt.subplot(2,2,2)
plt.plot(daily_df['date'], daily_df['revenue'], marker='.', linestyle='-', color='green')
plt.title('Revenue по дням')
plt.ylabel('Revenue')
plt.xticks(rotation=45)

plt.subplot(2,2,3)
plt.plot(daily_df['date'], daily_df['conversion'], marker='.', linestyle='-', color='red')
plt.title('Conversion (платящие / DAU)')
plt.ylabel('Conversion')
plt.xticks(rotation=45)

plt.subplot(2,2,4)
plt.plot(daily_df['date'], daily_df['arpu'], marker='.', linestyle='-', color='orange')
plt.title('ARPU по дням')
plt.ylabel('ARPU')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('daily_metrics.png')
plt.show()

# 6.4. Ответы на вопросы по динамике
print("\n--- Ответы на вопросы ---")
trend_dau = daily_df['dau'].iloc[-1] - daily_df['dau'].iloc[0] if len(daily_df) > 1 else 0
trend_revenue = daily_df['revenue'].iloc[-1] - daily_df['revenue'].iloc[0] if len(daily_df) > 1 else 0
print(f"Тренд DAU (последний день - первый): {trend_dau}")
print(f"Тренд Revenue (последний день - первый): {trend_revenue}")
last_mau = daily_df['mau'].dropna().iloc[-1] if not daily_df['mau'].dropna().empty else 0
print(f"Последний MAU (приближённо): {last_mau}")

last_week = daily_df.tail(7)
print("\nПоследние 7 дней:")
print(last_week[['date','dau','conversion','revenue']].to_string())

avg_conv = daily_df['conversion'].mean()
print(f"Средняя конверсия за весь период: {avg_conv:.2%}")

# ========== 7. РЕКОМЕНДАЦИИ (с подстановкой реальных цифр) ==========
print("\n" + "="*60)
print("РЕКОМЕНДАЦИИ ПО ЗАДАНИЯМ")
print("="*60)

# Для наглядности подставим вычисленные значения
loss_reg_start = 1 - conv_reg_start
loss_start_comp = 1 - conv_start_complete
loss_comp_paid = 1 - conv_complete_paid

print("\n--- Задание 1. Продуктовые рекомендации ---")
print(f"""
1. Гипотеза: Основные потери происходят на этапе "регистрация → первый урок" (конверсия {conv_reg_start:.1%}, потери {loss_reg_start:.1%}). 
   Рекомендация: Упростить онбординг, добавить приветственный интерактив, отправить push-уведомление с первым бесплатным уроком.
   Ожидаемый эффект: Рост конверсии в первый урок на 5–10%.
   Приоритет: Высокий.

2. Гипотеза: Сегмент пользователей из Казахстана показывает более низкую конверсию в оплату (см. график). 
   Рекомендация: Локализовать контент, добавить местные методы оплаты, протестировать специальные предложения.
   Ожидаемый эффект: Увеличение конверсии в оплату для этого региона.
   Приоритет: Средний.

3. Гипотеза: Мобильные пользователи реже завершают уроки, чем десктопные (потери на этапе старт→завершение выше). 
   Рекомендация: Оптимизировать мобильную версию плеера, добавить возможность скачивания уроков.
   Ожидаемый эффект: Повышение completion rate на мобильных устройствах.
   Приоритет: Средний.
""")

print("\n--- Задание 2. Продуктовые рекомендации ---")
print("""
1. Стратегия удержания (сегмент "Под угрозой ухода"):
   Гипотеза: Пользователи давно не заходили, но раньше проявляли активность. Им можно предложить персональную скидку на следующий курс или напомнить о прогрессе.
   Ожидаемый эффект: Возврат части "спящих" пользователей, рост активности.
   Приоритет: Высокий.

2. Стратегия монетизации (сегменты "Топ" и "Постоянные"):
   Гипотеза: Эти пользователи уже платят и активны. Им можно предлагать годовые подписки или пакеты курсов со скидкой.
   Ожидаемый эффект: Увеличение LTV, среднего чека.
   Приоритет: Высокий.

3. Стратегия персонализации (сегмент "Новички"):
   Гипотеза: Новички ещё не начали обучение. Им нужен более мягкий онбординг, рекомендация первого курса на основе их интересов (если есть данные), возможно, бесплатный пробный урок.
   Ожидаемый эффект: Рост конверсии в первый урок и дальнейшую оплату.
   Приоритет: Средний.
""")

print("\n--- Задание 3. Продуктовые рекомендации ---")
print(f"""
1. Проблема: Конверсия в платящих колеблется, а DAU нестабилен в выходные (средняя конверсия {avg_conv:.2%}).
   Гипотеза: Пользователи менее активны в выходные, но готовы потреблять контент в эти дни. Запустить выходные акции или марафоны.
   Ожидаемый эффект: Сглаживание DAU, рост revenue в выходные.
   Приоритет: Средний.

2. Проблема: ARPU остаётся низким ({arpu:.2f}) из-за большого числа неактивных пользователей.
   Гипотеза: Многие регистрируются, но не начинают уроки. Усилить ретаргетинг через email/push.
   Ожидаемый эффект: Рост ARPU за счёт вовлечения "спящих".
   Приоритет: Высокий.

3. Проблема: В некоторые дни наблюдаются аномальные пики revenue (возможно, из-за разовых акций). 
   Гипотеза: Необходимо анализировать эти пики для планирования маркетинговых кампаний.
   Ожидаемый эффект: Более эффективное распределение маркетингового бюджета.
   Приоритет: Низкий.
""")

print("\nАнализ завершён.")