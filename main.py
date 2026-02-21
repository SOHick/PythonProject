import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import clickhouse_connect
from datetime import datetime
import numpy as np

# ========== НАСТРОЙКИ ПОДКЛЮЧЕНИЯ ==========
CLICKHOUSE_HOST = 'localhost'
CLICKHOUSE_PORT = 8123
DATABASE = 'default'  # или создайте свою БД, например 'learnflow'

# Подключаемся к ClickHouse (без пароля)
client = clickhouse_connect.get_client(
    host='localhost',
    port=8123,
    username='default',
    password=''  # если пароль установлен
)

# ========== 1. СОЗДАНИЕ ТАБЛИЦ ==========
# Удаляем старые таблицы (если существуют)
client.command('DROP TABLE IF EXISTS users')
client.command('DROP TABLE IF EXISTS events')
client.command('DROP TABLE IF EXISTS payments')

# Создаём таблицы с движком MergeTree
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

print("Таблицы успешно созданы.")
