# config.py

import logging

# Настройка логирования
logging.basicConfig(filename='app.log', level=logging.INFO)

# Пользовательские входные данные для временного интервала
ticker = 'EURUSD=X'  # Тикер для загрузки данных
start_date = '2024-01-25'  # Начальная дата
end_date = '2025-01-20'  # Конечная дата
interval = '1h'  # Интервал данных (1d, 1h, 15m и т.д.)

# Проверка допустимости интервала
valid_intervals = ['1m', '5m', '15m', '30m', '1h', '1d']

# Установка future_steps в зависимости от интервала
if interval in ['1m', '5m', '15m', '30m', '1h']:
    future_steps = 32  # Количество шагов для прогнозирования
else:
    future_steps = 32

time_step = 512  # Размер временного окна для модели LSTM
