import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def download_and_prepare_data(ticker, start_date, end_date, interval):
    """
    Загружает данные с Yahoo Finance и подготавливает их для модели.
    """
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    
    # Убираем мультииндекс, если он есть
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)  # Убираем уровень 'Ticker'

    data = data[['Open', 'High', 'Low', 'Close']]  # Используем только OHLC данные
    data.dropna(inplace=True)  # Удаляем пропущенные данные

    # Фильтруем выходные (оставляем только понедельник-пятницу)
    if interval == '1d':
        data = data[data.index.dayofweek < 5]

    # Масштабирование данных
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])
    return data, scaled_data, scaler