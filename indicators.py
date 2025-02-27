# indicators.py

def calculate_atr(data, period=14):
    """
    Рассчитывает Average True Range (ATR).
    """
    data['H-L'] = data['High'] - data['Low']
    data['H-PC'] = abs(data['High'] - data['Close'].shift(1))
    data['L-PC'] = abs(data['Low'] - data['Close'].shift(1))
    data['TR'] = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)  # Используем max для всех строк
    data['ATR'] = data['TR'].rolling(window=period).mean()
    return data

def calculate_bollinger_bands(data, window=20, std_dev=3):
    """
    Рассчитывает Bollinger Bands.
    """
    data['MA'] = data['Close'].rolling(window=window).mean()
    data['STD'] = data['Close'].rolling(window=window).std()
    data['BB_upper'] = data['MA'] + std_dev * data['STD']
    data['BB_lower'] = data['MA'] - std_dev * data['STD']
    return data