import dash
from dash import html, dash_table, dcc
import plotly.graph_objects as go
from data_loader import download_and_prepare_data
from indicators import calculate_atr, calculate_bollinger_bands
from predictor import make_future_predictions
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

# Импорт конфигурации
from config import ticker, start_date, end_date, interval, valid_intervals, future_steps, time_step

# Инициализация Dash-приложения
app = dash.Dash(__name__, assets_folder='assets', index_string='''
<!DOCTYPE html>
<html>
  <head>
    <link rel="manifest" href="/assets/manifest.json">
    <script>
      if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/assets/service-worker.js').then(function(registration) {
          console.log('ServiceWorker registration successful with scope: ', registration.scope);
        }).catch(function(err) {
          console.log('ServiceWorker registration failed: ', err);
        });
      }
    </script>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>
''')

# Список доступных моделей
available_models = [
    {'label': '---ALL', 'value': 'script_trained_model1.h5'},
    {'label': ' EURUSD', 'value': 'script_trained_model1.h5'},
    {'label': ' GC=F', 'value': 'Script/script_trained_model1.h5'},
    {'label': ' BTC-USD', 'value': 'script_trained_model1.h5'},
]

# Установка темной темы
app.layout = html.Div([
    html.H1(f"LSTM Neuro Forecast", style={'color': 'white', 'textAlign': 'center'}),
    
    # Поля ввода и кнопки в верхней части
    html.Div([
        dcc.Input(id='ticker-input', type='text', placeholder='ticker_GC=F', style={'width': '20%', 'margin': '2px', 'color': 'black', 'backgroundColor': 'white'}),
        dcc.Input(id='start-date-input', type='text', placeholder='start_YYYY-MM-DD', style={'width': '20%', 'margin': '2px', 'color': 'black', 'backgroundColor': 'white'}),
        dcc.Input(id='end-date-input', type='text', placeholder='end_YYYY-MM-DD', style={'width': '20%', 'margin': '2px', 'color': 'black', 'backgroundColor': 'white'}),
        dcc.Input(id='interval-input', type='text', placeholder='interval_15m', style={'width': '20%', 'margin': '2px', 'color': 'black', 'backgroundColor': 'white'}),
        html.Button('Update', id='update-button', n_clicks=0, style={'width': '20%', 'margin': '5px', 'backgroundColor': 'blue', 'color': 'white'}),
        html.Button('Load Last 50d (15m)', id='load-last-month-button', n_clicks=0, style={'width': '20%', 'margin': '5px', 'backgroundColor': 'green', 'color': 'white'}),
        html.Button('Load Last Year (1h)', id='load-last-year-button', n_clicks=0, style={'width': '20%', 'margin': '5px', 'backgroundColor': 'purple', 'color': 'white'})
    ], style={'textAlign': 'center'}),
    
    # Основной график
    dcc.Graph(id='main-graph', style={'height': '90vh'}),
    
    # Таблицы рисков и настроек внизу
    html.Div([
        # Таблица рисков
        dash_table.DataTable(
            id='risk-table',
            columns=[{'name': 'Risk Metric', 'id': 'metric'}, {'name': 'Value', 'id': 'value'}],
            data=[],
            style_table={'width': '100%', 'overflowX': 'auto', 'color': '', 'backgroundColor': 'black', 'margin': '5px'},
            style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
            style_cell={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'}
        ),
        # Таблица настроек
        dash_table.DataTable(
            id='settings-table',
            columns=[{'name': 'Parameter', 'id': 'parameter'}, {'name': 'Value', 'id': 'value', 'editable': True}],
            data=[
                {'parameter': 'future_steps', 'value': future_steps},
                {'parameter': 'time_step', 'value': time_step},
                {'parameter': 'display_per', 'value': 300},
                {'parameter': 'ATR period', 'value': 14},
                {'parameter': 'BB window', 'value': 20},
                {'parameter': 'BB std_dev', 'value': 2},
                {'parameter': 'ATR stop X', 'value': 500},
                {'parameter': 'Vol stop X', 'value': 100}
            ],
            style_table={'width': '100%', 'overflowX': 'auto', 'color': 'white', 'backgroundColor': 'black', 'margin': '5px'},
            style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
            style_cell={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'}
        )
    ], style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px'}),
    
    # Выпадающий список моделей
    html.Div([
        dcc.Dropdown(
            id='model-dropdown',
            options=available_models,
            value=available_models[0]['value'],  # Значение по умолчанию
            placeholder="Select a model",
            style={'width': '70%', 'margin': '5px', 'color': 'black', 'backgroundColor': 'green'}
        )
    ], style={'textAlign': 'center', 'marginBottom': '2px'})
], style={'backgroundColor': 'black', 'color': 'white'})

# Callback для обновления графиков и таблицы
@app.callback(
    [dash.dependencies.Output('main-graph', 'figure'),
     dash.dependencies.Output('risk-table', 'data')],
    [dash.dependencies.Input('update-button', 'n_clicks'),
     dash.dependencies.Input('load-last-month-button', 'n_clicks'),
     dash.dependencies.Input('load-last-year-button', 'n_clicks'),
     dash.dependencies.Input('settings-table', 'data'),
     dash.dependencies.Input('model-dropdown', 'value')],  # Добавляем вход для выбора модели
    [dash.dependencies.State('ticker-input', 'value'),
     dash.dependencies.State('start-date-input', 'value'),
     dash.dependencies.State('end-date-input', 'value'),
     dash.dependencies.State('interval-input', 'value')]
)
def update_graphs(update_clicks, load_last_month_clicks, load_last_year_clicks, settings_data, selected_model, ticker_input, start_date_input, end_date_input, interval_input):
    """
    Обновляет графики и таблицу рисков при нажатии кнопки "Update", "Load Last Month (15m)" или "Load Last Year (1h)".
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        return {}, []
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Если нажата кнопка "Load Last Month (15m)"
    if button_id == 'load-last-month-button':
        end_date_input = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        start_date_input = (datetime.now() - timedelta(days=50)).strftime('%Y-%m-%d')
        interval_input = '15m'
        ticker_input = ticker_input if ticker_input else 'EURUSD=X'

    # Если нажата кнопка "Load Last Year (1h)"
    if button_id == 'load-last-year-button':
        end_date_input = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        start_date_input = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        interval_input = '1h'
        ticker_input = ticker_input if ticker_input else 'EURUSD=X'

    # Используем входные данные или значения по умолчанию
    ticker_input = ticker_input if ticker_input else ticker
    start_date_input = start_date_input if start_date_input else start_date
    end_date_input = end_date_input if end_date_input else end_date
    interval_input = interval_input if interval_input else interval

    # Проверка интервала
    if interval_input not in valid_intervals:
        raise ValueError(f"Invalid timeframe. Choose from {valid_intervals}")

    # Обновление параметров из таблицы настроек
    future_steps = int(settings_data[0]['value'])
    time_step = int(settings_data[1]['value'])
    display_periods = int(settings_data[2]['value'])
    atr_period = int(settings_data[3]['value'])
    bb_window = int(settings_data[4]['value'])
    bb_std_dev = int(settings_data[5]['value'])
    atr_stop_coefficient = float(settings_data[6]['value'])
    volatility_stop_coefficient = float(settings_data[7]['value'])

    # Загрузка данных
    data, scaled_data, scaler = download_and_prepare_data(ticker_input, start_date_input, end_date_input, interval_input)
    data = calculate_atr(data, period=atr_period)
    data = calculate_bollinger_bands(data, window=bb_window, std_dev=bb_std_dev)

    # Загрузка модели
    if os.path.exists(selected_model):
        model = load_model(selected_model)  # Используем выбранную модель
    else:
        raise FileNotFoundError(f"Model file {selected_model} not found. Please train the model first.")

    # Прогнозирование будущих значений
    last_sequence = scaled_data[-time_step:]

    # Дополнение last_sequence нулями
    if len(last_sequence) < time_step:
        padding = np.zeros((time_step - len(last_sequence), 1))
        last_sequence = np.vstack((padding, last_sequence))

    future_predictions = make_future_predictions(model, last_sequence, future_steps, scaler)

    # Подготовка будущих дат
    last_date = data.index[-1]
    if interval_input == '1d':
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps, freq='D')
    elif interval_input == '1h':
        future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=future_steps, freq='H')
    else:
        future_dates = pd.date_range(start=last_date + pd.Timedelta(minutes=int(interval_input[:-1])), periods=future_steps, freq=f'{interval_input[:-1]}T')

    # Создание DataFrame для прогнозов
    future_data = pd.DataFrame(index=future_dates, data=future_predictions, columns=['Close'])

    # Ограничиваем отображение последними display_periods периодами
    historical_data = data.iloc[-display_periods:]

    # Объединяем исторические данные и прогнозы
    combined_data = pd.concat([historical_data, future_data])

    # Сброс индекса для использования числового индекса в графике
    combined_data.reset_index(drop=True, inplace=True)

    # Расчет метрик ошибок
    real_data = data['Close'][-future_steps:].values
    mae = mean_absolute_error(real_data, future_predictions)
    mse = mean_squared_error(real_data, future_predictions)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((real_data - future_predictions.flatten()) / real_data)) * 100
    relative_mae = mae / np.mean(real_data)

    # Новые расчеты для VAR, ATR и VOLATILITY
    var_95_all = data['Close'].pct_change().quantile(0.05)
    var_95_last = data['Close'].pct_change().iloc[-1].item()
    atr_last = data['ATR'].iloc[-1]
    volatility_all = data['Close'].pct_change().std() * (252 ** 0.5)
    stop_loss_var = abs(var_95_all) * 100
    stop_loss_atr = atr_last / data['Close'].iloc[-1] * atr_stop_coefficient
    stop_loss_volatility = volatility_all * volatility_stop_coefficient

    # Определение направления прогноза на основе 10 шагов вперед
    future_close_10 = future_predictions[9][0] 
    last_close = data['Close'].iloc[-1].item()  # Извлекаем конкретное значение
    print(type(future_close_10))  # Проверяем тип данных future_close_10
    print(type(last_close))       # Проверяем тип данных last_close
    print(future_close_10)        # Выводим значение future_close_10
    print(last_close)             # Выводим значение last_close
    if future_close_10 > last_close:
        stop_loss_var_line = last_close * (1 - stop_loss_var / 100)
        stop_loss_atr_line = last_close * (1 - stop_loss_atr / 100)
        stop_loss_volatility_line = last_close * (1 - stop_loss_volatility / 100)
    else:
        stop_loss_var_line = last_close * (1 + stop_loss_var / 100)
        stop_loss_atr_line = last_close * (1 + stop_loss_atr / 100)
        stop_loss_volatility_line = last_close * (1 + stop_loss_volatility / 100)
        

    # Основной график с японскими свечами, Bollinger Bands и прогнозами
    fig = go.Figure(data=[
        go.Candlestick(
            x=combined_data.index[:len(historical_data)],
            open=historical_data['Open'],
            high=historical_data['High'],
            low=historical_data['Low'],
            close=historical_data['Close'],
            name='Candlestick'
        ),
        go.Scatter(
            x=combined_data.index,
            y=combined_data['Close'],
            name='Future Predictions',
            line=dict(color='orange', width=1)
        ),
        go.Scatter(
            x=combined_data.index,
            y=[stop_loss_var_line] * len(combined_data),
            name=f'Enter (lvl): {stop_loss_var.item():.2f}%',
            line=dict(color='red', width=1, dash='dash')
        ),
        go.Scatter(
            x=combined_data.index,
            y=[stop_loss_atr_line] * len(combined_data),
            name=f'Stop(ATR): {stop_loss_atr.item():.2f}%',
            line=dict(color='blue', width=1, dash='dash')
        ),
        go.Scatter(
            x=combined_data.index,
            y=[stop_loss_volatility_line] * len(combined_data),
            name=f'Stop(VOLATILITY): {stop_loss_volatility.item():.2f}%',
            line=dict(color='green', width=1, dash='dash')
        )
    ])

    # Добавление подписей на график
    model_name = next((model['label'] for model in available_models if model['value'] == selected_model), "Unknown Model")
    fig.add_annotation(
        x=0.02, y=0.98,  # Позиция в относительных координатах (0-1)
        xref='paper', yref='paper',
        text=f"Ticker: {ticker_input}<br>Model: {model_name}<br>Last Price: {last_close:.4f}<br>Time: {last_date}<br>Timeframe: {interval_input}",
        showarrow=False,
        font=dict(size=10, color='white'),
        bgcolor='black',
        bordercolor='white',
        borderwidth=1
    )

    # Настройка макета графика
    fig.update_layout(
        template='plotly_dark',
        dragmode='pan',
        hovermode='x unified',
        xaxis=dict(rangeslider=dict(visible=False)),
        yaxis=dict(fixedrange=False),
        legend=dict(
            orientation='h',
            x=0.5,
            y=-0.2,
            xanchor='center',
            yanchor='top'
        )
    )

    # Данные для таблицы рисков
    average_risk = np.std(future_predictions)
    risk_data = [
        {'metric': 'Model', 'value': model_name},  # Добавляем информацию о модели
        {'metric': 'Av.RT (std)', 'value': f"{average_risk:.4f}"},
        {'metric': 'VAR 95% (All)', 'value': f"{var_95_all.item():.4f}"},
        {'metric': 'VAR 95% (Last)', 'value': f"{var_95_last:.4f}"},
        {'metric': 'ATR (Last)', 'value': f"{atr_last:.4f}"},
        {'metric': 'VOLATIL (All)', 'value': f"{volatility_all.item():.4f}"},
        {'metric': 'Stop (VAR)', 'value': f"{stop_loss_var.item():.4f}%"},
        {'metric': 'Stop (ATR)', 'value': f"{stop_loss_atr.item():.4f}%"},
        {'metric': 'Stop (VOLATI)', 'value': f"{stop_loss_volatility.item():.4f}%"},
        {'metric': 'MAE', 'value': f"{mae:.4f}"},
        {'metric': 'MSE', 'value': f"{mse:.4f}"},
        {'metric': 'RMSE', 'value': f"{rmse:.4f}"},
        {'metric': 'MAPE', 'value': f"{mape:.4f}%"},
        {'metric': 'Relat MAE', 'value': f"{relative_mae:.4f}"}
    ]

    return fig, risk_data

# Запуск Dash-приложения
if __name__ == '__main__':
    app.run_server(debug=True, port=8051, host='0.0.0.0')
