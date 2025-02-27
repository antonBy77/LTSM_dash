# Используем базовый образ Python
FROM python:3.10.12

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта
COPY . /app
RUN pip install --upgrade pip
# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Указываем порт, который будет использовать приложение
EXPOSE 8051

# Запускаем приложение
CMD ["python", "app.py"]
