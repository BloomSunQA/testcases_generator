"""
Юнит-тесты для приложения app.py
Тестируем логику генерации тест-кейсов и оценки качества с использованием pytest и mocking.

Эти тесты проверяют:
- Успешное выполнение функций генерации и оценки
- Загрузку переменных окружения
- Обработку ошибок API

Все тесты используют mocking, чтобы избежать реальных запросов к API.
"""

import pytest
from unittest.mock import patch, MagicMock
import os
import sys

# Добавляем корневую директорию проекта в sys.path для импорта app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app import app, call_mentorpiece_api


class TestCallMentorpieceApi:
    """
    Тесты для функции call_mentorpiece_api
    """

    def test_api_key_not_found(self, monkeypatch):
        """
        Тест: Проверка, что функция возвращает ошибку, если API ключ не найден в переменных окружения.

        Это важно для тестирования логики загрузки конфигурации.
        """
        # Mock: устанавливаем API_KEY в None
        monkeypatch.setattr(sys.modules['app'], 'API_KEY', None)

        # Вызываем функцию
        result = call_mentorpiece_api('test_model', 'test_prompt')

        # Проверяем, что вернулась ошибка
        assert 'error' in result
        assert 'API ключ не найден' in result['error']

    @patch('app.api_session.post')
    def test_successful_api_call(self, mock_post, monkeypatch):
        """
        Тест: Успешный вызов API (200 OK).

        Mock: имитируем успешный ответ от API.
        """
        # Mock: устанавливаем API_KEY
        monkeypatch.setattr(sys.modules['app'], 'API_KEY', 'fake_api_key')

        # Mock: успешный ответ от API
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None  # Не вызывает исключение
        mock_response.json.return_value = {'response': 'Mocked AI response'}
        mock_post.return_value = mock_response

        # Вызываем функцию
        result = call_mentorpiece_api('test_model', 'test_prompt')

        # Проверяем результат
        assert result == {'response': 'Mocked AI response'}

        # Проверяем, что post был вызван с правильными параметрами
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs['json'] == {'model_name': 'test_model', 'prompt': 'test_prompt'}
        assert kwargs['headers']['Authorization'] == 'Bearer fake_api_key'

    @patch('app.api_session.post')
    def test_api_timeout_error(self, mock_post, monkeypatch):
        """
        Тест: Обработка таймаута API.

        Mock: requests.post выбрасывает Timeout исключение.
        """
        monkeypatch.setattr(sys.modules['app'], 'API_KEY', 'fake_api_key')

        # Mock: выбрасываем Timeout
        from requests.exceptions import Timeout
        mock_post.side_effect = Timeout()

        result = call_mentorpiece_api('test_model', 'test_prompt')

        assert 'error' in result
        assert 'Таймаут API' in result['error']

    @patch('app.api_session.post')
    def test_api_connection_error(self, mock_post, monkeypatch):
        """
        Тест: Обработка ошибки подключения.

        Mock: requests.post выбрасывает ConnectionError.
        """
        monkeypatch.setattr(sys.modules['app'], 'API_KEY', 'fake_api_key')

        from requests.exceptions import ConnectionError
        mock_post.side_effect = ConnectionError()

        result = call_mentorpiece_api('test_model', 'test_prompt')

        assert 'error' in result
        assert 'Ошибка подключения' in result['error']

    @patch('app.api_session.post')
    def test_api_http_error_401(self, mock_post, monkeypatch):
        """
        Тест: Обработка HTTP ошибки 401 (неверный API ключ).

        Mock: raise_for_status выбрасывает HTTPError с status 401.
        """
        monkeypatch.setattr(sys.modules['app'], 'API_KEY', 'fake_api_key')

        from requests.exceptions import HTTPError
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {}
        mock_http_error = HTTPError()
        mock_http_error.response = mock_response
        mock_post.return_value.raise_for_status.side_effect = mock_http_error

        result = call_mentorpiece_api('test_model', 'test_prompt')

        assert 'error' in result
        assert 'Неверный API ключ' in result['error']


class TestUploadRoute:
    """
    Тесты для маршрута /upload (генерация тест-кейсов)
    """

    def setup_method(self):
        """
        Настройка перед каждым тестом: создаем тестовый клиент Flask.
        """
        self.app = app.test_client()
        self.app.testing = True

    @patch('app.call_mentorpiece_api')
    def test_upload_success(self, mock_call_api):
        """
        Positive Test: Успешная генерация тест-кейсов.

        Mock: call_mentorpiece_api возвращает успешный ответ с фиктивными тест-кейсами.
        """
        # Mock: успешный ответ от API с фиктивными тест-кейсами (Worker модель)
        mock_call_api.return_value = {'response': 'Фиктивные тест-кейсы для веб-формы'}

        # Данные для POST запроса
        data = {'description': 'Test form description'}

        # Выполняем POST запрос
        response = self.app.post('/upload', json=data)

        # Проверяем статус код
        assert response.status_code == 200

        # Проверяем JSON ответ
        json_data = response.get_json()
        assert json_data['success'] is True
        assert json_data['test_cases'] == 'Фиктивные тест-кейсы для веб-формы'
        assert json_data['stage'] == 'generation'

        # Проверяем, что call_mentorpiece_api был вызван с правильными аргументами
        mock_call_api.assert_called_once()
        args, kwargs = mock_call_api.call_args
        assert kwargs['model_name'] == 'gpt-5.1-2025-11-13'  # Модель для генерации (Worker)
        assert 'Test form description' in kwargs['prompt']  # Промпт содержит описание

    def test_upload_missing_description(self):
        """
        Тест: Отсутствие описания формы в запросе.
        """
        # POST без данных
        response = self.app.post('/upload', json={})

        assert response.status_code == 400
        json_data = response.get_json()
        assert 'error' in json_data
        assert 'Описание формы не предоставлено' in json_data['error']

    def test_upload_empty_description(self):
        """
        Тест: Пустое описание формы.
        """
        data = {'description': '   '}  # Только пробелы

        response = self.app.post('/upload', json=data)

        assert response.status_code == 400
        json_data = response.get_json()
        assert 'error' in json_data
        assert 'Описание формы пустое' in json_data['error']

    @patch('app.call_mentorpiece_api')
    def test_upload_api_error(self, mock_call_api):
        """
        Тест: Ошибка от API при генерации тест-кейсов.

        Mock: call_mentorpiece_api возвращает ошибку.
        """
        mock_call_api.return_value = {'error': 'API limit exceeded'}

        data = {'description': 'Test form description'}

        response = self.app.post('/upload', json=data)

        assert response.status_code == 500
        json_data = response.get_json()
        assert 'error' in json_data
        assert 'Ошибка при генерации тест-кейсов' in json_data['error']

    @patch('app.call_mentorpiece_api')
    def test_upload_empty_response(self, mock_call_api):
        """
        Тест: API вернул пустой ответ.
        """
        mock_call_api.return_value = {'response': ''}

        data = {'description': 'Test form description'}

        response = self.app.post('/upload', json=data)

        assert response.status_code == 500
        json_data = response.get_json()
        assert 'error' in json_data
        assert 'API вернул пустой ответ' in json_data['error']


class TestEvaluateRoute:
    """
    Тесты для маршрута /evaluate (оценка тест-кейсов)
    """

    def setup_method(self):
        """
        Настройка перед каждым тестом.
        """
        self.app = app.test_client()
        self.app.testing = True

    @patch('app.call_mentorpiece_api')
    def test_evaluate_success(self, mock_call_api):
        """
        Positive Test: Успешная оценка тест-кейсов.

        Mock: call_mentorpiece_api возвращает фиктивную оценку (Judge модель).
        """
        mock_call_api.return_value = {'response': 'Оценка: 9/10. Отличные тест-кейсы.'}

        data = {'test_cases': 'Some test cases text'}

        response = self.app.post('/evaluate', json=data)

        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data['success'] is True
        assert json_data['evaluation'] == 'Оценка: 9/10. Отличные тест-кейсы.'
        assert json_data['stage'] == 'evaluation'

        # Проверяем вызов API с моделью Judge
        mock_call_api.assert_called_once()
        args, kwargs = mock_call_api.call_args
        assert kwargs['model_name'] == 'claude-sonnet-4-5-20250929'  # Модель Judge

    def test_evaluate_missing_test_cases(self):
        """
        Тест: Отсутствие тест-кейсов в запросе.
        """
        response = self.app.post('/evaluate', json={})

        assert response.status_code == 400
        json_data = response.get_json()
        assert 'error' in json_data
        assert 'Тест-кейсы не переданы' in json_data['error']

    def test_evaluate_empty_test_cases(self):
        """
        Тест: Пустые тест-кейсы.
        """
        data = {'test_cases': ''}

        response = self.app.post('/evaluate', json=data)

        assert response.status_code == 400
        json_data = response.get_json()
        assert 'error' in json_data
        assert 'Тест-кейсы не могут быть пустыми' in json_data['error']

    @patch('app.call_mentorpiece_api')
    def test_evaluate_fallback_model(self, mock_call_api):
        """
        Тест: Fallback на вторую модель, если первая вернула ошибку.

        Mock: первая модель возвращает ошибку, вторая - успех.
        """
        # Первая модель возвращает ошибку
        mock_call_api.side_effect = [
            {'error': 'cache_control error'},  # Ошибка первой модели
            {'response': 'Fallback evaluation: 8/10'}  # Успех второй модели
        ]

        data = {'test_cases': 'Some test cases'}

        response = self.app.post('/evaluate', json=data)

        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data['success'] is True
        assert json_data['evaluation'] == 'Fallback evaluation: 8/10'

        # Проверяем, что API был вызван два раза
        assert mock_call_api.call_count == 2
        # Первый вызов с claude, второй с gemma
        calls = mock_call_api.call_args_list
        assert calls[0][1]['model_name'] == 'claude-sonnet-4-5-20250929'
        assert calls[1][1]['model_name'] == 'google/gemma-3-27b-it'

    @patch('app.call_mentorpiece_api')
    def test_evaluate_api_error_no_fallback(self, mock_call_api):
        """
        Тест: Ошибка API без возможности fallback.
        """
        mock_call_api.return_value = {'error': 'Some other error'}

        data = {'test_cases': 'Some test cases'}

        response = self.app.post('/evaluate', json=data)

        assert response.status_code == 500
        json_data = response.get_json()
        assert 'error' in json_data
        assert 'Ошибка при оценке тест-кейсов' in json_data['error']


# Запуск тестов (если файл запущен напрямую)
if __name__ == '__main__':
    pytest.main([__file__])