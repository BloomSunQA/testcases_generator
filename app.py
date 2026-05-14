"""
Test-cases Generator - Web Application
Генерирует тест-кейсы на основе описания форм с помощью LLM API
"""

import os
import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

# Загружаем переменные из .env файла
load_dotenv()

app = Flask(__name__)

# Конфигурация приложения
API_KEY = os.getenv('MENTORPIECE_API_KEY')
API_BASE_URL = 'https://api.mentorpiece.org/v1/process-ai-request'
API_TIMEOUT = 120  # секунды

# Используем Session для переиспользования соединений
api_session = requests.Session()


def call_mentorpiece_api(model_name, prompt):
    """
    Отправляет запрос к API MentorPiece
    
    Args:
        model_name (str): Название модели (например, Qwen/Qwen3-VL-30B-A3B-Instruct)
        prompt (str): Текст промпта для модели
        
    Returns:
        dict: Ответ от API с ключом 'response' или ошибка
    """
    if not API_KEY:
        return {
            'error': 'API ключ не найден. Установите MENTORPIECE_API_KEY в переменных окружения.'
        }
    
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'model_name': model_name,
        'prompt': prompt
    }
    
    try:
        response = api_session.post(
            API_BASE_URL,
            json=payload,
            headers=headers,
            timeout=API_TIMEOUT
        )
        response.raise_for_status()  # Вызывает исключение при HTTP ошибке
        
        data = response.json()
        return data
        
    except requests.exceptions.Timeout:
        return {'error': f'Таймаут API (превышен лимит {API_TIMEOUT} сек)'}
    except requests.exceptions.ConnectionError:
        return {'error': 'Ошибка подключения к API. Проверьте интернет-соединение.'}
    except requests.exceptions.HTTPError as e:
        try:
            error_json = e.response.json()
            if isinstance(error_json, dict):
                if 'message' in error_json:
                    return {'error': f'HTTP ошибка {e.response.status_code}: {error_json.get("message")}' }
                if 'error' in error_json and isinstance(error_json['error'], dict):
                    return {'error': f'HTTP ошибка {e.response.status_code}: {error_json["error"].get("message", str(error_json["error"]))}'}
        except ValueError:
            pass

        if e.response.status_code == 401:
            return {'error': 'Неверный API ключ. Проверьте значение MENTORPIECE_API_KEY'}
        elif e.response.status_code == 429:
            return {'error': 'Лимит запросов к API превышен. Попробуйте позже.'}
        else:
            return {'error': f'HTTP ошибка {e.response.status_code}: {e.response.text}'}
    except requests.exceptions.RequestException as e:
        return {'error': f'Ошибка запроса к API: {str(e)}'}
    except ValueError:
        return {'error': 'Ошибка парсинга ответа API (не JSON)'}


@app.route('/')
def index():
    """Главная страница с формой описания веб-формы"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """
    Обрабатывает описание формы и генерирует тест-кейсы (первая модель)

    Returns:
        JSON с результатами или ошибкой
    """

    # Получаем данные из JSON запроса
    data = request.get_json()
    if not data or 'description' not in data:
        return jsonify({'error': 'Описание формы не предоставлено'}), 400

    description = data['description'].strip()
    if not description:
        return jsonify({'error': 'Описание формы пустое'}), 400

    try:
        # ============= ЭТАП 1: Генерируем тест-кейсы на основе описания =============
        test_cases_prompt = f"""На основе описания формы создай подробные тест-кейсы для тестирования этой формы.

Описание формы:
{description}

Создай полный набор тест-кейсов, включая:
- Позитивные сценарии (корректный ввод всех полей)
- Негативные сценарии (ошибки валидации, пустые поля, некорректные данные)
- Граничные значения для каждого поля
- Проверку всех кнопок и взаимодействий

Структурируй ответ в виде списка тест-кейсов с:
1. Названием тест-кейса
2. Предусловиями
3. Шагами выполнения
4. Ожидаемым результатом

Сделай тест-кейсы максимально подробными и реалистичными."""

        test_cases_response = call_mentorpiece_api(
            model_name='gpt-5.1-2025-11-13',
            prompt=test_cases_prompt
        )

        # Проверяем ошибки
        if 'error' in test_cases_response:
            return jsonify({
                'error': f'Ошибка при генерации тест-кейсов: {test_cases_response["error"]}'
            }), 500

        # Получаем сгенерированные тест-кейсы
        test_cases_text = test_cases_response.get('response', '')

        if not test_cases_text:
            return jsonify({
                'error': 'API вернул пустой ответ при генерации тест-кейсов'
            }), 500

        # Успешно! Возвращаем результаты первого этапа
        return jsonify({
            'success': True,
            'test_cases': test_cases_text,
            'stage': 'generation'  # Указываем, что это первый этап
        })

    except Exception as e:
        return jsonify({
            'error': f'Неожиданная ошибка: {str(e)}'
        }), 500


@app.route('/evaluate', methods=['POST'])
def evaluate():
    """
    Оценивает качество тест-кейсов (вторая модель - LLM-as-a-Judge)
    
    Returns:
        JSON с оценкой или ошибкой
    """
    
    data = request.get_json()
    
    # Проверяем наличие тест-кейсов
    if not data or 'test_cases' not in data:
        return jsonify({'error': 'Тест-кейсы не переданы'}), 400
    
    test_cases_text = data.get('test_cases', '')
    
    if not test_cases_text:
        return jsonify({'error': 'Тест-кейсы не могут быть пустыми'}), 400
    
    try:
        # ============= ЭТАП 2: Оцениваем качество (Claude - LLM-as-a-Judge) =============
        evaluation_prompt = f"""Проанализируй качество следующих тест-кейсов для веб-формы и дай оценку:

ТЕСТ-КЕЙСЫ ДЛЯ ОЦЕНКИ:
{test_cases_text}

ПРОАНАЛИЗИРУЙ:
1. Полнота покрытия - охвачены ли все поля формы?
2. Качество позитивных сценариев
3. Качество негативных сценариев (граничные значения, ошибки)
4. Четкость описания шагов и ожидаемых результатов
5. Возможность автоматизации тестов

ДАЙ ОЦЕНКУ ПО ШКАЛЕ ОТ 1 ДО 10 и кратко обоснуй."""

        evaluation_response = call_mentorpiece_api(
            model_name='claude-sonnet-4-5-20250929',
            prompt=evaluation_prompt
        )

        # Если Claude возвращает ошибку, пробуем fallback на google/gemma-3-27b-it
        if 'error' in evaluation_response:
            error_text = evaluation_response['error']
            if 'cache_control' in error_text or 'invalid_request_error' in error_text:
                evaluation_response = call_mentorpiece_api(
                    model_name='google/gemma-3-27b-it',
                    prompt=evaluation_prompt
                )
                if 'error' in evaluation_response:
                    return jsonify({
                        'error': f'Ошибка при оценке тест-кейсов: {evaluation_response["error"]}'
                    }), 500
            else:
                return jsonify({
                    'error': f'Ошибка при оценке тест-кейсов: {error_text}'
                }), 500

        # Получаем оценку
        evaluation_text = evaluation_response.get('response', '')
        
        if not evaluation_text:
            return jsonify({
                'error': 'API вернул пустой ответ при оценке тест-кейсов'
            }), 500
        
        # Успешно! Возвращаем оценку
        return jsonify({
            'success': True,
            'evaluation': evaluation_text,
            'stage': 'evaluation'  # Указываем, что это второй этап
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Неожиданная ошибка: {str(e)}'
        }), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Обработка ошибки слишком большого файла"""
    return jsonify({
        'error': 'Файл слишком большой. Максимальный размер: 16 MB'
    }), 413


@app.errorhandler(500)
def internal_error(error):
    """Обработка внутренних ошибок сервера"""
    return jsonify({
        'error': 'Внутренняя ошибка сервера'
    }), 500


if __name__ == '__main__':
    # Проверяем наличие API ключа при запуске
    if not API_KEY:
        print("⚠️  ВНИМАНИЕ: Переменная окружения MENTORPIECE_API_KEY не установлена!")
        print("Установите её перед запуском приложения.")
    else:
        print("✓ API ключ загружен")
    
    print("🚀 Запускаем Flask приложение на http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
