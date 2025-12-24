import streamlit as st
import requests
import tempfile
import os
from io import BytesIO
from typing import List, Any
import json
import time

# Конфигурация страницы
st.set_page_config(
    page_title="ScienceRAG Interface",
    page_icon="🔬",
    layout="wide"
)

# Конфигурация API
API_BASE_URL = "http://localhost:8025"  # Измените на ваш URL
DEFAULT_MODEL = ""  # Измените на вашу модель по умолчанию

# Инициализация состояния сессии
def init_session_state():
    if 'dataset_id' not in st.session_state:
        st.session_state.dataset_id = None
    if 'uploaded' not in st.session_state:
        st.session_state.uploaded = False
    if 'parsed' not in st.session_state:
        st.session_state.parsed = False
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

init_session_state()

# Проверка здоровья API
def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# Функции для вызова API
def upload_dataset_to_api(uploaded_files: List[Any], name: str = "default_dataset"):
    """Загрузка файлов на API"""
    files_data = []
    for uploaded_file in uploaded_files:
        files_data.append(('files', (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)))
    
    data = {
        'name': name,
        'chunk_method': 'naive',
        'embedding_model': 'mistral-embed@Mistral'
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/upload-dataset/",
            files=files_data,
            data=data,
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Ошибка загрузки: {response.text}")
            return None
    except Exception as e:
        st.error(f"Ошибка соединения: {str(e)}")
        return None

def parse_documents_api(dataset_id: str):
    """Парсинг документов через API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/parse-documents/",
            params={'dataset_id': dataset_id},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Ошибка парсинга: {response.text}")
            return None
    except Exception as e:
        st.error(f"Ошибка соединения: {str(e)}")
        return None

def generate_answer_api(query: str, dataset_id: str, model: str = DEFAULT_MODEL):
    """Генерация ответа через API"""
    payload = {
        "query": query,
        "dataset_ids": [dataset_id],
        "limit": 10,
        "similarity_threshold": 0.1,
        "model": model
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate/",
            json=payload,
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Ошибка генерации: {response.text}")
            return None
    except Exception as e:
        st.error(f"Ошибка соединения: {str(e)}")
        return None

# Интерфейс
st.title("🔬 ScienceRAG Document Assistant")
st.markdown("Загружайте научные документы и задавайте вопросы на их основе")

# Проверка доступности API
if not check_api_health():
    st.error("⚠️ API сервер недоступен. Убедитесь, что сервер запущен на порту 8025.")
    st.stop()

# Боковая панель с настройками
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Выбор модели
    model = st.selectbox(
        "Модель для генерации",
        ["mistral-small-latest", "model"],
        index=0
    )
    
    
    embedding_model = st.selectbox(
        "Модель эмбеддингов",
        ["mistral-embed", "model"],
        index=0
    )
    
    # Информация о статусе
    st.header("📊 Статус")
    if st.session_state.uploaded:
        st.success("✅ Файлы загружены")
        if st.session_state.dataset_id:
            st.code(f"Dataset ID: {st.session_state.dataset_id}...")
        
        if st.session_state.parsed:
            st.success("✅ Документы распарсены")
        else:
            st.warning("⏳ Требуется парсинг")
    else:
        st.info("📁 Ожидание загрузки файлов")
    
    # Кнопка сброса
    if st.button("🔄 Сбросить сессию", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        init_session_state()
        st.rerun()

# Основной интерфейс
tab1, tab2 = st.tabs(["📤 Загрузка документов", "💬 Вопрос-ответ"])

with tab1:
    st.header("1. Загрузка документов")
    
    # Поле для имени датасета
    dataset_name = st.text_input(
        "Название датасета",
        value=f"dataset_{int(time.time())}",
        help="Уникальное имя для набора документов"
    )
    
    # Поле для загрузки файлов
    uploaded_files = st.file_uploader(
        "Перетащите файлы сюда или выберите из проводника",
        type=['pdf', 'docx', 'txt', 'md', 'pptx', 'xlsx', 'csv'],
        accept_multiple_files=True,
        help="Поддерживаемые форматы: PDF, Word, Text, Markdown, PowerPoint, Excel, CSV"
    )
    
    # Кнопка загрузки
    if uploaded_files and not st.session_state.uploaded:
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("📤 Загрузить на сервер", type="primary"):
                with st.spinner("Загружаем файлы..."):
                    # Сохраняем файлы
                    st.session_state.uploaded_files = uploaded_files
                    
                    # Вызов API загрузки
                    result = upload_dataset_to_api(uploaded_files, dataset_name)
                    
                    if result and result.get("status") == "success":
                        st.session_state.dataset_id = result.get("dataset_id")
                        st.session_state.uploaded = True
                        st.success("✅ Файлы успешно загружены!")
                        st.rerun()
                    else:
                        st.error("Ошибка при загрузке файлов")
    
    # Кнопка парсинга
    if st.session_state.uploaded and not st.session_state.parsed:
        st.header("2. Парсинг документов")
        
        if st.button("⚙️ Распарсить документы", type="primary"):
            with st.spinner("Парсим документы... Это может занять некоторое время"):
                result = parse_documents_api(st.session_state.dataset_id)
                
                if result and result.get("status") == "parsing_success":
                    st.session_state.parsed = True
                    st.success("✅ Документы успешно распарсены!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("Ошибка при парсинге документов")

with tab2:
    # Только если документы загружены и распарсены
    if not st.session_state.uploaded:
        st.warning("Сначала загрузите документы на вкладке 'Загрузка документов'")
    elif not st.session_state.parsed:
        st.warning("Сначала распарсьте документы на вкладке 'Загрузка документов'")
    else:
        st.header("💬 Вопрос-ответ система")
        st.info(f"Используется датасет: {dataset_name}")
        
        # История чата
        if st.session_state.chat_history:
            st.subheader("История диалога")
            for i, chat in enumerate(st.session_state.chat_history):
                with st.expander(f"Вопрос {i+1}: {chat['query'][:50]}...", expanded=(i==len(st.session_state.chat_history)-1)):
                    st.markdown(f"**Вопрос:** {chat['query']}")
                    st.markdown(f"**Ответ:** {chat['answer']}")
                    
                    if chat.get('sources'):
                        st.markdown("**Источники:**")
                        for j, source in enumerate(chat['sources']):
                            st.markdown(f"{j+1}. `{source}`")
        
        # Поле для нового вопроса
        st.subheader("Новый вопрос")
        query = st.text_area(
            "Введите ваш вопрос:",
            placeholder="Например: Какие основные выводы из исследования? Или: Объясните методологию...",
            height=100,
            key="query_input"
        )
        
        # Кнопка генерации
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🤖 Сгенерировать ответ", type="primary", disabled=not query):
                with st.spinner("Ищем ответ в документах..."):
                    result = generate_answer_api(
                        query=query,
                        dataset_id=st.session_state.dataset_id,
                        model=model
                    )
                    
                    if result:
                        # Сохраняем в историю
                        st.session_state.chat_history.append({
                            'query': query,
                            'answer': result['answer'],
                            'sources': result['sources'],
                            'timestamp': time.time()
                        })
                        
                        st.markdown("### Ответ:")
                        st.markdown(result['answer'])

                        # Отображаем источники
                        if result.get('sources'):
                            st.markdown("### 📚 Источники информации:")
                            for i, source in enumerate(result['sources'], 1):
                                st.write(f"{i}. **{source}**")
        
        # Статистика
        st.divider()

# CSS стили для улучшения интерфейса
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        margin-top: 10px;
    }
    .stSuccess {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
    }
    .stWarning {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stInfo {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
    }
</style>
""", unsafe_allow_html=True)

# Информация для разработчика (можно скрыть)
with st.expander("🛠️ Информация для отладки"):
    st.json({
        "session_state": {
            "uploaded": st.session_state.uploaded,
            "parsed": st.session_state.parsed,
            "dataset_id": st.session_state.dataset_id,
            "uploaded_files_count": len(st.session_state.uploaded_files),
            "chat_history_count": len(st.session_state.chat_history)
        },
        "api_endpoints": {
            "health": f"{API_BASE_URL}/health",
            "upload": f"{API_BASE_URL}/upload-dataset/",
            "parse": f"{API_BASE_URL}/parse-documents/",
            "generate": f"{API_BASE_URL}/generate/"
        }
    })