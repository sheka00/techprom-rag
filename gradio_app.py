import os
import logging
import gradio as gr
import httpx
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# URL API бэкенда (используем имя сервиса из docker-compose)
API_URL = os.getenv("API_URL", "http://api:8000")


async def get_docs_list():
    """Получает список документов через API"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{API_URL}/documents")
            if resp.status_code == 200:
                docs = resp.json().get("documents", [])
                if not docs:
                    return [["База пуста", 0]]
                # Сортируем для красоты
                sorted_docs = sorted(
                    [[d["name"], d["chunks"]] for d in docs], key=lambda x: x[0]
                )
                return sorted_docs
            else:
                logger.error(f"❌ Ошибка API при получении списка: {resp.text}")
                return [["Ошибка API", 0]]
    except Exception as e:
        logger.error(f"❌ Ошибка подключения к API: {e}")
        return [["Ошибка подключения", 0]]


async def process_upload(files, progress=gr.Progress()):
    """Загружает файлы в API по одному"""
    if not files:
        return "⚠️ Пожалуйста, сначала выберите файлы для загрузки."

    progress(0, desc="🚀 Начало загрузки на бэкенд...")
    success_count = 0

    async with httpx.AsyncClient(timeout=600.0) as client:
        for i, file in enumerate(files):
            file_name = Path(file.name).name
            progress((i) / len(files), desc=f"📄 [Загрузка] {file_name}...")

            try:
                with open(file.name, "rb") as f:
                    files_payload = {"file": (file_name, f)}
                    resp = await client.post(f"{API_URL}/upload", files=files_payload)

                    if resp.status_code == 200:
                        success_count += 1
                        logger.info(f"✅ Успешно загружен {file_name}")
                    else:
                        logger.error(f"❌ Ошибка загрузки {file_name}: {resp.text}")
            except Exception as e:
                logger.error(f"❌ Критическая ошибка при загрузке {file_name}: {e}")

    progress(1.0, desc="🌟 Загрузка завершена!")
    return f"✅ Проиндексировано: {success_count} из {len(files)} файл(ов)."


async def ask_question(question):
    """Задает вопрос через API"""
    if not question:
        return "⚠️ Пожалуйста, введите вопрос.", "", "", ""

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{API_URL}/query", json={"question": question})

            if resp.status_code == 200:
                result = resp.json()
                answer = result.get("answer", "Нет ответа")
                source = result.get("source", "Не указано")
                quote = result.get("quote", "Нет цитаты")

                # Формируем техническую информацию
                chunks_text = ""
                for i, chunk in enumerate(result.get("sources", [])):
                    text = chunk.get("content", "").strip()
                    chunks_text += f"---\n[Фрагмент {i + 1}]:\n{text}\n\n"

                return answer, source, quote, chunks_text
            else:
                logger.error(f"❌ Ошибка API при поиске: {resp.text}")
                return f"❌ Ошибка бэкенда: {resp.text}", "", "", ""
    except Exception as e:
        logger.error(f"❌ Ошибка при поиске через API: {e}")
        return f"❌ Ошибка подключения: {str(e)}", "", "", ""


async def handle_delete(doc_name):
    """Удаляет документ через API"""
    if not doc_name or doc_name == "База пуста":
        return gr.update(visible=False), "Ошибка: документ не выбран"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Кодируем имя файла для URL
            import urllib.parse

            encoded_name = urllib.parse.quote(doc_name)
            resp = await client.delete(f"{API_URL}/documents/{encoded_name}")

            if resp.status_code == 200:
                return gr.update(
                    visible=False
                ), f"✅ Документ '{doc_name}' успешно удален."
            else:
                return gr.update(
                    visible=True
                ), f"❌ Ошибка API при удалении: {resp.text}"
    except Exception as e:
        logger.error(f"❌ Ошибка удаления через API: {e}")
        return gr.update(visible=True), f"❌ Ошибка подключения: {str(e)}"


# Настройки CSS
custom_css = """
footer {display: none !important;}
.gradio-container footer {display: none !important;}
.gradio-container {padding-bottom: 50px !important;}
"""

with gr.Blocks(
    title="TechProm RAG System", theme=gr.themes.Soft(), css=custom_css
) as demo:
    gr.Markdown("# 🤖 TechProm RAG System (Frontend)")
    gr.Markdown("Интерфейс работает через централизованный API бэкенд.")

    with gr.Tab("📁 1. Управление документами"):
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="📤 Загрузка новых файлов (.pdf, .docx)",
                    file_count="multiple",
                    file_types=[".pdf", ".docx"],
                )
                upload_btn = gr.Button("🚀 Загрузить в базу", variant="primary")
                upload_output = gr.Textbox(label="Статус", interactive=False, lines=3)

            with gr.Column(scale=1):
                gr.Markdown("### 📜 Индексированные файлы")
                docs_table = gr.Dataframe(
                    headers=["Название документа", "Чанков"],
                    datatype=["str", "number"],
                    label="Нажмите на строку для выбора файла",
                    interactive=False,
                )
                refresh_btn = gr.Button("🔄 Обновить список")

                selected_doc = gr.State("")
                with gr.Accordion(
                    "🗑️ Удаление выбранного документа", open=True, visible=False
                ) as delete_row:
                    delete_name_label = gr.Markdown("Файл не выбран")
                    delete_btn = gr.Button("❌ Удалить из базы", variant="stop")

            def on_select(evt: gr.SelectData, table_data):
                row_idx = evt.index[0]
                doc_name = table_data.iloc[row_idx, 0]
                if doc_name == "База пуста":
                    return gr.update(visible=False), "", ""
                return (
                    gr.update(visible=True),
                    f"⚠️ Вы выбрали: `{doc_name}`",
                    doc_name,
                )

            docs_table.select(
                fn=on_select,
                inputs=[docs_table],
                outputs=[delete_row, delete_name_label, selected_doc],
                show_progress="hidden",
            )

            delete_btn.click(
                fn=handle_delete,
                inputs=[selected_doc],
                outputs=[delete_row, upload_output],
            ).then(fn=get_docs_list, outputs=[docs_table])

        upload_btn.click(
            fn=process_upload, inputs=[file_input], outputs=[upload_output]
        ).then(
            fn=lambda: gr.update(value=None), inputs=None, outputs=[file_input]
        ).then(fn=get_docs_list, outputs=[docs_table])

        refresh_btn.click(fn=get_docs_list, outputs=docs_table)

    with gr.Tab("❓ 2. Вопрос-Ответ"):
        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="Ваш вопрос",
                    placeholder="Например: Какая максимальная температура эксплуатации изделия?",
                    lines=3,
                )
                ask_btn = gr.Button("🔍 Спросить систему", variant="primary")

            with gr.Column(scale=3):
                gr.Markdown("### 📝 Ответ нейросети")
                answer_output = gr.Markdown()

                with gr.Row():
                    source_output = gr.Textbox(label="Источник")
                    quote_output = gr.Textbox(label="Цитата")

        with gr.Accordion("🔍 Найденные фрагменты (Retriever)", open=False):
            retrieved_chunks = gr.Textbox(label="Контент из Qdrant", lines=15)

        ask_btn.click(
            fn=ask_question,
            inputs=[question_input],
            outputs=[answer_output, source_output, quote_output, retrieved_chunks],
        )

    # Автозагрузка списка при старте
    demo.load(fn=get_docs_list, outputs=docs_table)

if __name__ == "__main__":
    auth_user = os.getenv("GRADIO_USER", "admin")
    auth_password = os.getenv("GRADIO_PASSWORD", "admin")

    logger.info(f"🚀 Gradio Frontend запущен. Подключение к API: {API_URL}")

    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        auth=(auth_user, auth_password),
    )
