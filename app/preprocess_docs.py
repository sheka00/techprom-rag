import logging
import argparse
from pathlib import Path
from app.docx_preprocessor import DocxPreprocessor

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DOCX-Preprocessor")


def main():
    parser = argparse.ArgumentParser(
        description="Предобработка DOCX файлов для улучшения работы Docling"
    )
    parser.add_argument(
        "--input", default="data/ТИ", help="Путь к папке с исходными файлами"
    )
    parser.add_argument(
        "--output", default="data/preprocessed_ТИ", help="Путь к папке для сохранения"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        logger.error(f"Входная папка не найдена: {input_dir}")
        return

    logger.info("=" * 60)
    logger.info("🚀 ЗАПУСК ПРЕДОБРАБОТКИ ДОКУМЕНТОВ")
    logger.info(f"   Вход:  {input_dir}")
    logger.info(f"   Выход: {output_dir}")
    logger.info("=" * 60)

    preprocessor = DocxPreprocessor(output_dir=str(output_dir))
    processed_files = preprocessor.process_directory(str(input_dir))

    logger.info("=" * 60)
    logger.info("✅ ПРЕДОБРАБОТКА ЗАВЕРШЕНА")
    logger.info(f"   Обработано файлов: {len(processed_files)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
