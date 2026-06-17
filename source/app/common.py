import sys
import os
from pathlib import Path

def resource_path(relative_path):
    """Получить абсолютный путь к ресурсу, работает как в разработке, так и в собранном exe."""
    try:
        # PyInstaller создает временную папку и хранит путь к ней в _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        # Если запущено как обычный скрипт, используем путь к текущему файлу
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
