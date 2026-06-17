# app/widgets/image_label.py
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QSize

class ScaledImageLabelHorizontal(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_pixmap = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)   # центрирование
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,      # разрешаем растягиваться
            QtWidgets.QSizePolicy.Policy.Expanding
        )
        self.setMinimumSize(10, 10)                      # чтобы не схлопывался
        self.setScaledContents(False)                    # отключаем встроенное масштабирование

    def setPixmap(self, pixmap: QPixmap):
        """Сохраняем оригинальный pixmap и пересчитываем масштаб."""
        self.original_pixmap = pixmap
        self.scale_image()

    def resizeEvent(self, event):
        """При изменении размера виджета пересчитываем масштаб."""
        self.scale_image()
        super().resizeEvent(event)

    def scale_image(self):
        """Масштабирует оригинальное изображение под текущие размеры виджета."""
        if self.original_pixmap is None or self.original_pixmap.isNull():
            return

        widget_width = self.width()
        widget_height = self.height()
        if widget_width <= 0 or widget_height <= 0:
            return

        # Вычисляем размер, который впишется в виджет с сохранением пропорций
        scaled_size = self.original_pixmap.size().scaled(
            widget_width,
            widget_height,
            Qt.AspectRatioMode.KeepAspectRatio
        )
        scaled_pixmap = self.original_pixmap.scaled(
            scaled_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        # Устанавливаем масштабированное изображение (родительский QLabel отцентрирует его автоматически)
        super().setPixmap(scaled_pixmap)