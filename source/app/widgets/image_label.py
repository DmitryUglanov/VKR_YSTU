from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

class ScaledImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_pixmap = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding   # по вертикали тоже Expanding, но fixedHeight перекроет
        )
        self.setMinimumSize(10, 10)

    def setPixmap(self, pixmap):
        self.original_pixmap = pixmap
        self.scale_image()

    def resizeEvent(self, event):
        self.scale_image()
        super().resizeEvent(event)

    def scale_image(self):
        if self.original_pixmap is None or self.original_pixmap.isNull():
            return
        if self.width() <= 0:
            return

        # Вычисляем масштабированную высоту исходя из текущей ширины
        scaled_height = int(self.original_pixmap.height() * self.width() / self.original_pixmap.width())
        # Ограничиваем минимальной высотой (не даём схлопнуться)
        scaled_height = max(scaled_height, 10)

        # Устанавливаем фиксированную высоту метки — layout будет её учитывать
        self.setFixedHeight(scaled_height)

        # Теперь масштабируем сам pixmap до новой ширины и высоты
        scaled_pixmap = self.original_pixmap.scaled(
            self.width(), scaled_height,
            Qt.AspectRatioMode.IgnoreAspectRatio,  # уже точный размер
            Qt.TransformationMode.SmoothTransformation
        )
        super().setPixmap(scaled_pixmap)