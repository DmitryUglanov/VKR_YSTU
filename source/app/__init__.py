import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QTableWidgetItem, QHeaderView
from app.design import Ui_MainWindow
from app.engine import Engine


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.engine = Engine([f"app/sources/train_FD00{i}.txt" for i in range(1, 5)])
        self.dataset_index = 0
        self.setup_info(self.dataset_index)
        self.setup_main_info()
        self.ui.next_button.clicked.connect(self.toggle_next_dataset_index)
        self.ui.previous_button.clicked.connect(self.toggle_previous_dataset_index)
        self.ui.previous_button.setEnabled(False)

    def toggle_next_dataset_index(self):

        if self.dataset_index + 1 == 3:
            self.ui.next_button.setEnabled(False)
        if self.dataset_index + 1 > 0:
            self.ui.previous_button.setEnabled(True)
        self.dataset_index += 1
        self.setup_info(self.dataset_index)


    def toggle_previous_dataset_index(self):
        if self.dataset_index - 1 == 0:
            self.ui.previous_button.setEnabled(False)
        if self.dataset_index - 1 < 3:
            self.ui.next_button.setEnabled(True)
        self.dataset_index -= 1
        self.setup_info(self.dataset_index)

    def setup_info(self, index):
        # Имя
        self.ui.dataset_name.setText(f"Датасет '{self.engine.get_dataset_name(index)}'")
        # Все графики и тексты
        self.ui.life_time_gistogram.setPixmap(self.engine.get_life_time_gistogram(self.engine.get_dataset(index)))
        self.ui.ecdf_func.setPixmap(self.engine.get_ECDF_func(self.engine.get_dataset(index)))
        self.ui.veybula_linear_func.setPixmap(self.engine.get_veybula_linear_func(self.engine.get_dataset(index)))
        self.ui.linear_normal_dispersion_func.setPixmap(self.engine.get_linear_normal_dispersion_func(self.engine.get_dataset(index)))

        self.ui.linear_lognormal_dispersion_func.setPixmap(self.engine.get_linear_lognormal_dispersion_func(self.engine.get_dataset(index)))
        self.ui.gamma_funcs.setPixmap(
            self.engine.get_gamma_funcs(self.engine.get_dataset(index)))
    def setup_main_info(self):
        modeling_text_out, modeling_func, table = self.engine.get_text_and_func_modeling_output()
        self.ui.modeling_func.setPixmap(modeling_func)
        self.ui.text_output.setPlainText(modeling_text_out)

        self.ui.table.setRowCount(table.shape[0])
        self.ui.table.setColumnCount(table.shape[1])
        self.ui.table.setHorizontalHeaderLabels(table.columns)
        header = self.ui.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for row in range(table.shape[0]):
            for col in range(table.shape[1]):
                item = QTableWidgetItem(str(table.iat[row, col]))

                self.ui.table.setItem(row, col, item)