
# from PySide6.QtWidgets import (
#     QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout,
#     QHBoxLayout, QTableWidget, QTableWidgetItem, QDialog,
#     QDialogButtonBox, QMessageBox, QComboBox
# )
# from PySide6.QtCore import Qt 
# import sys
# from enum import Enum



# class TypeData(Enum):
#     LAMBDA_FROM_F = "График зависимости длины волны от фок.раст"
#     MODEL_TWO_LINSE = "Моделирование для 2-х линз"
#     MODE_THREE_LINSE = "Моделирование для 2-х линз"


# class LensConfigurator(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Конфигурация гармонических линз")
#         self.resize(500, 300)

#         self.lens_count = 0
#         self.data_storage = {}
#         self.type_dates = \
#             {
#                 "График зависимости длины волны от фок.раст" : TypeData.LAMBDA_FROM_F,
#                 "Моделирование для 2-х линз" : TypeData.MODEL_TWO_LINSE,
#                 "Моделирование для 3-х линз" : TypeData.MODE_THREE_LINSE
#             }
#         self.cur_type_data = TypeData.LAMBDA_FROM_F

#         self.init_ui()


#     def init_ui(self):
#         self.layout = QVBoxLayout()     # как будут расположены пользов.элементы - вертикально друг под дргуом
#         self.setLayout(self.layout)

#         self.dataDisplayTypeComboBox = QComboBox()
#         self.dataDisplayTypeComboBox.addItems\
#             (
#                 [
#                     TypeData.LAMBDA_FROM_F.value,
#                     TypeData.MODEL_TWO_LINSE.value,
#                     TypeData.MODE_THREE_LINSE.value
#                 ]
#             )
#         self.dataDisplayTypeComboBox.currentTextChanged.connect(
#             self.on_data_display_type_changed)
#         self.layout.addWidget(self.dataDisplayTypeComboBox)

#         self.setDataButton = QPushButton("Задать данные")
#         self.setDataButton.clicked.connect(self.show_table)
#         self.layout.addWidget(self.setDataButton)

#         self.runButton = QPushButton("Запустить")
#         self.runButton.clicked.connect(self.on_run_btn_connect)
#         self.layout.addWidget(self.runButton)


#     def show_table(self):
#         if self.lens_count <= 0 and self.cur_type_data is not TypeData.LAMBDA_FROM_F:
#             QMessageBox.warning(self, "Ошибка", "Сначала установите кол-во линз!")
#             return
        
#         dialog = LensDataDialog(self.cur_type_data, self.lens_count, self.data_storage)
#         if dialog.exec() == QDialog.accepted:
#             self.data_storage = dialog.previous_data 
#             pass


#     def on_data_display_type_changed(self, text : str):
#         self.cur_type_data = self.type_dates[text]


#     def on_set_data_connect(self, *args):
#         print(type(args))
#         pass    
    
#     def on_run_btn_connect(self, *args):
#         print(type(args))
#         pass
        



# class LensDataDialog(QDialog):
#     def __init__(self,
#                  type_data : TypeData,
#                  lens_count : int = None, 
#                  previous_data=None):
        
#         super().__init__()
#         self.setWindowTitle(type_data.value)
#         self.resize(500, 300)
#         self.layout = QVBoxLayout()
#         self.setLayout(self.layout)

#         self.type_data = type_data
#         self.lens_count = lens_count
#         self.previous_data = previous_data

#         self.init_gui()


#     def init_gui(self):
#         match self.type_data:
#             case TypeData.LAMBDA_FROM_F:
#                 self.create_lmbd_f_table()
#             case TypeData.MODEL_TWO_LINSE:
#                 self.create_model_two_linse_table()
#             case TypeData.MODE_THREE_LINSE:
#                 self.create_model_three_linse_table()
#             case _:
#                 raise Exception("Не тот тип!")


#     def create_lmbd_f_table(self):
#         count_linse_layout = QHBoxLayout()
#         count_linse_layout.addWidget(QLabel("Число линз:"))
#         self.lens_count_input = QLineEdit()
#         count_linse_layout.addWidget(self.lens_count)
#         self.set_count_btn = QPushButton("Установить")
#         self.set_count_btn.clicked.connect(self.on_set_lens_clicked)
#         count_linse_layout.addWidget(self.set_count_btn)
#         self.layout.addLayout(count_linse_layout)

#         self.table = QTableWidget(self.lens_count, 7)
#         self.table.setHorizontalHeaderLabels\
#             (
#                 [
#                     "№ линзы",
#                     "Гармоника",
#                     "Показатель преломления",
#                     "Показатель преломления среды"
#                     "Базовая длина волны, нм",
#                     "Базовый фокус, мм",
#                     "Расстояние м/у линзами, мм",
#                 ]
#             )

#         self.table.setItem(1, 4, QTableWidgetItem(str(0)))
#         self.table.setItem(1, 7, QTableWidgetItem(str(0)))
#         self.layout.addWidget(self.table)

    
#     def create_model_two_linse_table(self):
#         pass


#     def create_model_three_linse_table(self):
#         pass


#     def create_ok_cancel_btns(self):
#         self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
#         self.buttons.accepted.connect(self.collect_data)
#         self.buttons.rejected.connect(self.reject)
#         self.layout.addWidget(self.buttons)

    

#     def on_set_lens_clicked(self):
#         try:
#             count = int(self.lens_count_input.text())
#             if count > 0:
#                 self.lens_count = count
#             else:
#                 QMessageBox.warning(self, "Ошибка", "Введите положительное число")
#         except ValueError:
#             QMessageBox.warning(self, "Ошибка", "Сначала установите кол-во линз")


#     def collect_data(self):
#         self.collect_data = []
#         for i in range(self.lens_count):
#             val_item = self.table.item(i)
#             var = 3
#         self.accept()