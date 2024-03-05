import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction, QLineEdit, QMessageBox
from PyQt5.QtCore import pyqtSlot
import test


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'Lecturer helper 1.0'
        self.left = 500
        self.top = 500
        self.width = 500
        self.height = 200
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Create textbox
        self.textbox = QLineEdit(self)
        self.textbox.move(20, 20)
        self.textbox.resize(400, 40)

        # Create a button in the window
        self.button = QPushButton('Evaluate', self)
        self.button.move(20, 80)

        # connect button to function on_click
        self.button.clicked.connect(self.on_click)
        self.show()

    @pyqtSlot()
    def on_click(self):
        textboxValue = self.textbox.text()
        string = test.evaluate(textboxValue)
        QMessageBox.question(self, "Result", string, QMessageBox.Ok,
                             QMessageBox.Ok)
        self.textbox.setText("")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
