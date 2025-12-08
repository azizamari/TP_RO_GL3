import PyQt5
import sys
from PyQt5.QtWidgets import QApplication, QLabel

app = QApplication(sys.argv)

label = QLabel("Hello PyQt!")
label.show()

sys.exit(app.exec_())