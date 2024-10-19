import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Minimal PyQt5 Window")
        self.setGeometry(100, 100, 800, 600)

        # Create buttons
        button1 = QPushButton("Button 1")
        button2 = QPushButton("Button 2")

        # Set layout
        layout = QVBoxLayout()
        layout.addWidget(button1)
        layout.addWidget(button2)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
