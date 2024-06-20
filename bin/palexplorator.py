#!/usr/bin/env python3
from PyQt5.QtWidgets import QApplication
import sys
from palexploration import InteractiveViewerFrame


def main():
    app = QApplication(sys.argv)
    viewer = InteractiveViewerFrame()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
