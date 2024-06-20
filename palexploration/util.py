from PyQt5.QtWidgets import QLabel, QFrame, QVBoxLayout, QWidget, QSlider
from PyQt5.QtCore import Qt, pyqtSignal
import sys
import numpy as np


def h_splitter():
    separator = QFrame()
    separator.setFrameShape(QFrame.HLine)  # Horizontal line
    separator.setFrameShadow(QFrame.Sunken)
    return separator


def v_splitter():
    separator = QFrame()
    separator.setFrameShape(QFrame.VLine)  # Horizontal line
    separator.setFrameShadow(QFrame.Sunken)
    return separator


class ExponentialSlider(QWidget):

    processedValueChanged = pyqtSignal(float)

    def __init__(self, orientation=Qt.Horizontal):
        super().__init__()
        self.orientation = orientation
        self.init_ui()
        self.linear_scale = .1
        self.exponential_scale = 333.33

    def init_ui(self):
        # Vertical layout for the widget
        layout = QVBoxLayout()
        # Label to display slider value
        self.label = QLabel('Value: 0.01', self)
        layout.addWidget(self.label)

        # Slider setup
        self.slider = QSlider(orientation=self.orientation, parent=self)
        self.slider.setMinimum(1)
        self.slider.setMaximum(1000)  # A good range for precision
        self.slider.valueChanged.connect(self.update_label)
        layout.addWidget(self.slider)

        self.setLayout(layout)

    def update_label(self):
        # Map the slider value to the exponential scale
        value = self.get_real_value()
        self.label.setText(f'Value: {value:.2f}')
        self.processedValueChanged.emit(value)

    def __float_to_int(self, value: float):
        return int((self.exponential_scale * np.log10(value/self.linear_scale)).round())

    def __int_to_float(self, value: int):
        return self.linear_scale * (10 ** (value / self.exponential_scale))

    def __stable_int_to_float(self, value: int):
        return self.__stabilise_float_value(self.__int_to_float(value))

    def __stable_float_to_int(self, value: float):
        return self.__float_to_int(self.__stabilise_float_value(value))

    def __chop_float_value(self, value):
        r_min, r_max = self.get_real_range()
        if value < r_min:
            value = 0.01
            print(f"Truncate Value out of range. Setting to {r_min}.", file=sys.stderr)
        elif value > r_max:
            value = r_max
            print("Truncate Value out of range. Setting to {r_max}.", file=sys.stderr)
        return value

    def get_selectable_float(self, value):
        chooped = self.__chop_float_value(value)
        stable = self.__stabilise_float_value(chooped)
        return stable

    def __stabilise_float_value(self, value):
        float_val = self.__int_to_float(self.__float_to_int(value))
        assert float_val == self.__int_to_float(self.__float_to_int(float_val))
        return float_val

    def get_real_value(self):
        return self.__stable_int_to_float(self.slider.value())

    def get_real_range(self):
        r_min = self.__stable_int_to_float(self.slider.minimum())
        r_max = self.__stable_int_to_float(self.slider.maximum())
        return r_min, r_max

    def set_real_value(self, value):
        value = self.__chop_float_value(value)
        int_val = self.__stable_float_to_int(value)
        self.slider.setValue(int_val)
