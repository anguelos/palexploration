from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QHBoxLayout, QComboBox, QPushButton
from .util import ExponentialSlider
import torch
import sys
from PyQt5.QtCore import Qt


class LBPConfigFrame(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.corpus_frame = None
        self.default_config_dict = {
            "radii": 2,
            "diff_hardness": self.diff_temp_scroll.get_selectable_float(10.),
            "output_hardness": self.softmax_temp_scroll.get_selectable_float(10.),
            "comparisson": "quantile",
            "device": self.__get_selected_device_str(),
            "inversed": False
        }
        self.update_gui_to_defaults()

    def set_min_size(self, width=None, height=None):
        if width is None and height is not None:
            self.setMinHeight(height)
        if width is not None and height is None:
            self.setMinWidth(width)
        if width is None and height is not None:
            self.setMinSize(width, height)

    def set_size(self, width=None, height=None):
        if width is None and height is not None:
            self.setFixedHeight(30)
        if width is not None and height is None:
            self.setFixedWidth(30)
        if width is None and height is not None:
            self.setFixedSize(width, height)

    def set_corpus_frame(self, corpus_frame):
        if self.corpus_frame is corpus_frame and self.corpus_frame._lbp_config_frame is self:
            return
        self.corpus_frame = corpus_frame
        self.corpus_frame.set_lbp_config_frame(self)
        self.corpus_frame.update_config(self.export_gui_to_conf_dict())

    def initUI(self):
        self.layout = QVBoxLayout(self)
        self.titlelabel = QLabel("LBP Configuration")
        self.titlelabel.setAlignment(Qt.AlignLeft)
        self.titlelabel.setWordWrap(True)
        self.layout.addWidget(self.titlelabel)

        hlayout = QHBoxLayout()
        hlayout.addWidget(QLabel("Radii:"))
        self.radii_scroll = QSlider(Qt.Horizontal)
        self.radii_scroll.setRange(1, 16)
        self.radii_scroll.setValue(1)
        hlayout.addWidget(self.radii_scroll)
        self.radii_label = QLabel("1")
        self.radii_label.setFixedWidth(30)
        self.radii_scroll.valueChanged.connect(lambda x: self.radii_label.setText(str(x)))
        self.radii_scroll.valueChanged.connect(self.setting_changed)
        hlayout.addWidget(self.radii_label)
        self.layout.addLayout(hlayout)

        hlayout = QHBoxLayout()
        hlayout.addWidget(QLabel("Diff hardness:"))
        self.diff_temp_scroll = ExponentialSlider(Qt.Horizontal)
        self.diff_temp_scroll.set_real_value(10.)
        self.diff_temp_scroll.processedValueChanged.connect(self.setting_changed)
        hlayout.addWidget(self.diff_temp_scroll)
        self.layout.addLayout(hlayout)

        hlayout = QHBoxLayout()
        hlayout.addWidget(QLabel("Softmax Temperture:"))
        self.softmax_temp_scroll = ExponentialSlider(Qt.Horizontal)
        self.softmax_temp_scroll.set_real_value(10.)
        self.softmax_temp_scroll.processedValueChanged.connect(self.setting_changed)
        hlayout.addWidget(self.softmax_temp_scroll)
        self.layout.addLayout(hlayout)

        hlayout = QHBoxLayout()
        hlayout.addWidget(QLabel("Comparisson layer:"))
        self.comparisson_box = QComboBox()
        self.comparisson_box.addItems(["quantile", "otsu", "simple"])
        self.comparisson_box.setCurrentIndex(0)
        self.comparisson_box.currentIndexChanged.connect(self.setting_changed)
        hlayout.addWidget(self.comparisson_box)
        self.layout.addLayout(hlayout)

        hlayout = QHBoxLayout()
        hlayout.addWidget(QLabel("Computation device"))
        self.device_box = QComboBox()
        for n in range(torch.cuda.device_count()):
            self.device_box.addItems([f"{n}: {torch.cuda.get_device_name(n)}"])
        self.device_box.addItems([f"{torch.cuda.device_count()}: cpu"])
        self.device_box.setCurrentIndex(0)
        self.device_box.currentIndexChanged.connect(self.setting_changed)
        hlayout.addWidget(self.device_box)
        self.layout.addLayout(hlayout)

        hlayout = QHBoxLayout()
        hlayout.addWidget(QLabel("Image Mode:"))
        self.inversed_box = QComboBox()
        self.inversed_box.addItems(["Normal", "Inversed"])
        self.inversed_box.setCurrentIndex(0)
        self.inversed_box.currentIndexChanged.connect(self.setting_changed)
        hlayout.addWidget(self.inversed_box)
        self.layout.addLayout(hlayout)

        hlayout = QHBoxLayout()
        self.button_reset_defaults = QPushButton("Defaults")
        self.button_reset_defaults.clicked.connect(self.update_gui_to_defaults)
        hlayout.addWidget(self.button_reset_defaults)

        self.button_reload_active = QPushButton("Active")
        self.button_reload_active.clicked.connect(self.update_gui_from_corpus_frame_conf_dict)
        hlayout.addWidget(self.button_reload_active)

        self.button_apply = QPushButton("Apply")
        self.button_apply.clicked.connect(self.apply_config_to_corpus_frame)
        hlayout.addWidget(self.button_apply)
        self.layout.addLayout(hlayout)

        self.setLayout(self.layout)

    def setting_changed(self):
        active_config_dict = self.export_gui_to_conf_dict()
        corpus_config = {}
        if self.corpus_frame is not None:
            corpus_config.update(self.corpus_frame.lbp_config_dict)

        if corpus_config == active_config_dict:
            self.button_apply.setEnabled(False)
            self.button_reload_active.setEnabled(False)
        else:
            self.button_apply.setEnabled(True)
            self.button_reload_active.setEnabled(True)

        if active_config_dict != self.default_config_dict:
            self.button_reset_defaults.setEnabled(True)
        else:
            self.button_reset_defaults.setEnabled(False)

    def apply_config_to_corpus_frame(self):
        if self.corpus_frame is None:
            print("Corpus frame not set. Cannot apply config.", file=sys.stderr)
            return
        self.setEnabled(False)
        self.corpus_frame.update_config(self.export_gui_to_conf_dict())
        self.corpus_frame.compute_one_if_all_uncomputed()
        self.setEnabled(True)
        self.setting_changed()

    def update_gui_from_corpus_frame_conf_dict(self):
        if self.corpus_frame is None or not self.corpus_frame.lbp_config_dict:
            print("Corpus frame or it's config not set. Cannot get config.", file=sys.stderr)
            return
        self.update_gui_from_config_dict(self.corpus_frame.lbp_config_dict)
        self.setting_changed()

    def update_gui_to_defaults(self):
        self.update_gui_from_config_dict(self.default_config_dict)
        self.setting_changed()

    def update_gui_from_config_dict(self, config):
        if isinstance(config, dict):
            config.update(config)
        else:
            print(f"LBP CONFIG: Invalid config type: {type(config)}", file=sys.stderr)
            raise ValueError("Invalid config type")

        self.radii_scroll.setValue(config["radii"])
        self.diff_temp_scroll.set_real_value(config["diff_hardness"])
        self.softmax_temp_scroll.set_real_value(config["output_hardness"])

        comparisson_options = [self.comparisson_box.itemText(i) for i in range(self.comparisson_box.count())]
        assert config["comparisson"] in comparisson_options
        self.comparisson_box.setCurrentIndex(self.comparisson_box.findText(config["comparisson"]))
        if config["device"].find("cpu") >= 0:
            config_choice = torch.cuda.device_count()
        else:
            config_choice = int(config["device"].split(":")[1])
        self.device_box.setCurrentIndex(config_choice)
        if config["inversed"]:
            self.inversed_box.setCurrentIndex(1)
        else:
            self.inversed_box.setCurrentIndex(0)

    def __get_selected_device_str(self):
        device = "cpu"
        have_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
        if have_gpu and self.device_box.currentIndex() < torch.cuda.device_count():
            device = f"cuda:{self.device_box.currentIndex()}"
        else:
            device = "cpu"
        return device

    def export_gui_to_conf_dict(self):
        return {
            "radii": self.radii_scroll.value(),
            "diff_hardness": self.diff_temp_scroll.get_real_value(),
            "output_hardness": self.softmax_temp_scroll.get_real_value(),
            "comparisson": self.comparisson_box.currentText(),
            "device": self.__get_selected_device_str(),
            "inversed": self.inversed_box.currentIndex() == 1
        }
