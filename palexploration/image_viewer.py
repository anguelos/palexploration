from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QSlider, QPushButton, QFileDialog
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from PIL import Image, ImageDraw
import numpy as np
import scipy.ndimage
import sys
from .corpus_frame import CorpusFrame, ImageFile
from .pattern_frame import FilterFrame
from .lbp_config import LBPConfigFrame


class ClickableImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setText("Select image")
        self.setMinimumSize(40, 40)
        self.zoom_level = 1.
        self.heatmap = None
        self.filter_frame = None
        self.corpus_image = None
        self.corpus_frame = None
        self.clicked_color = 'red'
        self.heatmap_opacity = 128
        self.heatmap_blue = 0
        self.heatmap_green = 255
        self.heatmap_red = 0
        self.heatmap_dilate = 1

    def set_filter_frame(self, filter_frame: FilterFrame):
        self.filter_frame = filter_frame

    def set_corpus_frame(self, corpus_frame):
        if self.corpus_frame is corpus_frame and corpus_frame._image_viewer is self:
            return
        self.corpus_frame = corpus_frame
        self.corpus_frame.set_image_viewer(self)
        if corpus_frame.selected_image is not None:
            self.corpus_image = corpus_frame.selected_image
            self.render_image()
        if self.corpus_frame._image_viewer is not self:
            self.corpus_frame.set_image_viewer(self)

    def set_image(self, image: ImageFile):
        if image is None or image.requires_computation:
            if image is not None:
                print(f"Image {image.path} requires computation. Cannot set.", file=sys.stderr)
                image.why_requires_computation()
            else:
                print("Set to No image.", file=sys.stderr)
            self.corpus_image = None
            self.image = None
            self.heatmap = None
            self.update_buffer_image()
            self.update_pixmap()
            return
        self.corpus_image = image
        self.image = image.img
        self.heatmap = None
        if self.filter_frame is not None and self.filter_frame.pattern is not None and not image.requires_computation:
            heatmap, range = self.corpus_image.get_similarity_map(pattern=self.filter_frame.pattern, xy=None)
            self.heatmap = heatmap / range
        self.update_buffer_image()
        self.update_pixmap()

    def update_buffer_image(self):
        if self.heatmap is not None:
            rgba = np.zeros([self.heatmap.shape[0], self.heatmap.shape[1], 4], dtype=np.uint8)
            rgba[:, :, 0] = self.heatmap_red
            rgba[:, :, 1] = self.heatmap_green
            rgba[:, :, 2] = self.heatmap_blue
            dilated_heatmap = scipy.ndimage.maximum_filter(self.heatmap, self.heatmap_dilate)
            rgba[:, :, 3] = self.heatmap_opacity * dilated_heatmap
            rgba = Image.fromarray(rgba, mode='RGBA')
            img = self.image.convert('RGBA')
            img.putalpha(255 - self.heatmap_opacity)
            self._buffer_img = Image.alpha_composite(self.image.convert('RGBA'), rgba)
        elif self.image is not None:
            self._buffer_img = self.image.convert('RGBA')
        else:
            self._buffer_img = None
            print("No image to update buffer.", file=sys.stderr)

    def update_pixmap(self):
        if self._buffer_img is None:
            self.setPixmap(QPixmap())
            self.setText("No image")
            print("No self._buffer_img to update pixmap.", file=sys.stderr)
            return
        width = int(self._buffer_img.width * self.zoom_level)
        height = int(self._buffer_img.height * self.zoom_level)
        # QImage buffer Format_ARGB32 seems to require [B, G, R, A] order probably an endianness issue
        self._buffer_argb = np.array(self._buffer_img, dtype=np.uint8)[:, :, [2, 1, 0, 3]].tobytes()
        qt_img = QImage(self._buffer_argb, self._buffer_img.width, self._buffer_img.height, QImage.Format_ARGB32)
        qt_img = qt_img.scaled(width, height, Qt.KeepAspectRatio)
        pixmap = QPixmap.fromImage(qt_img)
        self.setPixmap(pixmap)
        self.resize(pixmap.size())

    def update_heatmap_from_pattern(self, pattern_list, do_update_pixmap=True):
        if self.corpus_image is None:
            print("No image to update heatmap from pattern.", file=sys.stderr)
            return
        if not self.corpus_image.requires_computation:
            heatmap, radii = self.corpus_image.get_similarity_map(xy=None, pattern=pattern_list)
            self.heatmap = (heatmap / radii) ** 2
            correct_count = (heatmap == radii).sum()
            correct_ratio = correct_count / heatmap.size
            self.corpus_frame._filter_viewer.set_pattern_freq(pattern_list, correct_count, correct_ratio)
        else:
            self.heatmap = None
        self.update_buffer_image()
        if do_update_pixmap:
            self.update_pixmap()

    def mousePressEvent(self, event):
        if self.corpus_image:
            display_size = self.pixmap().size()
            scale_width = self._buffer_img.width / display_size.width()
            scale_height = self._buffer_img.height / display_size.height()
            original_x = int(event.x() * scale_width)
            original_y = int(event.y() * scale_height)
            if not self.corpus_image.requires_computation:
                patterns = self.corpus_image.lbp_patterns
                if not 0 <= original_x < patterns.shape[2] or not 0 <= original_y < patterns.shape[1]:
                    print(f"Mouse Pressed out of bounds: Scaled {original_x}, {original_y}", file=sys.stderr)
                    print(f"\tPatterns: {patterns.shape[2]}, {patterns.shape[1]}", file=sys.stderr)
                    print(f"\tOriginal: {event.x()}, {event.y()}", file=sys.stderr)
                    print(f"\tDisplay: {display_size.width()}, {display_size.height()}", file=sys.stderr)
                    print(f"\tScale: {scale_width}, {scale_height} Zoom:{self.zoom_level}\n\n", file=sys.stderr)
                    return
                pattern = self.corpus_image.lbp_patterns[:, original_y:original_y+1, original_x:original_x+1]
                pattern_list = pattern[:, 0, 0].astype(int).tolist()
            else:
                print("Mouse Pressed Image not computed yet.", file=sys.stderr)
            self.update_heatmap_from_pattern(pattern_list, do_update_pixmap=False)
            self.corpus_frame._filter_viewer.set_pattern(pattern_list)
            draw = ImageDraw.Draw(self._buffer_img)
            radii = self.corpus_frame.lbp_config_dict["radii"]
            if self.clicked_color != "none":
                draw.ellipse((original_x - radii, original_y-radii, original_x+radii,
                              original_y+radii), fill=None, outline=self.clicked_color)
            self.update_pixmap()


class InteractiveViewerFrame(QWidget):
    def __init__(self):
        super().__init__()
        self.corpus = CorpusFrame()
        self.lbp_config = LBPConfigFrame()
        self.filter_frame = FilterFrame(200, 200)
        self.image_label = ClickableImageLabel()
        self.initUI()
        self.corpus.set_image_viewer(self.image_label)
        self.corpus.set_filter_viewer(self.filter_frame)
        self.corpus.set_lbp_config_frame(self.lbp_config)
        self.image_label.set_filter_frame(self.filter_frame)
        self.lbp_config.set_corpus_frame(self.corpus)

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

    def initUI(self):
        self.h_layout1 = QHBoxLayout()
        self.image_layout = QVBoxLayout()
        self.naviagtion_layout = QVBoxLayout()

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)  # Important to keep the image at its original size

        self.image_label.setText("Select image")
        self.scroll_area.setWidget(self.image_label)
        self.image_layout.addWidget(self.scroll_area)

        image_controls_layout = QHBoxLayout()

        image_color_layout = QVBoxLayout()
        image_color_layout.addWidget(QLabel("Heatmap Color:"))
        image_red_layout = QHBoxLayout()
        image_red_layout.addWidget(QLabel("Red  "))
        self.red_heatmap_slider = QSlider(Qt.Horizontal)
        self.red_heatmap_slider.setRange(0, 255)
        self.red_heatmap_slider.setValue(0)
        self.red_heatmap_slider.valueChanged.connect(self.set_image_label_choices)
        image_red_layout.addWidget(self.red_heatmap_slider)
        image_color_layout.addLayout(image_red_layout)

        image_green_layout = QHBoxLayout()
        image_green_layout.addWidget(QLabel("Green"))
        self.green_heatmap_slider = QSlider(Qt.Horizontal)
        self.green_heatmap_slider.setRange(0, 255)
        self.green_heatmap_slider.setValue(255)
        self.green_heatmap_slider.valueChanged.connect(self.set_image_label_choices)
        image_green_layout.addWidget(self.green_heatmap_slider)
        image_color_layout.addLayout(image_green_layout)

        image_blue_layout = QHBoxLayout()
        image_blue_layout.addWidget(QLabel("Blue "))
        self.blue_heatmap_slider = QSlider(Qt.Horizontal)
        self.blue_heatmap_slider.setRange(0, 255)
        self.blue_heatmap_slider.setValue(0)
        self.blue_heatmap_slider.valueChanged.connect(self.set_image_label_choices)
        image_blue_layout.addWidget(self.blue_heatmap_slider)
        image_color_layout.addLayout(image_blue_layout)
        image_controls_layout.addLayout(image_color_layout)

        heatmap_options2_layout = QVBoxLayout()

        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("Heatmap Opacity"))
        self.heatmap_opacity_slider = QSlider(Qt.Horizontal)
        self.heatmap_opacity_slider.setRange(0, 255)
        self.heatmap_opacity_slider.setValue(127)
        self.heatmap_opacity_slider.valueChanged.connect(self.set_image_label_choices)
        opacity_layout.addWidget(self.heatmap_opacity_slider)
        heatmap_options2_layout.addLayout(opacity_layout)

        clicked_layout = QHBoxLayout()
        clicked_layout.addWidget(QLabel("Clicked Color:"))
        self.clicked_color_combo = QComboBox()
        self.clicked_color_combo.addItems(["red", "green", "blue", "black", "white", "none"])
        self.clicked_color_combo.setCurrentIndex(0)
        self.clicked_color_combo.currentIndexChanged.connect(self.set_image_label_choices)
        clicked_layout.addWidget(self.clicked_color_combo)
        heatmap_options2_layout.addLayout(clicked_layout)

        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Zoom"))
        self.heatmap_zoom_slider = QSlider(Qt.Horizontal)
        self.heatmap_zoom_slider.setRange(0, 8)
        self.heatmap_zoom_slider.setValue(4)
        self.heatmap_zoom_slider.valueChanged.connect(self.set_image_label_choices)
        zoom_layout.addWidget(self.heatmap_zoom_slider)
        heatmap_options2_layout.addLayout(zoom_layout)

        heatmap_dilate_layout = QHBoxLayout()
        heatmap_dilate_layout.addWidget(QLabel("Heatmap Dilate"))
        self.heatmap_dilation_slider = QSlider(Qt.Horizontal)
        self.heatmap_dilation_slider.setRange(1, 20)
        self.heatmap_dilation_slider.setValue(1)
        self.heatmap_dilation_slider.valueChanged.connect(self.set_image_label_choices)
        heatmap_dilate_layout.addWidget(self.heatmap_dilation_slider)
        heatmap_options2_layout.addLayout(heatmap_dilate_layout)
        image_controls_layout.addLayout(heatmap_options2_layout)

        export_layout = QVBoxLayout()
        export_heatmap_button = QPushButton("Export Heatmap")
        export_heatmap_button.clicked.connect(self.export_heatmap)
        export_layout.addWidget(export_heatmap_button)
        export_overlayed_button = QPushButton("Export Overlayed")
        export_overlayed_button.clicked.connect(self.export_overlayed)
        export_layout.addWidget(export_overlayed_button)
        image_controls_layout.addLayout(export_layout)

        self.image_layout.addLayout(image_controls_layout)

        self.naviagtion_layout.addWidget(self.filter_frame)
        self.naviagtion_layout.addWidget(self.corpus)
        self.naviagtion_layout.addWidget(self.lbp_config)

        self.h_layout1.addLayout(self.image_layout, 1)
        self.h_layout1.addLayout(self.naviagtion_layout, 0)
        self.setLayout(self.h_layout1)

        self.show()

    def set_image_label_choices(self):
        self.image_label.clicked_color = self.clicked_color_combo.currentText()
        self.image_label.heatmap_opacity = self.heatmap_opacity_slider.value()
        self.image_label.heatmap_blue = self.blue_heatmap_slider.value()
        self.image_label.heatmap_green = self.green_heatmap_slider.value()
        self.image_label.heatmap_red = self.red_heatmap_slider.value()
        self.image_label.heatmap_dilate = self.heatmap_dilation_slider.value()
        zoom = [.125, .25, .5, .75, 1., 1.5, 2., 3., 4.][self.heatmap_zoom_slider.value()]
        self.image_label.zoom_level = zoom
        self.image_label.update_buffer_image()
        self.image_label.update_pixmap()

    def export_heatmap(self):
        if self.filter_frame.pattern is None:
            print("No pattern selected. Cannot export heatmap.", file=sys.stderr)
            return
        pattern_as_fname = '_'.join(map(str, self.filter_frame.pattern))
        img_dir = self.image_label.corpus_image.path.parent
        img_filename = self.image_label.corpus_image.path.stem
        target_filename = img_dir / f"{img_filename}_hmonly_P{pattern_as_fname}.png"
        target_filename, _ = QFileDialog.getSaveFileName(self, "Save Heatmap", str(target_filename), "PNG (*.png)")
        heatmap_img = Image.fromarray((self.image_label.heatmap * 255).astype('uint8'))
        print("Saving heatmap to:", target_filename, file=sys.stderr)
        heatmap_img.save(target_filename)

    def export_overlayed(self):
        if self.filter_frame.pattern is None:
            print("No pattern selected. Cannot export heatmap.", file=sys.stderr)
            return
        pattern_as_fname = '_'.join(map(str, self.filter_frame.pattern))
        img_dir = self.image_label.corpus_image.path.parent
        img_filename = self.image_label.corpus_image.path.stem
        target_filename = img_dir / f"{img_filename}_hmoverlay_P{pattern_as_fname}.png"
        target_filename, _ = QFileDialog.getSaveFileName(self, "Save Heatmap", str(target_filename), "PNG (*.png)")
        print("Savingd overlay to :", target_filename, file=sys.stderr)
        self.image_label._buffer_img.save(target_filename)
