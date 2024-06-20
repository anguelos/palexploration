from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PIL import Image
import ast
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from .util import h_splitter


class FilterLabel(QLabel):
    def __init__(self, parent, label_width, label_height):
        super().__init__(parent)
        self._img = None
        self._patterns = None
        self._radii = 1
        self.setAlignment(Qt.AlignCenter)
        self.label_width = label_width
        self.label_height = label_height
        self.corpus_frame = None
        self.set_PIL_image(Image.new("RGBA", (label_width, label_height), (0, 0, 0, 0)), [0] * self._radii)

    def set_corpus_frame(self, corpus_frame):
        if self.corpus_frame is corpus_frame and self.corpus_frame._filter_viewer is self:
            return
        self.corpus_frame = corpus_frame
        self.corpus_frame.set_filter_viewer(self)

    def __set_pixmap(self, pixmap):
        # Override setPixmap to resize the pixmap when a new pixmap is set
        scaled_pixmap = pixmap.scaled(self.label_width, self.label_height, Qt.KeepAspectRatio, Qt.FastTransformation)
        super().setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        # Override resizeEvent to resize the pixmap when the label is resized
        if self.pixmap():
            self.__set_pixmap(self.pixmap())  # Reapply the current pixmap to resize it

    def set_PIL_image(self, pil_image, patterns):
        self._img = pil_image.convert("RGBA")
        self._patterns = patterns
        q_image = QImage(self._img.tobytes(), self._img.width, self._img.height, QImage.Format_ARGB32)
        q_pixmap = QPixmap.fromImage(q_image)
        q_pixmap = q_pixmap.scaled(self.label_width, self.label_height, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.__set_pixmap(q_pixmap)

    def mousePressEvent(self, event):
        QMessageBox.information(self, "Filter Label", f"Patterns: {self._patterns}")

    @property
    def pattern(self):
        return self._patternssetPIL


class FilterTextBox(QLineEdit):
    def __init__(self, parent):
        super(FilterTextBox, self).__init__(parent)
        self.filter_frame = parent
        assert isinstance(self.filter_frame, FilterFrame)

    def focusOutEvent(self, e):
        self.validate_content()
        super(FilterTextBox, self).focusOutEvent(e)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Enter or e.key() == Qt.Key_Return:
            self.validate_content()
        super(FilterTextBox, self).keyPressEvent(e)

    def validate_content(self):
        text = self.text()
        self.filter_frame.set_pattern_from_str(text)


class FilterFrame(QWidget):
    def __init__(self, label_width, label_height):
        super().__init__()
        self.label_width = label_width
        self.label_height = label_height
        self.corpus_frame = None
        self._radii = 1
        self._pattern = []  # This will be changed by set_null_pattern
        self.pil_img = None
        self.initUI()
        self.set_null_pattern()

    def set_corpus_frame(self, corpus_frame):
        if self.corpus_frame is corpus_frame and self.corpus_frame._filter_frame is self:
            return
        self.corpus_frame = corpus_frame
        if "radii" in self.corpus_frame.lbp_config_dict:
            self.set_radii(corpus_frame.lbp_config_dict["radii"])
        self.corpus_frame.set_filter_viewer(self)

    def initUI(self):
        self.layout = QVBoxLayout(self)
        self.titlelabel = QLabel("Pattern Viewer")
        self.titlelabel.setAlignment(Qt.AlignLeft)
        self.titlelabel.setWordWrap(True)
        self.layout.addWidget(self.titlelabel)

        self.filter_label = FilterLabel(self, label_width=self.label_width, label_height=self.label_height)
        self.filter_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.filter_label)

        self.pattern_text = FilterTextBox(self)
        self.pattern_text.setFixedWidth(self.label_width)
        self.pattern_text.setFixedHeight(16)
        self.pattern_text.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.pattern_text)

        self.patter_freq_label = QLabel("NA")
        self.patter_freq_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.patter_freq_label)

        self.layout.setAlignment(self.pattern_text, Qt.AlignCenter)
        self.layout.setAlignment(self.filter_label, Qt.AlignCenter)
        self.layout.setAlignment(self.titlelabel, Qt.AlignCenter)
        self.layout.setAlignment(self.patter_freq_label, Qt.AlignCenter)

        view_button = QPushButton("View Similar")
        view_button.clicked.connect(self.plot_similar_occurences)
        self.layout.addWidget(view_button)

        self.layout.addWidget(h_splitter())

        self.setLayout(self.layout)

    def __render_pattern_list(self, pattern_list):
        if self.corpus_frame is None:
            print("Corpus frame not set. Cannot render patterns.", file=sys.stderr)
            return
        if self.corpus_frame._lbp_layers is None:
            print("Corpus frame LBP layers not set. Cannot render patterns.", file=sys.stderr)
            return
        radii = len(pattern_list)  # TODO(anguelos): counting the patterns is not the best way to get the radii
        out_sz = radii * 2 + 5
        filters = []
        for pattern, layer in zip(pattern_list, self.corpus_frame._lbp_layers):
            filters.append(layer.point_sampler.render_pattern(pattern, out_sz, out_sz))
        if len(filters) == 0:
            print(f"No filters to render for pattern {pattern_list}.\
                with {self.corpus_frame._lbp_layers}", file=sys.stderr)
            filters = [np.ones((10, 10), dtype=np.uint8) * .5]
        filters = np.stack(filters, axis=0).sum(axis=0)
        if self.filter_label:
            filters = filters-filters.reshape(-1).min()
            filters = filters/filters.reshape(-1).max()
            filters = (filters*255).astype('uint8')
            pil_img = Image.fromarray(filters, mode='L')
            self.pil_img = pil_img
            self.filter_label.set_PIL_image(pil_img, pattern_list)
        else:
            print(f"No filter label available. Cannot render pattern {pattern_list}.", file=sys.stderr)

    def set_null_pattern(self):
        self.set_pattern([255] * self._radii)  # This more rare so it doesnt clutter the view

    def set_radii(self, radii):
        if radii < 1 or radii > 16:
            print(f"Invalid radii: {radii}", file=sys.stderr)
            return
        if self._radii != radii:
            self._radii = int(radii)
            print(f"Setting radii to {self._radii}", file=sys.stderr)
            self.set_null_pattern()

    def is_valid_pattern(self, pattern):
        if len(pattern) != self._radii:
            return False
        for p in pattern:
            if not 0 <= p < 256 or not isinstance(p, int):
                return False
        return True

    def set_pattern_from_str(self, pattern_str):
        try:
            computed_pattern = ast.literal_eval(pattern_str)
        except (SyntaxError, ValueError):
            print(f"Invalid pattern string: {pattern_str}", file=sys.stderr)
            self.set_null_pattern()
            return

        if self.is_valid_pattern(computed_pattern):
            self.set_pattern(computed_pattern)
        else:
            self.set_null_pattern()

    def set_pattern(self, pattern_list):
        assert len(pattern_list) == self._radii
        if self._pattern != pattern_list:
            self.__render_pattern_list(pattern_list)
            self.pattern_text.setText(f"{pattern_list}")
            self._pattern = [p for p in pattern_list]
            self.patter_freq_label.setText("NA")
            if self.corpus_frame is not None:
                if self.corpus_frame._image_viewer is not None:
                    self.corpus_frame._image_viewer.update_heatmap_from_pattern(self.pattern)
        self.pattern_text.setText(f"{self._pattern}")

    def set_pattern_freq(self, pattern, count, ratio):
        self.set_pattern(pattern)
        self.patter_freq_label.setText(f"Occurence: # {count},  % {(ratio*100):.5f}")

    def plot_similar_occurences(self):
        pattern = np.array(self.pattern)[:, None, None]
        print("Plot Pattern:", pattern)
        results, max_radius = self.corpus_frame.get_similar_occurences(pattern=self.pattern, max_distance=6)
        box_sz = 48
        box_slice = box_sz // 2
        if self.pil_img is not None:
            pattern = np.array(self.pil_img)
            if len(pattern.shape) == 3:
                pattern = np.stack([pattern[:, :, 0]]*3)
            else:
                pattern = np.stack([pattern]*3, axis=2)
            crops = [(pattern, -1, "")]
        else:
            crops = []
        images_paths = sorted(set([img.path for _, img, _ in results]))
        image_ids = {path: n + 1 for n, path in enumerate(images_paths)}
        print("Image_ids:", image_ids)
        for distance, img, (y, x) in results:
            box = np.zeros((box_sz, box_sz, 3), dtype=np.uint8)
            box[:, :, :] = np.array(img.img.crop((x-box_slice, y-box_slice, x+box_slice, y+box_slice)).convert("RGB"))
            crops.append((box, distance, str(image_ids[img.path])))
        print("Crops:", len(crops))
        crops = crops[:16]
        rows = 2
        cols = 8
        heights = (1 * np.ones(rows)).tolist()
        widths = (1 * np.ones(cols)).tolist()
        fig = plt.figure(constrained_layout=False)
        gs = GridSpec(rows, cols, figure=fig, width_ratios=widths, height_ratios=heights, left=0.00,
                      right=1., wspace=0.0, hspace=0.0)
        for n, (crop, distance, label) in enumerate(crops):
            row = n // cols
            col = n % cols
            ax = fig.add_subplot(gs[row, col])
            _ = ax.imshow(crop, extent=[0, 100, 0, 100])
            if distance >= 0:
                circle = plt.Circle((50, 50), max_radius * 100/box_sz, color='red', fill=False)
                ax.add_patch(circle)
                ax.text(2, 80, str(distance), fontsize=14, color='red')
            ax.text(2, 2, label, fontsize=14, color='green')
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
        #if title:
        #    fig.suptitle(title)
        plt.show(block=False)

    @property
    def pattern(self):
        return self._pattern
