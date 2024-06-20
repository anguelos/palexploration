from PIL import Image, UnidentifiedImageError
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget, QListWidgetItem, QMessageBox, QProgressBar
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib import pyplot as plt
import sys
import time
import torch
import numpy as np
import requests
from requests.exceptions import RequestException, Timeout
from io import BytesIO
from pathlib import Path
import os
import ptlbp
from typing import List


class ImageFile(QListWidgetItem):
    def __init__(self, path):
        super().__init__(path.name)
        self.disable()
        self.path = path
        self.img = Image.open(path)
        self.lbp_patterns = None
        self.confidences = None
        self.__requires_computation = True
        self.lbp_config = None

    @property
    def requires_computation(self):
        requires = self.lbp_patterns is None
        #  requires = requires or self.confidences is None  # confidences is disabled
        requires = requires or self.__requires_computation
        requires = requires or not bool(self.lbp_config)
        requires = requires or self.is_disabled()
        return requires

    def why_requires_computation(self):
        print(f"Image {self.text()} requires computation: {self.requires_computation}", file=sys.stderr)
        print(f"lbp_patterns is None: {self.lbp_patterns is None}")
        print(f">>> confidences is None: {self.confidences is None}")
        print(f"__requires_computation: {self.__requires_computation}")
        print(f"not bool(lbp_config): {bool(self.lbp_config)} lbp_config = {self.lbp_config}")
        print(f"is_disabled: {self.is_disabled()} Qt.ItemIsEnabled:{self.flags() & Qt.ItemIsEnabled != 0}\
            Qt.ItemIsSelectable:{self.flags() & Qt.ItemIsSelectable != 0}")

    def discard_computation(self):
        self.lbp_patterns = None
        self.confidences = None
        self.__requires_computation = True
        self.lbp_config = None
        self.disable()

    def disable(self):
        self.setFlags(self.flags() & ~Qt.ItemIsEnabled & ~Qt.ItemIsSelectable)

    def enable(self):
        self.setFlags(self.flags() | Qt.ItemIsEnabled | Qt.ItemIsSelectable)

    def is_enabled(self):
        return self.flags() & Qt.ItemIsEnabled and self.flags() & Qt.ItemIsSelectable

    def is_disabled(self):
        return not self.is_enabled()

    def update_lbps(self, lbp_list, use_confidence=True, config=None):
        if isinstance(config, dict):
            if self.lbp_config == config:
                print(f"Image {self.text()} already computed with this config. Skipping.", file=sys.stderr)
                return
        self.disable()
        device = lbp_list[0].device
        self.lbp_patterns = []
        self.confidences = []
        np_img = np.array(self.img.convert('L'), dtype=float) / 255.
        pt_img = torch.tensor(np_img[None, None, :, :], dtype=torch.float32, device=device)
        if config["inversed"]:
            pt_img = 1. - pt_img
        with torch.no_grad():
            for lbp in lbp_list:
                lbp = lbp.to(device)
                desired_slice_width = 100000 // pt_img.size(2) - lbp.point_sampler.weights.size(2)
                try:
                    pattern_img, confidence_img = lbp.compute_lbp_image(pt_img, desired_slice_width)
                    time.sleep(0.001)
                except torch.cuda.OutOfMemoryError:
                    self.discard_computation()
                    QMessageBox.information(None, "Out of Memory",
                                            f"Out of Memory error while computing {self.path}.\
                                                \n Try computing with CPU.")
                    return
                self.lbp_patterns.append(pattern_img.cpu())
                if use_confidence:
                    self.confidences.append(confidence_img.cpu())
            self.lbp_patterns = torch.cat(self.lbp_patterns, dim=1)[0, :, :, :].numpy()
            if use_confidence:
                self.confidences = torch.cat(self.confidences, dim=1).numpy()
            else:
                self.confidences = None
        self.__requires_computation = False
        self.lbp_config = config
        self.enable()

    def get_similarity_map(self, xy, pattern):
        if not self.is_enabled():
            print(f"Image {self.text()} is disabled. Aborting similarity map computation.", file=sys.stderr)
            raise Exception(f"Image {self.text()} is disabled. Aborting similarity map computation.")
        self.disable()
        if self.lbp_patterns is None:
            raise ValueError("LBP patterns are not computed yet.")
        if xy is not None and pattern is None:
            x, y = xy
            assert 0 <= x < self.lbp_patterns.shape[2] and 0 <= y < self.lbp_patterns.shape[1]
            pattern = self.lbp_patterns[:, y:y+1, x:x+1]
        elif type(pattern) in (tuple, list):
            pattern = np.array(pattern, dtype=np.uint8)[:, None, None]
        elif isinstance(pattern, np.ndarray):
            pattern = pattern.astype(np.uint8)
            # This should raise an exception if the shape is wrong, it is also a sanity check
            pattern = pattern.reshape(self.lbp_patterns.shape[0], 1, 1)
        else:
            raise ValueError(f"Invalid pattern type: {type(pattern)}")
        print(f"Pattern shape: {pattern.shape} {pattern}", file=sys.stderr)
        match_cube = (pattern == self.lbp_patterns)
        if self.confidences is not None:
            match_cube = match_cube * self.confidences
        agrement_count = match_cube.sum(axis=0)
        print("Agreemnet:", agrement_count.reshape(-1).sum())
        self.enable()
        return agrement_count, self.lbp_patterns.shape[0]

    def get_similar_occurences(self, pattern, max_distance):
        similiarity_map, best_possible = self.get_similarity_map(pattern=pattern, xy=None)
        similiarity_map = best_possible + 1 - similiarity_map
        results = {"img": np.array(self.img)}
        for n in range(max_distance):
            results[n] = np.argwhere(similiarity_map == n)
        return results, best_possible


class CustomListWidget(QListWidget):
    disabledItemClick = pyqtSignal(QListWidgetItem)

    def __init__(self, parent):
        super().__init__(parent)
        self.path2img = {}
        self.path2item = {}
        self.setDragDropMode(self.InternalMove)
        self.setAcceptDrops(True)
        self.accept_url_drops = True
        self.timeout = 5
        if isinstance(parent, CorpusFrame):
            self.corpus_frame = parent
        else:
            self.corpus_frame = None

    def mousePressEvent(self, event):
        super(CustomListWidget, self).mousePressEvent(event)  # Call the base class method
        item = self.itemAt(event.pos())  # Get the item at the click position
        if item and not item.is_enabled():  # Check if the item is disabled
            self.disabledItemClick.emit(item)  # Emit the custom signal

    def get_image_paths_from_droped(self, mimeData):
        images = []
        for format in mimeData.formats():
            if format.startswith('text/uri-list'):
                urls = mimeData.urls()
                for url in urls:
                    path = url.toLocalFile()
                    if os.path.isfile(path):
                        # It's a local file, directly return the path
                        images.append(path)
                    elif self.accept_url_drops:
                        # It's a URL, try to download and convert to a PIL image
                        image_url = url.toString()
                        if image_url.startswith('http://') or image_url.startswith('https://'):
                            try:
                                response = requests.get(image_url, timeout=self.timeout)
                                response.raise_for_status()  # Raise an exception for bad responses
                                image_bytes = BytesIO(response.content)
                                try:
                                    image = Image.open(image_bytes)
                                except UnidentifiedImageError:
                                    print(f"Failed to open image from {image_url}. Skipping.", file=sys.stderr)
                                    continue
                                else:
                                    image_cache_path = Path(f"/tmp/ptlbp_{abs(hash(image_url))}.png")
                                    image.save(image_cache_path)
                                    images.append(image_cache_path)
                            except (RequestException, Timeout) as e:
                                print(f"Failed to download or convert image from {image_url}: {e}")
        print(f"Images dropped: {images}", file=sys.stderr)
        return [Path(img_name) for img_name in images]

    def triger_compute_one_if_needed(self):
        if self.corpus_frame is not None:
            self.corpus_frame.compute_one_if_all_uncomputed()
        else:
            print("No corpus frame available. Cannot compute.", file=sys.stderr)

    def unset_computation(self):
        for item in self.path2img.values():
            item.discard_computation()

    def change_item_order(self, item_names):
        assert set(item_names) == set(self.path2img.keys())
        self.setEnabled(False)
        while self.count() > 0:
            _ = self.takeItem(0)
        for name in item_names:
            self.addItem(self.path2img[name])
        self.setEnabled(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        image_paths = self.get_image_paths_from_droped(event.mimeData())
        if len(image_paths) == 0:
            event.ignore()
        else:
            for p in image_paths:
                if p in self.path2img:
                    continue
                try:
                    item = ImageFile(p)
                except UnidentifiedImageError:
                    print(f"Could not load image from {p}. Skipping.", file=sys.stderr)
                    continue
                else:
                    self.addItem(item)
                    self.path2img[p] = item
            event.accept()
            self.triger_compute_one_if_needed()
        return


class CorpusFrame(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self._lbp_layers = []
        self._lbp_config = {}
        self._image_viewer = None
        self._filter_viewer = None
        self._lbp_config_frame = None

    def set_lbp_config_frame(self, lbp_config_frame):
        if self._lbp_config_frame is lbp_config_frame and self._lbp_config_frame.corpus_frame is self:
            return
        self._lbp_config_frame = lbp_config_frame
        self._lbp_config_frame.set_corpus_frame(self)

    def set_image_viewer(self, image_viewer):
        self._image_viewer = image_viewer
        if self._image_viewer.corpus_frame is not self:
            self._image_viewer.set_corpus_frame(self)

    def set_filter_viewer(self, filter_viewer):
        if self._filter_viewer is filter_viewer and self._filter_viewer.corpus_frame is self:
            return
        self._filter_viewer = filter_viewer
        if self._filter_viewer.corpus_frame is not self:
            self._filter_viewer.set_corpus_frame(self)

    @property
    def lbp_config_dict(self) -> dict:
        return self._lbp_config.copy()

    @property
    def selected_image(self) -> ImageFile | None:
        if self.list_widget.currentItem() is not None:
            return self.list_widget.currentItem()
        else:
            return None

    @property
    def images(self) -> List[ImageFile]:
        return list(self.list_widget.path2img.values())

    def update_config(self, config):
        if config != self._lbp_config:
            self._lbp_config = {}
            self._lbp_config.update(config)
        else:
            print("LBP config already set to:", config)
            return
        self._lbp_layers = []
        for radius in range(1, self._lbp_config["radii"] + 1):
            layer = ptlbp.DiffLBP(f"8r{radius}", supress_zero_pattern=False, supress_full_pattern=False,
                                  comparisson=config["comparisson"], block_normalise=False,
                                  diff_hardness=config["diff_hardness"], output_hardness=config["output_hardness"])
            layer = layer.to(config["device"])
            self._lbp_layers.append(layer)
        self.list_widget.unset_computation()
        self._filter_viewer.set_radii(config["radii"])

    def initUI(self):
        self.layout = QVBoxLayout(self)
        hlayout = QHBoxLayout()
        self.titlelabel = QLabel("Ready")
        self.titlelabel.setAlignment(Qt.AlignLeft)
        self.titlelabel.setWordWrap(True)
        hlayout.addWidget(self.titlelabel)
        self.statusbar = QProgressBar()
        self.statusbar.hide()
        hlayout.addWidget(self.statusbar)
        self.layout.addLayout(hlayout)

        self.list_widget = CustomListWidget(self)
        self.layout.addWidget(self.list_widget)
        self.list_widget.itemSelectionChanged.connect(self.show_selected_item)
        self.list_widget.disabledItemClick.connect(lambda item: self.compute_and_show(item))

        self.computebutton = QPushButton('Compute LBP')
        self.computebutton.clicked.connect(self.compute_all_uncomputed)
        self.layout.addWidget(self.computebutton)

        self.searchbutton = QPushButton('Search LBP')
        self.searchbutton.clicked.connect(self.search_pattern)
        self.layout.addWidget(self.searchbutton)
        self.setLayout(self.layout)

    def show_selected_item(self):
        item = self.list_widget.currentItem()
        if item and (item.flags() & Qt.ItemIsEnabled):
            if self._image_viewer is not None:
                self._image_viewer.set_image(item)
            else:
                QMessageBox.information(self, "No Viewer available Item Selected", f"Content: {item.text()}")
        else:
            QMessageBox.information(self, "Can not view unprocessed Item", f"Content: {item.text()}")

    def select_most_apropriate_item(self):
        computed_items = self.computed_items
        if len(computed_items) == 0:
            self.titlelabel.setText("Drop an image to select.")
            return
        if self.list_widget.currentItem() in computed_items:
            if not self.list_widget.currentItem().requires_computation:
                self.show_selected_item()
                return
        else:
            self.list_widget.setCurrentItem(computed_items[0])
            self.show_selected_item()

    @property
    def uncomputed_items(self):
        return [item for item in self.list_widget.path2img.values() if item.requires_computation]

    @property
    def computed_items(self):
        return [item for item in self.list_widget.path2img.values() if not item.requires_computation]

    def compute_and_show(self, item):
        self.compute_item(item)
        self.list_widget.setCurrentItem(item)
        self.show_selected_item()

    def compute_item(self, item):
        t = time.time()
        if item is not None:
            if item.requires_computation:
                item.update_lbps(self._lbp_layers, use_confidence=False, config=self.lbp_config_dict)
                print(f"Computed LBP for {item.path} {(item.lbp_patterns.shape)}", file=sys.stderr)
            else:
                print(f"Skipping {item.path}. Already computed with this config.", file=sys.stderr)
            if item.requires_computation:
                print(f"Failed to compute LBP for {item.path}.", file=sys.stderr)
                item.why_requires_computation()
                print("\n\n", file=sys.stderr)
            else:
                print(f"Computed LBP for {item.path}.", file=sys.stderr)
        return time.time() - t

    def compute_one_uncomputed(self):
        uncomputed_items = self.uncomputed_items
        if len(uncomputed_items) == 0:
            self.titlelabel.setText("No uncomputed images.")
            return
        item = uncomputed_items[0]
        seconds = self.compute_item(item)
        self.titlelabel.setText(f"Computated {item.path} in {seconds:.2f} sec.!")
        self._image_viewer.set_image(item)

    def compute_one_if_all_uncomputed(self):
        computed_items = self.computed_items
        if len(computed_items) > 0:
            self.titlelabel.setText("There is an image already computed.")
        else:
            self.compute_one_uncomputed()
        self.select_most_apropriate_item()

    def compute_all_uncomputed(self):
        # TODO (anguelos): Put this in a QThread, or some better way to escape the GIL
        self.setEnabled(False)
        self._image_viewer.set_image(None)
        original_style_sheet = self.styleSheet()
        self.setStyleSheet("{ background-color: yellow; color: black; }")
        self.titlelabel.setText("Computing LBPs:")
        uncomputed_items = self.uncomputed_items
        self.statusbar.setRange(0, len(uncomputed_items))
        self.statusbar.show()
        seconds = 0.
        for n, item in enumerate(uncomputed_items):
            self.statusbar.setValue(n)
            seconds += self.compute_item(item)
            self.update()
        self.statusbar.hide()
        self.setStyleSheet(original_style_sheet)
        self.titlelabel.setText(f"Computation on {len(uncomputed_items)} images Complete in {seconds:.2f} sec.!")
        self.setEnabled(True)

    def search_pattern(self, pattern=None, xy=None):
        #  if pattern is None and xy is None:
        if self._filter_viewer is None:
            print("No filter viewer available. Cannot search.", file=sys.stderr)
            return
        elif self._filter_viewer.pattern is None:
            print("No pattern selected. Cannot search.", file=sys.stderr)
            return
        pattern = np.array(self._filter_viewer.pattern)[:, None, None]
        self.searchbutton.setEnabled(False)
        self.titlelabel.setText(f"searching {tuple(pattern.astype(int).tolist())}")
        ready_images = [item for item in self.list_widget.path2img.values() if not item.requires_computation]
        self.statusbar.setRange(0, len(ready_images))
        count_frequencies = {}
        scores = {}
        for n, image in enumerate(ready_images):
            self.statusbar.setValue(n)
            match_counts, radii = image.get_similarity_map(xy, pattern)
            count_frequencies[image.path] = np.unique(match_counts, return_counts=True)
            scores[image.path] = ((match_counts / radii)**2).mean()
        if len(scores) == 0:
            self.titlelabel.setText("No images to search.")
        else:
            winner = sorted([(v, k) for k, v in scores.items()])[-1][1]
            self.titlelabel.setText(f"Searching complete! Winner: {winner} score: {scores[winner]}")
        self.searchbutton.setEnabled(True)
        if len(scores) > 0:
            self.draw_counts(count_frequencies, radii, pattern.squeeze().astype(int).tolist())

    def draw_counts(self, count_frequencies, best, pattern):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        total_matches = 0
        total_pixels = 0
        by_perf = [(f[c == best].sum()/f.sum(), n) for n, (c, f) in count_frequencies.items()]
        by_perf = [n for _, n in reversed(sorted(by_perf))]
        for path in by_perf:
            counts, freqs = count_frequencies[path]
            if best in counts:
                total_matches += freqs[counts == best].sum()
                print(f"{freqs[counts == best]} matches in {path}")
            total_pixels += freqs.sum()
            relative_freqs = freqs / freqs.sum()
            label = f"{path.stem} {freqs[counts == best].sum()}"
            ax.semilogy((counts / best) * 100, relative_freqs, label=label)
        plt.legend()
        percentage = 100 * (total_matches / total_pixels)
        title_str = f"Pattern: {pattern} \nPerfect Matches: # {total_matches} ,  % {percentage:.6f}"
        ax.set_title(title_str)
        ax.set_xticks(np.linspace(0, 100, best + 1))
        ax.set_xlabel("Radii Agrement %")
        ax.set_ylabel("Pixel #")
        plt.show()

    def get_similar_occurences(self, pattern, max_distance=4, max_results=100):
        per_image = {}
        images = {}
        for img in self.images:
            per_image[img.path], radii = img.get_similar_occurences(pattern, max_distance=max_distance)
            images[img.path] = img
        results = []
        for distance in range(max_distance):
            if len(results) > max_results:
                break
            for img_path, img in per_image.items():
                for n in range(len(per_image[img_path][distance])):
                    results.append((distance, images[img_path], per_image[img_path][distance][n].tolist()))
        return results[:max_results], radii
