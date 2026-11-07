"""The model card at the top of the parameter panel.

It keeps the choices that decide what an image looks like in plain sight,
in simple and advanced mode alike: where images are generated (local
Stable Diffusion, a local modern model, or a ComfyUI server), which model,
and the size, steps, CFG and seed. Size, steps, CFG and seed are the same
values as the fields in the advanced panel; both stay in sync.
"""
import random

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QSizePolicy, QSpinBox, QToolButton, QVBoxLayout)

from backend.singleton import singleton as gs

LOCAL_SD, LOCAL_MODERN, COMFYUI = "sd", "modern", "comfyui"
BACKENDS = [(LOCAL_SD, "Local · Stable Diffusion"), (LOCAL_MODERN, "Local · FLUX / Krea"), (COMFYUI, "ComfyUI server")]
SIZES = [(512, 512), (768, 768), (1024, 1024), (1152, 896), (896, 1152), (1344, 768), (768, 1344)]


def current_backend():
    if getattr(gs.system, "use_comfyui", False):
        return COMFYUI
    if getattr(gs.system, "native_model", ""):
        return LOCAL_MODERN
    return LOCAL_SD


def size_label(w, h):
    shape = "square" if w == h else ("landscape" if w > h else "portrait")
    return f"{w} × {h}  {shape}"


class ModelCard(QFrame):
    def __init__(self, window, controls):
        super().__init__()
        self.window = window
        self.ui = controls  # the UniControl widget with selected_model, W, H, steps, scale, seed
        self._updating = False
        self._comfy_available = None
        self._accelerator = None
        self.setObjectName("modelCard")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        title = QLabel("MODEL"); title.setObjectName("cardTitle")
        self.hw_chip = QLabel(); self.hw_chip.setObjectName("hwChip")
        header = QHBoxLayout(); header.addWidget(title); header.addStretch(1); header.addWidget(self.hw_chip)

        self.backend = QComboBox(); self.backend.setObjectName("backendPicker")
        for key, label in BACKENDS:
            self.backend.addItem(label, key)
        self.backend.setToolTip("Where images are generated")
        self.model = QComboBox(); self.model.setObjectName("modelPicker")
        self.model.setMinimumContentsLength(14)
        self.model.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.refresh = QToolButton(); self.refresh.setObjectName("cardTool"); self.refresh.setText("⟳")
        self.refresh.setToolTip("Reload the model list")
        pick = QHBoxLayout(); pick.addWidget(self.backend, 2); pick.addWidget(self.model, 3); pick.addWidget(self.refresh)

        self.hint = QLabel(); self.hint.setObjectName("cardHint"); self.hint.setWordWrap(True)

        self.size = QComboBox(); self.size.setObjectName("sizePicker")
        self.steps = QSpinBox(); self.steps.setRange(1, 200)
        self.cfg = QDoubleSpinBox(); self.cfg.setRange(0.0, 30.0); self.cfg.setSingleStep(0.5); self.cfg.setDecimals(1)
        self.seed = QLineEdit(); self.seed.setPlaceholderText("random")
        self.dice = QToolButton(); self.dice.setObjectName("cardTool"); self.dice.setText("Random")
        self.dice.setToolTip("Random seed (empty = new one every time)")
        grid = QGridLayout(); grid.setHorizontalSpacing(8); grid.setVerticalSpacing(4)
        for col, (text, widget) in enumerate([("Size", self.size), ("Steps", self.steps), ("CFG", self.cfg)]):
            label = QLabel(text); label.setObjectName("cardHint")
            grid.addWidget(label, 0, col); grid.addWidget(widget, 1, col)
        seed_label = QLabel("Seed"); seed_label.setObjectName("cardHint")
        seed_row = QHBoxLayout(); seed_row.setSpacing(4); seed_row.addWidget(self.seed); seed_row.addWidget(self.dice)
        grid.addWidget(seed_label, 0, 3); grid.addLayout(seed_row, 1, 3)
        grid.setColumnStretch(0, 3); grid.setColumnStretch(1, 1); grid.setColumnStretch(2, 1); grid.setColumnStretch(3, 2)

        layout = QVBoxLayout(self); layout.setContentsMargins(10, 8, 10, 10); layout.setSpacing(6)
        layout.addLayout(header); layout.addLayout(pick); layout.addWidget(self.hint); layout.addLayout(grid)

        self.backend.activated.connect(self._backend_chosen)
        self.model.activated.connect(self._model_chosen)
        self.refresh.clicked.connect(self.reload_models)
        self.size.activated.connect(self._size_chosen)
        self.steps.valueChanged.connect(lambda v: self._push(self.ui.steps.setValue, v))
        self.cfg.valueChanged.connect(lambda v: self._push(self.ui.scale.setValue, v))
        self.seed.textEdited.connect(lambda t: self._push(self.ui.seed.setText, t))
        self.dice.clicked.connect(self._random_seed)
        for widget in (self.ui.W, self.ui.H, self.ui.steps, self.ui.scale):
            widget.valueChanged.connect(self.sync_params)
        self.ui.seed.textChanged.connect(self.sync_params)
        self.ui.selected_model.currentIndexChanged.connect(lambda *_: self.sync())

        self.hw_timer = QTimer(self); self.hw_timer.timeout.connect(self.update_hardware); self.hw_timer.start(5000)
        self.update_hardware()
        self.sync()

    # keeping in step with settings and the advanced fields --------------------------------
    def _push(self, setter, value):
        if not self._updating:
            setter(value)

    def sync(self):
        """Show the backend and model the settings point at."""
        if self._updating:
            return
        self._updating = True
        try:
            backend = current_backend()
            self.backend.setCurrentIndex(self.backend.findData(backend))
            self._fill_models(backend)
            self._update_hint(backend)
        finally:
            self._updating = False
        self.sync_params()

    def sync_params(self, *args):
        if self._updating:
            return
        self._updating = True
        try:
            w, h = self.ui.W.value(), self.ui.H.value()
            self.size.clear()
            sizes = SIZES if (w, h) in SIZES else SIZES + [(w, h)]
            for sw, sh in sizes:
                self.size.addItem(size_label(sw, sh), f"{sw}x{sh}")
            self.size.setCurrentIndex(sizes.index((w, h)))
            self.steps.setValue(self.ui.steps.value())
            self.cfg.setValue(self.ui.scale.value())
            if self.seed.text() != self.ui.seed.text():
                self.seed.setText(self.ui.seed.text())
        finally:
            self._updating = False

    def _fill_models(self, backend):
        self.model.clear()
        if backend == LOCAL_SD:
            for i in range(self.ui.selected_model.count()):
                self.model.addItem(self.ui.selected_model.itemText(i), self.ui.selected_model.itemText(i))
            self.model.setCurrentIndex(self.ui.selected_model.currentIndex())
            if not self.model.count():
                self.model.addItem("no checkpoints in data/models", "")
        elif backend == LOCAL_MODERN:
            from backend import modern_models
            for spec in modern_models.MODELS.values():
                self.model.addItem(spec.label + ("  🔒" if spec.gated else ""), spec.key)
            self.model.setCurrentIndex(max(0, self.model.findData(gs.system.native_model)))
        else:
            current = getattr(gs.system, "comfyui_checkpoint", "") or ""
            for name, label in self._comfy_models():
                self.model.addItem(f"{name}  ·  {label}", name)
            if current and self.model.findData(current) < 0:
                self.model.addItem(current, current)
            if not self.model.count():
                self.model.addItem("ComfyUI not reachable – check Settings", "")
            self.model.setCurrentIndex(max(0, self.model.findData(current)))

    def _comfy_models(self):
        from backend.comfyui import families
        available = self._comfy_parts()
        return families.generation_models(available) if available else []

    def _comfy_parts(self):
        if self._comfy_available is None:
            from backend.comfyui.client import ComfyUIClient
            from backend.comfyui import families
            try:
                self._comfy_available = families.available_models(
                    ComfyUIClient(getattr(gs.system, "comfyui_url", None), timeout=3))
            except Exception as e:
                print("ComfyUI model list:", e)
                self._comfy_available = {}
        return self._comfy_available

    def _update_hint(self, backend):
        warn = False
        if backend == LOCAL_SD:
            text = "Stable Diffusion 1.x / 2.x checkpoints from data/models · 512–768 px"
        elif backend == LOCAL_MODERN:
            from backend import modern_models
            spec = modern_models.MODELS.get(gs.system.native_model)
            if spec is None:
                text = "Pick a model"
            else:
                text = (f"{spec.steps} steps · CFG {spec.guidance:g} · 1024 px · ~{spec.size_bytes / 2**30:.0f} GB, "
                        "loaded to fit your free VRAM")
                if spec.gated:
                    text += " · needs a Hugging Face login and accepted license"
        else:
            from backend.comfyui import families
            name = getattr(gs.system, "comfyui_checkpoint", "") or ""
            available = self._comfy_parts()
            if not available:
                text, warn = f"No answer from {getattr(gs.system, 'comfyui_url', '')}", True
            elif not name:
                text = "First model on the server"
            else:
                family = families.detect_family(name)
                text = f"{family.label} · {family.steps} steps · CFG {family.cfg:g} · {family.native_size} px"
                missing = families.resolve_components(name, available).missing
                if missing:
                    text, warn = "Missing: " + "; ".join(missing), True
        self.hint.setText(text)
        self.hint.setObjectName("cardWarn" if warn else "cardHint")
        self.hint.style().unpolish(self.hint); self.hint.style().polish(self.hint)

    # user choices -----------------------------------------------------------------------
    def _backend_chosen(self, index):
        backend = self.backend.itemData(index)
        menu = self.window.menu
        if backend == COMFYUI:
            self._comfy_available = None
            menu.set_use_comfyui(True)
        else:
            if getattr(gs.system, "use_comfyui", False):
                menu.set_use_comfyui(False)
            if backend == LOCAL_MODERN and not gs.system.native_model:
                from backend import modern_models
                menu.set_native_model(next(k for k, m in modern_models.MODELS.items() if not m.gated))
            elif backend == LOCAL_SD and gs.system.native_model:
                menu.set_native_model("")
        self.sync()
        self.apply_recommended()

    def _model_chosen(self, index):
        key = self.model.itemData(index)
        if not key:
            return
        backend = self.backend.currentData()
        if backend == LOCAL_SD:
            self.ui.selected_model.setCurrentIndex(index)  # loads it through the panel's own handler
        elif backend == LOCAL_MODERN:
            self.window.menu.set_native_model(key)
        else:
            self.window.menu.set_comfy_model(key)
        self.sync()
        self.apply_recommended()

    def apply_recommended(self):
        """Steps, CFG and size the chosen model was made for (distilled models need their own)."""
        backend = current_backend()
        if backend == LOCAL_MODERN:
            from backend import modern_models
            spec = modern_models.MODELS.get(gs.system.native_model)
            if spec:
                self._apply(spec.steps, spec.guidance, 1024)
        elif backend == COMFYUI and getattr(gs.system, "comfyui_checkpoint", ""):
            from backend.comfyui import families
            family = families.detect_family(gs.system.comfyui_checkpoint)
            self._apply(family.steps, family.cfg, family.native_size)
        elif self.ui.W.value() > 768 or self.ui.H.value() > 768:
            self._apply(None, None, 512)

    def _apply(self, steps, cfg, side):
        if steps is not None:
            self.ui.steps.setValue(steps)
            self.ui.scale.setValue(cfg)
        if max(self.ui.W.value(), self.ui.H.value()) != side:
            self.ui.W.setValue(side)
            self.ui.H.setValue(side)

    def reload_models(self):
        self._comfy_available = None
        if self.backend.currentData() == LOCAL_SD:
            self.ui.update_models.click()
        self.sync()

    def _size_chosen(self, index):
        w, h = map(int, self.size.itemData(index).split("x"))
        self.ui.W.setValue(w)
        self.ui.H.setValue(h)

    def _random_seed(self):
        self.ui.seed.setText(str(random.randint(0, 2**32 - 1)))

    def update_hardware(self):
        if self.hw_chip.text() and not self.isVisible():
            return
        try:
            from backend import hardware
            if self._accelerator is None:
                self._accelerator = hardware.detect().primary
                self.hw_chip.setToolTip("\n".join(hardware.describe()))
            acc = self._accelerator
            free, total = hardware.memory_info(acc)
            name = acc.name.replace("NVIDIA ", "").replace("GeForce ", "")
            self.hw_chip.setText(f"{name} · {free / 2**30:.1f} / {total / 2**30:.0f} GB free")
        except Exception as e:
            self.hw_chip.setText("hardware unknown")
            self.hw_chip.setToolTip(str(e))
