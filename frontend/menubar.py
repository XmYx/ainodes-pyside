"""Main window menu bar: file, edit, generation, canvas, view, tools and help actions.

Most actions call functions that already exist on the canvas, the
parameter panel or the main window; this module only gives them a menu,
shortcuts and checked states that stay in sync with the settings.
"""
import os
import platform

from PySide6.QtCore import QUrl
from PySide6.QtGui import QAction, QActionGroup, QDesktopServices, QKeySequence
from PySide6.QtWidgets import QApplication, QFileDialog, QLabel, QMessageBox

from backend.singleton import singleton as gs
from backend.sqlite import setting_db
from frontend.style import theme

VERSION = "1.0"
GITHUB_URL = "https://github.com/XmYx/ainodes-pyside"
DISCORD_URL = "https://discord.gg/XDQm9pk5pd"

# (label, key, canvas method, canvas.mode value). The keys are handled by the
# canvas itself, so the menu only shows them (after a tab) instead of
# registering them globally, which would steal those letters from the prompt box.
CANVAS_MODES = [
    ("Select", "B", "select_mode", "select"),
    ("Pan", "C", "drag_mode", "drag"),
    ("Move", "M", "move_mode", "move"),
    ("Outpaint", "N", "add_mode", "outpaint"),
    ("Inpaint", "", "inpaint_mode", "inpaint"),
]

SHORTCUT_HELP = [
    ("Ctrl+Enter", "Dream"),
    ("Ctrl+.", "Stop"),
    ("Ctrl+Shift+A", "Toggle advanced mode"),
    ("Ctrl+,", "Settings"),
    ("Ctrl+N / Ctrl+O / Ctrl+S", "New canvas / open image / save project"),
    ("Ctrl+E", "Export canvas as PNG"),
    ("Ctrl+Z / Del", "Undo / delete the selected rectangle"),
    ("F11", "Full screen"),
    ("B, C, M, N", "Canvas: select, pan, move, outpaint (while the canvas has focus)"),
]


def _persist():
    try:
        setting_db.save_settings()
    except Exception as e:  # settings DB problems must not break the UI
        print("could not save settings:", e)


class MainMenu:
    def __init__(self, window, theme_manager=None):
        self.window = window
        self.theme_manager = theme_manager or theme.ThemeManager(QApplication.instance(),
                                                                 getattr(gs.system, "theme", "system"))
        self.bar = window.menuBar()
        self.actions = {}
        self.backend_label = QLabel()
        window.statusBar().addPermanentWidget(self.backend_label)
        self._build_file()
        self._build_edit()
        self._build_generate()
        self._build_canvas()
        self._build_view()
        self._build_tools()
        self._build_help()
        self.sync()

    # helpers ---------------------------------------------------------------
    @property
    def canvas(self):
        return self.window.canvas.canvas

    @property
    def controls(self):
        return self.window.widgets[self.window.current_widget]

    def _action(self, menu, key, text, slot=None, shortcut=None, checkable=False, tip=None):
        action = QAction(text, self.window)
        if shortcut:
            if isinstance(shortcut, (list, tuple)):
                action.setShortcuts([QKeySequence(s) for s in shortcut])
            else:
                action.setShortcut(QKeySequence(shortcut))
        action.setCheckable(checkable)
        if tip:
            action.setStatusTip(tip)
        if slot is not None:
            action.triggered.connect(slot)
        menu.addAction(action)
        self.actions[key] = action
        return action

    def _docks(self):
        w = self.window
        return [
            ("Parameters", self.controls.w.dockWidget),
            ("Settings", w.system_setup.w.dockWidget),
            ("Model Download", w.model_download_ui.w.dockWidget),
            ("Image Lab", w.image_lab_ui.w.dockWidget),
            ("Prompt Fetcher", w.prompt_fetcher.w.dockWidget),
            ("Lexica Art", w.lexicart.w.dockWidget),
            ("Krea", w.krea.w.dockWidget),
            ("Outpaint History", w.thumbs.w.dockWidget),
        ]

    def show_dock(self, dock):
        dock.setVisible(True)
        dock.raise_()

    # menus -----------------------------------------------------------------
    def _build_file(self):
        menu = self.bar.addMenu("&File")
        self._action(menu, "new_canvas", "&New Canvas", self.new_canvas, QKeySequence.New)
        self._action(menu, "open_image", "&Open Image…", self.canvas.load_img_into_rect, QKeySequence.Open,
                     tip="Load an image into the selected canvas rectangle")
        menu.addSeparator()
        self._action(menu, "open_project", "Open &Project…", self.open_project, "Ctrl+Shift+O")
        self._action(menu, "save_project", "&Save Project…", self.save_project, QKeySequence.Save)
        self._action(menu, "export_canvas", "&Export Canvas as PNG…", self.canvas.save_canvas, "Ctrl+E")
        menu.addSeparator()
        self._action(menu, "open_output", "Open Output &Folder", self.open_output_folder)
        menu.addSeparator()
        self._action(menu, "quit", "&Quit", self.window.close, QKeySequence.Quit)

    def _build_edit(self):
        menu = self.bar.addMenu("&Edit")
        self._action(menu, "undo", "&Undo", self.undo, QKeySequence.Undo)
        self._action(menu, "delete_rect", "&Delete Selected Rectangle", self.canvas.delete_rect, QKeySequence.Delete)
        menu.addSeparator()
        self._action(menu, "settings", "Se&ttings…", self.open_settings, QKeySequence.Preferences)

    def _build_generate(self):
        menu = self.bar.addMenu("&Generate")
        menu.aboutToShow.connect(self.sync)
        self._action(menu, "dream", "&Dream", self.window.task_switcher, ["Ctrl+Return", "Ctrl+Enter"])
        self._action(menu, "stop", "&Stop", self.controls.stop_all, "Ctrl+.")
        menu.addSeparator()
        self._action(menu, "use_comfyui", "Generate with &ComfyUI", self.set_use_comfyui, checkable=True,
                     tip="Send txt2img jobs to the ComfyUI server from Settings")
        self.comfy_models = menu.addMenu("ComfyUI &Model")
        self.comfy_models.aboutToShow.connect(self.fill_comfy_models)
        self._action(menu, "comfyui_compile", "Compile Model (torch.compile)", self.set_comfyui_compile, checkable=True,
                     tip="Slower first image, faster after; needs a CUDA / ROCm GPU on the ComfyUI side")
        menu.addSeparator()
        native = menu.addMenu("&Native Model")
        group = QActionGroup(self.window)
        from backend import modern_models
        choices = [("", "Stable Diffusion 1.x / 2.x (model list in the panel)")] + \
            [(m.key, m.label + ("  (gated: needs a Hugging Face login)" if m.gated else "")) for m in modern_models.MODELS.values()]
        for key, label in choices:
            action = self._action(native, f"native_{key or 'sd'}", label,
                                  lambda checked=False, k=key: self.set_native_model(k), checkable=True)
            group.addAction(action)
        native.addSeparator()
        self._action(native, "native_compile", "Compile Model (torch.compile)", self.set_native_compile, checkable=True)
        self._action(menu, "load_model", "&Load Selected Model", self.window.deforum_ui.deforum_six.load_model_from_config)
        self._action(menu, "free_memory", "&Free GPU Memory", self.free_memory)

    def _build_canvas(self):
        menu = self.bar.addMenu("&Canvas")
        menu.aboutToShow.connect(self.sync)
        group = QActionGroup(self.window)
        for label, key, method, _ in CANVAS_MODES:
            text = f"{label} Mode\t{key}" if key else f"{label} Mode"
            action = self._action(menu, f"mode_{method}", text, getattr(self.canvas, method), checkable=True)
            group.addAction(action)
        menu.addSeparator()
        self._action(menu, "reset_canvas", "&Reset Canvas", self.new_canvas)

    def _build_view(self):
        menu = self.bar.addMenu("&View")
        menu.aboutToShow.connect(self.sync)
        self._action(menu, "advanced", "&Advanced Mode", self.window.set_advanced_mode, "Ctrl+Shift+A",
                     checkable=True, tip="Show every option and panel")
        theme_menu = menu.addMenu("&Theme")
        group = QActionGroup(self.window)
        for choice in theme.THEME_CHOICES:
            action = self._action(theme_menu, f"theme_{choice}", choice.capitalize(),
                                  lambda checked=False, c=choice: self.set_theme(c), checkable=True)
            group.addAction(action)
        panels = menu.addMenu("&Panels")
        for title, dock in self._docks():
            action = dock.toggleViewAction()
            action.setText(title)
            panels.addAction(action)
        menu.addSeparator()
        self._action(menu, "fullscreen", "&Full Screen", self.toggle_fullscreen, "F11", checkable=True)

    def _build_tools(self):
        menu = self.bar.addMenu("&Tools")
        for title, dock in self._docks()[2:7]:
            self._action(menu, f"tool_{title}", title, lambda checked=False, d=dock: self.show_dock(d))

    def _build_help(self):
        menu = self.bar.addMenu("&Help")
        self._action(menu, "shortcuts", "&Keyboard Shortcuts", self.show_shortcuts)
        self._action(menu, "docs", "&Documentation Folder", lambda: self.open_path("documentation"))
        menu.addSeparator()
        self._action(menu, "github", "Project on &GitHub", lambda: QDesktopServices.openUrl(QUrl(GITHUB_URL)))
        self._action(menu, "issue", "&Report an Issue", lambda: QDesktopServices.openUrl(QUrl(GITHUB_URL + "/issues")))
        self._action(menu, "discord", "&Discord", lambda: QDesktopServices.openUrl(QUrl(DISCORD_URL)))
        menu.addSeparator()
        self._action(menu, "about", "&About aiNodes", self.show_about)

    # state -----------------------------------------------------------------
    def sync(self):
        """Refresh checked states from the current settings."""
        self.actions["advanced"].setChecked(not self.window.default_hidden)
        self.actions["use_comfyui"].setChecked(bool(getattr(gs.system, "use_comfyui", False)))
        self.actions["comfyui_compile"].setChecked(bool(getattr(gs.system, "comfyui_compile", False)))
        self.actions["native_compile"].setChecked(bool(getattr(gs.system, "native_compile", False)))
        native = getattr(gs.system, "native_model", "") or ""
        if f"native_{native or 'sd'}" in self.actions:
            self.actions[f"native_{native or 'sd'}"].setChecked(True)
        self.actions[f"theme_{self.theme_manager.choice}"].setChecked(True)
        self.actions["fullscreen"].setChecked(self.window.isFullScreen())
        mode = getattr(self.canvas, "mode", None)
        for _, _, method, mode_name in CANVAS_MODES:
            if mode == mode_name:
                self.actions[f"mode_{method}"].setChecked(True)
        if getattr(gs.system, "use_comfyui", False):
            model = getattr(gs.system, "comfyui_checkpoint", "") or "first available model"
            self.backend_label.setText(f"Backend: ComfyUI · {model}")
        elif getattr(gs.system, "native_model", ""):
            from backend import modern_models
            spec = modern_models.MODELS.get(gs.system.native_model)
            self.backend_label.setText(f"Backend: local · {spec.label if spec else gs.system.native_model}")
        else:
            self.backend_label.setText("Backend: local models")
        if getattr(self.window, "model_card", None) is not None:
            self.window.model_card.sync()

    # actions ---------------------------------------------------------------
    def set_theme(self, choice):
        gs.system.theme = choice
        self.theme_manager.apply(choice)
        self.window.apply_canvas_theme()
        _persist()
        self.sync()

    def set_use_comfyui(self, enabled):
        gs.system.use_comfyui = bool(enabled)
        self.window.system_setup.w.use_comfyui.setChecked(bool(enabled))
        _persist()
        self.sync()

    def fill_comfy_models(self):
        """List the image models the ComfyUI server has, grouped by family."""
        from backend.comfyui.client import ComfyUIClient
        from backend.comfyui import families
        self.comfy_models.clear()
        try:
            models = families.generation_models(families.available_models(
                ComfyUIClient(getattr(gs.system, "comfyui_url", None), timeout=3)))
        except Exception as e:
            self.comfy_models.addAction(f"ComfyUI not reachable: {e}").setEnabled(False)
            return
        current = getattr(gs.system, "comfyui_checkpoint", "")
        group = QActionGroup(self.comfy_models)
        for name, label in models:
            action = QAction(f"{name}\t{label}", self.comfy_models, checkable=True, checked=name == current)
            action.triggered.connect(lambda checked=False, n=name: self.set_comfy_model(n))
            group.addAction(action)
            self.comfy_models.addAction(action)
        if not models:
            self.comfy_models.addAction("No image models on the server").setEnabled(False)

    def set_comfy_model(self, name):
        gs.system.comfyui_checkpoint = name
        self.window.system_setup.w.comfyui_checkpoint.setText(name)
        _persist()
        self.sync()

    def set_native_model(self, key):
        gs.system.native_model = key
        _persist()
        self.sync()

    def set_native_compile(self, enabled):
        gs.system.native_compile = bool(enabled)
        _persist()

    def set_comfyui_compile(self, enabled):
        gs.system.comfyui_compile = bool(enabled)
        _persist()

    def new_canvas(self):
        self.canvas.reset()

    def undo(self):
        self.canvas.undoEvent()
        self.canvas.redraw()

    def free_memory(self):
        from backend.shared import model_killer
        model_killer()
        from backend.comfyui.client import ComfyUIClient
        comfy = ComfyUIClient(getattr(gs.system, "comfyui_url", None), timeout=3).free_memory()
        self.window.set_status_bar("GPU memory released" + (" (local models and ComfyUI)" if comfy else ""))

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self.window, "Open Canvas Project", "", "aiNodes canvas (*.json)")
        if path:
            self.canvas.load_rects_from_json(path)

    def save_project(self):
        path, _ = QFileDialog.getSaveFileName(self.window, "Save Canvas Project", "", "aiNodes canvas (*.json)")
        if path:
            if not path.endswith(".json"):
                path += ".json"
            self.canvas.save_rects_as_json(path)
            self.window.set_status_bar(f"Saved {path}")

    def open_settings(self):
        self.show_dock(self.window.system_setup.w.dockWidget)

    def open_path(self, path):
        path = os.path.abspath(path)
        os.makedirs(path, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def open_output_folder(self):
        self.open_path(getattr(gs.system, "txt2img_out_dir", "output"))

    def toggle_fullscreen(self, enabled):
        if enabled:
            self.window.showFullScreen()
        else:
            self.window.showNormal()

    def show_shortcuts(self):
        rows = "".join(f"<tr><td><b>{keys}</b></td><td style='padding-left:16px'>{what}</td></tr>"
                       for keys, what in SHORTCUT_HELP)
        QMessageBox.information(self.window, "Keyboard Shortcuts", f"<table>{rows}</table>")

    def about_text(self):
        import PySide6
        try:
            import torch
            torch_info = f"PyTorch {torch.__version__}" + (
                f", CUDA {torch.version.cuda} on {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ", CPU")
        except Exception:
            torch_info = "PyTorch not available"
        try:
            from backend.hardware import describe
            hardware = "<br>".join(describe())
        except Exception as e:
            hardware = f"hardware detection failed: {e}"
        return (f"<h3>aiNodes {VERSION}</h3><p>Stable Diffusion desktop with Deforum, outpainting, "
                f"upscalers and a ComfyUI backend.</p>"
                f"<p>Python {platform.python_version()} · Qt {PySide6.__version__}<br>{torch_info}</p>"
                f"<p>{hardware}</p>"
                f"<p><a href='{GITHUB_URL}'>{GITHUB_URL}</a></p>")

    def show_about(self):
        QMessageBox.about(self.window, "About aiNodes", self.about_text())
