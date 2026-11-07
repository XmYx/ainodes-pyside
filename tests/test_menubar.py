"""Integration tests: build the real main window offscreen and drive the menu bar."""
import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def window():
    import backend.paths  # noqa: F401
    from PySide6.QtWidgets import QApplication
    from backend.sqlite import setting_db
    from frontend import mainwindow

    app = QApplication.instance() or QApplication([])
    saved = setting_db.save_settings
    first_run = mainwindow.MainWindow.check_first_run
    setting_db.save_settings = lambda *a, **k: None  # keep the user's settings DB untouched
    mainwindow.MainWindow.check_first_run = lambda self: None  # modal welcome dialog
    try:
        win = mainwindow.MainWindow()
        win.show()
        app.processEvents()
        yield win
    finally:
        setting_db.save_settings = saved
        mainwindow.MainWindow.check_first_run = first_run


def menu_titles(win):
    return [a.text().replace("&", "") for a in win.menuBar().actions()]


def test_menus_present(window):
    assert menu_titles(window) == ["File", "Edit", "Generate", "Canvas", "View", "Tools", "Help"]


def test_every_action_is_wired(window):
    for key, action in window.menu.actions.items():
        assert action.receivers("2triggered(bool)") > 0, key


def test_no_single_key_global_shortcuts(window):
    # Single letters must stay free for typing prompts; the canvas handles its own mode keys.
    for key, action in window.menu.actions.items():
        for sequence in action.shortcuts():
            text = sequence.toString()
            assert len(text) != 1, (key, text)


def test_dream_shortcut_covers_both_enter_keys(window):
    shortcuts = {s.toString() for s in window.menu.actions["dream"].shortcuts()}
    assert {"Ctrl+Return", "Ctrl+Enter"} <= shortcuts


def test_advanced_mode_round_trip(window):
    from backend.singleton import singleton as gs
    controls = window.widgets[window.current_widget].w
    action = window.menu.actions["advanced"]

    window.set_advanced_mode(True)
    assert not window.default_hidden
    assert action.isChecked() and controls.lucky.isChecked()
    assert controls.scrollArea.isVisible() and controls.negative_prompts.isVisible()
    assert gs.system.show_settings is True

    action.trigger()  # menu toggles back to simple mode
    assert window.default_hidden
    assert not action.isChecked() and not controls.lucky.isChecked()
    assert not controls.scrollArea.isVisible()
    assert gs.system.show_settings is False

    controls.lucky.click()  # the panel button does the same
    assert not window.default_hidden and action.isChecked()


@pytest.mark.parametrize("choice", ["dark", "light", "system"])
def test_theme_menu_switches_and_stores_choice(window, choice):
    from PySide6.QtWidgets import QApplication
    from backend.singleton import singleton as gs
    from frontend import ui_paint
    from frontend.style import theme

    window.menu.actions[f"theme_{choice}"].trigger()
    assert gs.system.theme == choice
    assert window.menu.actions[f"theme_{choice}"].isChecked()
    active = theme.resolve(choice)
    assert theme.THEMES[active]["accent"] in QApplication.instance().styleSheet()
    assert ui_paint.__backgroudColor__.name() == theme.THEMES[active]["paper"]


def test_comfyui_toggle_syncs_settings(window):
    from backend.singleton import singleton as gs
    action = window.menu.actions["use_comfyui"]
    before = bool(gs.system.use_comfyui)
    try:
        action.trigger()
        assert gs.system.use_comfyui is (not before)
        assert window.system_setup.w.use_comfyui.isChecked() is (not before)
        assert ("ComfyUI" in window.menu.backend_label.text()) is (not before)
    finally:
        window.menu.set_use_comfyui(before)


def test_canvas_mode_actions(window):
    for key, mode in [("mode_drag_mode", "drag"), ("mode_move_mode", "move"), ("mode_select_mode", "select")]:
        window.menu.actions[key].trigger()
        assert window.canvas.canvas.mode == mode
        window.menu.sync()
        assert window.menu.actions[key].isChecked()


def test_settings_and_tools_raise_their_panels(window):
    window.menu.actions["settings"].trigger()
    assert window.system_setup.w.dockWidget.isVisible()
    window.menu.actions["tool_Image Lab"].trigger()
    assert window.image_lab_ui.w.dockWidget.isVisible()


def test_project_save_and_load(window, tmp_path):
    canvas = window.canvas.canvas
    canvas.addrect_atpos(prompt="", x=0, y=0)
    path = tmp_path / "project.json"
    canvas.save_rects_as_json(str(path))
    saved = json.loads(path.read_text())
    assert saved and {"x", "y", "w", "h", "id"} <= saved[0].keys()
    canvas.load_rects_from_json(str(path))
    assert len(canvas.rectlist) == len(saved)


def test_about_mentions_versions(window):
    text = window.menu.about_text()
    assert "aiNodes" in text and "Qt" in text and "Python" in text


@pytest.fixture
def restore_backend():
    from backend.singleton import singleton as gs
    saved = {k: getattr(gs.system, k, None) for k in ("use_comfyui", "native_model", "comfyui_checkpoint")}
    yield gs
    for key, value in saved.items():
        setattr(gs.system, key, value)


def test_model_card_is_visible_in_simple_mode(window):
    window.set_advanced_mode(False)
    card = window.model_card
    assert card.isVisible() and card.backend.isVisible() and card.model.isVisible()
    assert card.size.isVisible() and card.steps.isVisible() and card.cfg.isVisible()
    assert "GB free" in card.hw_chip.text() or card.hw_chip.text() == "hardware unknown"


def test_model_card_params_follow_the_panel_both_ways(window):
    card, ui = window.model_card, window.widgets[window.current_widget].w
    ui.steps.setValue(33)
    assert card.steps.value() == 33
    card.cfg.setValue(4.5)
    assert ui.scale.value() == 4.5
    card._size_chosen(card.size.findData("1344x768"))
    assert (ui.W.value(), ui.H.value()) == (1344, 768)
    ui.W.setValue(640)  # not a preset: shown as an extra entry
    assert card.size.currentData() == "640x768"


def test_model_card_switches_backend_and_applies_model_defaults(window, restore_backend):
    gs, card, ui = restore_backend, window.model_card, window.widgets[window.current_widget].w
    card._backend_chosen(card.backend.findData("modern"))
    assert gs.system.native_model == "flux2-klein-4b" and not gs.system.use_comfyui
    assert (ui.steps.value(), ui.scale.value(), ui.W.value()) == (4, 1.0, 1024)
    assert "4 steps" in card.hint.text()

    card._comfy_available = {"checkpoints": ["sd_xl_base_1.0.safetensors"], "unets": ["krea2_turbo_fp8_scaled.safetensors"],
                             "clips": ["qwen3vl_4b_fp8_scaled.safetensors"], "clip_types": ["krea2"],
                             "vaes": ["qwen_image_vae.safetensors"]}
    gs.system.use_comfyui = True
    card.sync()
    card._model_chosen(card.model.findData("krea2_turbo_fp8_scaled.safetensors"))
    assert gs.system.comfyui_checkpoint == "krea2_turbo_fp8_scaled.safetensors"
    assert (ui.steps.value(), ui.scale.value()) == (8, 1.0) and "Krea" in card.hint.text()
    assert "ComfyUI" in window.menu.backend_label.text()

    card._backend_chosen(card.backend.findData("sd"))
    assert not gs.system.use_comfyui and gs.system.native_model == "" and ui.W.value() == 512
