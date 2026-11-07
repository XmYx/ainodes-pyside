import os
import re

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtGui import QColor, QPalette  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from frontend.style import theme  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_themes_define_the_same_tokens():
    dark, light = theme.THEMES["dark"], theme.THEMES["light"]
    assert dark.keys() == light.keys()
    for tokens in (dark, light):
        for value in tokens.values():
            assert QColor(value).isValid(), value


def contrast(a, b):
    def luminance(color):
        channels = []
        for c in (color.redF(), color.greenF(), color.blueF()):
            channels.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]
    hi, lo = sorted((luminance(QColor(a)), luminance(QColor(b))), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize("name", ["dark", "light"])
def test_text_is_readable(name):
    t = theme.THEMES[name]
    for background in ("window", "surface", "input", "raised"):
        assert contrast(t["text"], t[background]) >= 7, background
        assert contrast(t["text_muted"], t[background]) >= 4.5, background
    assert contrast(t["accent_text"], t["accent"]) >= 4.5  # selected menu items, Dream button
    assert contrast(t["accent"], t["window"]) >= 3  # focus rings, checkboxes, tab underline


def test_resolve_maps_choices(monkeypatch):
    monkeypatch.setattr(theme, "system_prefers_dark", lambda: True)
    assert theme.resolve("system") == "dark"
    assert theme.resolve(None) == "dark"
    monkeypatch.setattr(theme, "system_prefers_dark", lambda: False)
    assert theme.resolve("system") == "light"
    assert theme.resolve("LIGHT") == "light"
    assert theme.resolve("unknown") == "dark"


@pytest.mark.parametrize("name", ["dark", "light"])
def test_stylesheet_is_fully_formatted(name, app):
    tokens = theme.THEMES[name]
    icons = theme.write_icons(tokens, name)
    qss = theme.stylesheet(tokens, icons)
    assert "{" in qss and not re.search(r"\{[a-z_]+\}", qss)
    for url in re.findall(r"url\(([^)]+)\)", qss):
        assert os.path.exists(url), url
    assert tokens["accent"] in qss


def test_palette_uses_tokens(app):
    tokens = theme.THEMES["dark"]
    palette = theme.build_palette(tokens)
    assert palette.color(QPalette.Window).name() == tokens["window"]
    assert palette.color(QPalette.Highlight).name() == tokens["accent"]
    assert palette.color(QPalette.Disabled, QPalette.Text).name() == tokens["text_muted"]


def test_manager_applies_and_follows_system(app, monkeypatch):
    manager = theme.ThemeManager(app, "light")
    assert manager.apply() == "light"
    assert app.palette().color(QPalette.Window).name() == theme.THEMES["light"]["window"]
    assert theme.THEMES["light"]["accent"] in app.styleSheet()
    assert manager.token("paper") == theme.THEMES["light"]["paper"]

    monkeypatch.setattr(theme, "system_prefers_dark", lambda: True)
    manager.apply("system")
    assert manager.active == "dark"
    monkeypatch.setattr(theme, "system_prefers_dark", lambda: False)
    manager._system_changed()
    assert manager.active == "light"
    manager.apply("dark")
    manager._system_changed()  # an explicit choice ignores the OS
    assert manager.active == "dark"
