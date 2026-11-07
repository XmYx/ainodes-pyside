"""Light and dark themes for aiNodes.

A theme is a small set of color tokens. apply_theme() turns them into a
QPalette for the Fusion style (so natively drawn parts follow the theme)
and a single stylesheet for the details Fusion leaves plain. "system"
follows the OS light/dark preference and switches along with it.
"""
import os

from PySide6.QtCore import QStandardPaths, Qt
from PySide6.QtGui import QColor, QGuiApplication, QPalette

THEMES = {
    "dark": {
        "window": "#1c1d21",
        "surface": "#24262b",
        "raised": "#2d3036",
        "hover": "#363a41",
        "pressed": "#3f444c",
        "input": "#17181b",
        "border": "#383b42",
        "border_strong": "#50555e",
        "text": "#e7e8eb",
        "text_muted": "#9ba1aa",
        "accent": "#7360ee",
        "accent_hover": "#8273f1",
        "accent_text": "#ffffff",
        "danger": "#ef6b63",
        "selection": "#3d3770",
        "canvas": "#121316",
        "paper": "#2b2d33",
    },
    "light": {
        "window": "#eef0f3",
        "surface": "#f8f9fb",
        "raised": "#ffffff",
        "hover": "#eceef2",
        "pressed": "#e1e4e9",
        "input": "#ffffff",
        "border": "#d3d7dd",
        "border_strong": "#aeb4bd",
        "text": "#1b1e23",
        "text_muted": "#5c636d",
        "accent": "#5b45e0",
        "accent_hover": "#4b36cc",
        "accent_text": "#ffffff",
        "danger": "#c8382f",
        "selection": "#dcd6fb",
        "canvas": "#dfe2e7",
        "paper": "#ffffff",
    },
}
THEME_CHOICES = ("system", "light", "dark")

_ICONS = {
    "chevron-down": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><path d="M2 3.5l3 3 3-3" '
                    'fill="none" stroke="{c}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    "chevron-up": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><path d="M2 6.5l3-3 3 3" '
                  'fill="none" stroke="{c}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    "check": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 12 12"><path d="M2.5 6.2l2.4 2.4 4.6-5" '
             'fill="none" stroke="{c}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    "dot": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 12 12"><circle cx="6" cy="6" r="2.6" fill="{c}"/></svg>',
}

STYLESHEET = """
QMainWindow, QDialog {{ background: {window}; }}
QMainWindow::separator {{ background: {border}; width: 1px; height: 1px; }}
QToolTip {{ background: {surface}; color: {text}; border: 1px solid {border}; padding: 4px 6px; }}

QDockWidget {{ color: {text}; }}
QDockWidget::title {{ background: {window}; padding: 6px 8px; border-bottom: 1px solid {border}; }}
QDockWidget > QWidget {{ background: {surface}; }}

QTabBar::tab {{ background: transparent; color: {text_muted}; padding: 6px 12px; border: none;
               border-top: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {text}; background: {surface}; border-top: 2px solid {accent}; }}
QTabBar::tab:hover:!selected {{ color: {text}; background: {hover}; }}
QTabWidget::pane {{ border: 1px solid {border}; border-radius: 6px; top: -1px; }}

QPushButton, QToolButton {{ background: {raised}; color: {text}; border: 1px solid {border};
                            border-radius: 6px; padding: 5px 12px; }}
QPushButton:hover, QToolButton:hover {{ background: {hover}; border-color: {border_strong}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {pressed}; }}
QPushButton:checked {{ background: {selection}; border-color: {accent}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {text_muted}; background: {surface}; border-color: {border}; }}
QPushButton#dream {{ background: {accent}; color: {accent_text}; border: 1px solid {accent};
                     font-weight: 600; font-size: 14px; padding: 8px 18px; }}
QPushButton#dream:hover {{ background: {accent_hover}; border-color: {accent_hover}; }}
QPushButton#dream:disabled {{ background: {pressed}; color: {text_muted}; border-color: {border}; }}
QPushButton#stop_dream {{ color: {danger}; }}
QPushButton#stop_dream:hover {{ border-color: {danger}; }}

QFrame#modelCard {{ background: {raised}; border: 1px solid {border}; border-radius: 8px; }}
QFrame#modelCard QLabel {{ background: transparent; }}
QLabel#cardTitle {{ color: {text_muted}; font-size: 11px; font-weight: 600; letter-spacing: 1px; }}
QLabel#cardHint {{ color: {text_muted}; font-size: 11px; }}
QLabel#cardWarn {{ color: {danger}; font-size: 11px; }}
QLabel#hwChip {{ color: {text_muted}; background: {surface}; border: 1px solid {border}; border-radius: 9px;
                 padding: 1px 8px; font-size: 11px; }}
QComboBox#modelPicker {{ font-weight: 600; }}
QToolButton#cardTool {{ border: 1px solid {border}; border-radius: 5px; padding: 2px 6px; background: {surface}; }}
QToolButton#cardTool:hover {{ background: {hover}; }}

QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox, QComboBox {{
    background: {input}; color: {text}; border: 1px solid {border}; border-radius: 5px; padding: 3px 6px;
    selection-background-color: {accent}; selection-color: {accent_text}; }}
QLineEdit:hover, QTextEdit:hover, QPlainTextEdit:hover, QAbstractSpinBox:hover, QComboBox:hover {{
    border-color: {border_strong}; }}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QAbstractSpinBox:focus, QComboBox:focus {{
    border-color: {accent}; }}
QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled {{ color: {text_muted}; background: {surface}; }}
QAbstractSpinBox {{ padding-right: 18px; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ subcontrol-origin: border; width: 16px;
    border: none; background: transparent; }}
QAbstractSpinBox::up-button {{ subcontrol-position: top right; }}
QAbstractSpinBox::down-button {{ subcontrol-position: bottom right; }}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{ background: {hover}; }}
QAbstractSpinBox::up-arrow {{ image: url({icons}/chevron-up.svg); width: 9px; height: 9px; }}
QAbstractSpinBox::down-arrow {{ image: url({icons}/chevron-down.svg); width: 9px; height: 9px; }}
QComboBox {{ padding-right: 22px; }}
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right; width: 20px; border: none; }}
QComboBox::down-arrow {{ image: url({icons}/chevron-down.svg); width: 10px; height: 10px; }}
QComboBox QAbstractItemView {{ background: {surface}; color: {text}; border: 1px solid {border};
    selection-background-color: {accent}; selection-color: {accent_text}; outline: 0; }}

QCheckBox, QRadioButton {{ color: {text}; spacing: 7px; }}
QCheckBox:disabled, QRadioButton:disabled {{ color: {text_muted}; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 14px; height: 14px; border: 1px solid {border_strong};
    background: {input}; }}
QCheckBox::indicator {{ border-radius: 4px; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {accent}; }}
QCheckBox::indicator:checked {{ background: {accent}; border-color: {accent}; image: url({icons}/check-on.svg); }}
QRadioButton::indicator:checked {{ background: {accent}; border-color: {accent}; image: url({icons}/dot-on.svg); }}

QGroupBox {{ border: 1px solid {border}; border-radius: 6px; margin-top: 12px; padding-top: 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {text_muted}; }}

QMenuBar {{ background: {window}; color: {text}; border-bottom: 1px solid {border}; padding: 2px 4px; }}
QMenuBar::item {{ background: transparent; padding: 4px 10px; border-radius: 4px; }}
QMenuBar::item:selected {{ background: {hover}; }}
QMenu {{ background: {surface}; color: {text}; border: 1px solid {border}; padding: 4px; }}
QMenu::item {{ padding: 5px 28px 5px 26px; border-radius: 4px; }}
QMenu::item:selected {{ background: {accent}; color: {accent_text}; }}
QMenu::item:disabled {{ color: {text_muted}; }}
QMenu::separator {{ height: 1px; background: {border}; margin: 4px 8px; }}
QMenu::indicator {{ width: 14px; height: 14px; left: 6px; }}
QMenu::indicator:checked {{ image: url({icons}/check.svg); }}
QMenu::indicator:checked:selected {{ image: url({icons}/check-on.svg); }}

QStatusBar {{ background: {window}; color: {text_muted}; border-top: 1px solid {border}; }}
QStatusBar::item {{ border: none; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle {{ background: {border_strong}; border-radius: 4px; }}
QScrollBar::handle:vertical {{ min-height: 28px; margin: 2px; }}
QScrollBar::handle:horizontal {{ min-width: 28px; margin: 2px; }}
QScrollBar::handle:hover {{ background: {text_muted}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

QSlider::groove:horizontal {{ height: 4px; background: {border}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {accent}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {raised}; border: 2px solid {accent}; width: 12px; height: 12px;
    margin: -6px 0; border-radius: 8px; }}

QListView, QListWidget, QTreeView, QTableView {{ background: {input}; color: {text}; border: 1px solid {border};
    border-radius: 5px; alternate-background-color: {surface}; }}
QListView::item:selected, QTreeView::item:selected, QTableView::item:selected {{
    background: {selection}; color: {text}; }}
QHeaderView::section {{ background: {surface}; color: {text_muted}; border: none;
    border-bottom: 1px solid {border}; padding: 4px 6px; }}

QProgressBar {{ background: {input}; border: 1px solid {border}; border-radius: 5px; text-align: center;
    color: {text}; }}
QProgressBar::chunk {{ background: {accent}; border-radius: 4px; }}
QGraphicsView {{ background: {canvas}; border: none; }}
"""


def system_prefers_dark():
    hints = QGuiApplication.styleHints()
    scheme = getattr(hints, "colorScheme", None)
    return scheme is not None and scheme() == Qt.ColorScheme.Dark


def resolve(name):
    """Map a theme choice ("system", "light", "dark") to a concrete theme."""
    name = (name or "system").lower()
    if name == "system":
        return "dark" if system_prefers_dark() else "light"
    return name if name in THEMES else "dark"


def build_palette(tokens):
    palette = QPalette()
    roles = {
        QPalette.Window: "window", QPalette.WindowText: "text", QPalette.Base: "input",
        QPalette.AlternateBase: "surface", QPalette.Text: "text", QPalette.Button: "raised",
        QPalette.ButtonText: "text", QPalette.BrightText: "accent_text", QPalette.Highlight: "accent",
        QPalette.HighlightedText: "accent_text", QPalette.ToolTipBase: "surface", QPalette.ToolTipText: "text",
        QPalette.PlaceholderText: "text_muted", QPalette.Link: "accent", QPalette.Light: "hover",
        QPalette.Midlight: "border", QPalette.Mid: "border_strong", QPalette.Dark: "border_strong",
        QPalette.Shadow: "window",
    }
    for role, token in roles.items():
        palette.setColor(role, QColor(tokens[token]))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        palette.setColor(QPalette.Disabled, role, QColor(tokens["text_muted"]))
    return palette


def write_icons(tokens, theme_name):
    """Write the themed SVG glyphs the stylesheet references; returns their folder."""
    base = QStandardPaths.writableLocation(QStandardPaths.CacheLocation) or os.path.expanduser("~/.cache")
    folder = os.path.join(base, "ainodes-theme", theme_name)
    os.makedirs(folder, exist_ok=True)
    variants = {name: tokens["text"] for name in _ICONS}
    variants.update({f"{name}-on": tokens["accent_text"] for name in ("check", "dot")})
    for name, color in variants.items():
        svg = _ICONS[name.removesuffix("-on")].format(c=color)
        path = os.path.join(folder, f"{name}.svg")
        if not os.path.exists(path) or open(path).read() != svg:
            with open(path, "w") as f:
                f.write(svg)
    return folder.replace(os.sep, "/")


def stylesheet(tokens, icons):
    return STYLESHEET.format(icons=icons, **tokens)


class ThemeManager:
    """Applies a theme choice to the app and keeps "system" in sync with the OS."""

    def __init__(self, app, choice="system"):
        self.app = app
        self.choice = choice
        self.active = None
        hints = QGuiApplication.styleHints()
        if hasattr(hints, "colorSchemeChanged"):
            hints.colorSchemeChanged.connect(self._system_changed)

    def apply(self, choice=None):
        if choice is not None:
            self.choice = choice
        name = resolve(self.choice)
        tokens = THEMES[name]
        self.app.setStyle("Fusion")
        self.app.setPalette(build_palette(tokens))
        self.app.setStyleSheet(stylesheet(tokens, write_icons(tokens, name)))
        self.active = name
        return name

    def _system_changed(self, *args):
        if self.choice == "system":
            self.apply()

    def token(self, key):
        return THEMES[self.active or resolve(self.choice)][key]


def apply_theme(app, choice="system"):
    return ThemeManager(app, choice).apply()
