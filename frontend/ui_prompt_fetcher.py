import json
import urllib

import requests
from PySide6 import QtUiTools
from PySide6.QtCore import Signal, QObject, Slot, QFile
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFileDialog
from backend.guess_prompt import get_prompt_guess


class Callbacks(QObject):
    run_ai_prompt = Signal()
    run_img_to_prompt = Signal()
    get_lexica_prompts = Signal()
    got_lexica_prompts = Signal(str)
    get_krea_prompts = Signal()
    got_krea_prompts = Signal(str)
    got_image_to_prompt = Signal(str)


class FetchPrompts(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui/prompt_fetcher.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()

class PromptFetcher_UI:
    def __init__(self, parent):
        self.prompt_image = None
        self.parent = parent
        self.prompt_fetcher = parent.prompt_fetcher
        self.signals = Callbacks()
        self.connections()

    def connections(self):
        self.prompt_fetcher.w.getPrompts.clicked.connect(self.run_get_lexica_prompts)
        self.prompt_fetcher.w.getKreaPrompts.clicked.connect(self.run_get_krea_prompts)
        self.prompt_fetcher.w.aiPrompt.clicked.connect(self.run_ai_prompt)
        self.prompt_fetcher.w.usePrompt.clicked.connect(self.use_prompt)
        self.prompt_fetcher.w.dreamPrompt.clicked.connect(self.dream_prompt)
        self.prompt_fetcher.w.img2prompt.clicked.connect(self.image_to_prompt)

    @Slot()
    def dream_prompt(self):
        prompt = self.prompt_fetcher.w.output.textCursor().selectedText()
        if prompt == '':
            prompt = 'No prompt selected, please select the prompt you want to use'

        self.parent.widgets[self.parent.current_widget].w.prompts.setPlainText(prompt.replace(u'\u2029\u2029', '\n'))
        self.parent.task_switcher()

    @Slot()
    def use_prompt(self):
        prompt = self.prompt_fetcher.w.output.textCursor().selectedText()
        self.parent.widgets[self.parent.current_widget].w.prompts.setPlainText(prompt.replace(u'\u2029\u2029', '\n'))

    @Slot()
    def image_to_prompt_thread(self):
        self.parent.run_as_thread(self.get_img_to_prompt)

    @Slot()
    def set_lexica_prompts(self, prompts):
        self.prompt_fetcher.w.output.setPlainText(prompts)

    @Slot()
    def get_lexica_prompts_thread(self):
        self.parent.run_as_thread(self.get_lexica_prompts)

    @Slot()
    def get_krea_prompts_thread(self):
        self.parent.run_as_thread(self.get_krea_prompts)

    @Slot()
    def run_get_lexica_prompts(self):
        self.signals.get_lexica_prompts.emit()

    @Slot()
    def run_get_krea_prompts(self):
        self.signals.get_krea_prompts.emit()

    @Slot()
    def run_ai_prompt(self):
        self.signals.run_ai_prompt.emit()

    def image_to_prompt(self):
        filename = list(QFileDialog.getOpenFileName(caption='Load Input Image', filter='Images (*.png *.jpg)'))
        self.prompt_image = filename[0]
        self.signals.run_img_to_prompt.emit()

    def get_img_to_prompt(self, progress_callback=False):
        if self.prompt_image is not None:
            prompt = get_prompt_guess(self.prompt_image)
            self.signals.got_image_to_prompt.emit(prompt)

    @Slot()
    def set_img_to_prompt_text(self, image_filename):
        self.prompt_fetcher.w.output.setPlainText(str(image_filename))

    @Slot()
    def set_krea_prompts(self, prompts):
        if prompts == '':
            prompts = 'Nothing was found'
        self.prompt_fetcher.w.output.setPlainText(prompts)

    def set_ai_prompt(self, txt):
        self.prompt_fetcher.w.output.setPlainText(txt)
        self.parent.signals.setStatusBar.emit("Ai Prompt finished...")

    def get_lexica_prompts(self, progress_callback=False):
        out_text = ''
        prompts_txt = self.prompt_fetcher.w.input.toPlainText()
        prompts_array = prompts_txt.split('\n')
        for prompt in prompts_array:
            prompt = urllib.parse.quote_plus(prompt)
            response = requests.get("https://lexica.art/api/v1/search?q=" + prompt)
            res = response.text
            res = json.loads(res)
            if 'images' in res:
                for image in res['images']:
                    out_text = out_text + str(image['prompt']) + '\n\n'
        self.signals.got_lexica_prompts.emit(out_text)

    def get_krea_prompts(self, progress_callback=False):
        out_text = ''
        prompts_txt = self.prompt_fetcher.w.input.toPlainText()
        prompts_array = prompts_txt.split('\n')
        for prompt in prompts_array:
            prompt = urllib.parse.quote_plus(prompt)
            response = requests.get("https://devapi.krea.ai/prompts/?format=json&search=" + prompt)
            res = response.text
            res = json.loads(res)
            if 'results' in res:
                for image in res['results']:
                    out_text = out_text + str(image['prompt']) + '\n\n'
        self.signals.got_krea_prompts.emit(out_text)
