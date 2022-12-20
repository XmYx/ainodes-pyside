import copy

from PySide6 import QtCore
from PySide6.QtCore import QRect, Slot, Qt, \
    QTimer, QEvent
from PySide6.QtGui import Qt, QColor, QFont
from PySide6.QtWidgets import QGraphicsProxyWidget
from backend.singleton import singleton
gs = singleton
__textColor__ = QColor(187, 187, 187)
__backgroudColor__ = QColor(60, 63, 65)
__font__ = QFont('Decorative', 10)
__idleColor__ = QColor(91, 48, 232)
__selColor__ = QColor(255, 102, 102)

class MyProxyWidget(QGraphicsProxyWidget):

    def __init__(self, widget, parent):
        super(MyProxyWidget, self).__init__()
        self.parent = parent

        self.widget = widget.w
        self.setWidget(self.widget)
        self.moving = False
        self.uid = copy.deepcopy(self.parent.uid)
        self.widget.loadbutton.clicked.connect(self.load_img)
        self.setup_widgets()
    def setup_widgets(self):
        print("init method called")

        self.rect_index = copy.deepcopy(len(self.parent.rectlist) - 1)
        rect = QRect(self.parent.rectlist[self.rect_index].x,self.parent.rectlist[self.rect_index].y,self.parent.rectlist[self.rect_index].w,self.parent.rectlist[self.rect_index].h,)
        rect.translate(512, 0)
        self.parent.krea.w.setGeometry(rect)
        self.parent.krea.w.show()


    def proxy_task(self):
        self.parent.parent.parent.widgets['unicontrol'].w.with_inpaint.setCheckState(Qt.CheckState.Unchecked)
        for i in gs.models:
            print(i)
        #print(self.parent.parent.widgets['unicontrol'].w.prompts.setText('11123'))
        text = self.widget.prompts.toPlainText()
        self.parent.parent.parent.widgets['unicontrol'].w.show_sample_per_step.setCheckState(Qt.CheckState.Checked)
        self.parent.selected_item = self.uid
        self.parent.render_item = self.uid
        #for i in self.parent.rectlist:
        #    if i.id == self.uid:
        #        self.parent.parent.parent.rect_index = self.parent.rectlist.index(i)
        self.parent.parent.parent.rect_index = self.rect_index

        self.parent.parent.parent.widgets['unicontrol'].w.prompts.setText(str(text))
        with_inpaint = self.parent.check_for_frame_overlap()
        i = self.parent.parent.parent.rect_index
        self.parent.rectlist[i].prompt = text
        if with_inpaint == True:
            #print("initiating inpaint")

            #self.parent.rectlist[i].w, self.parent.rectlist[i].h = map(lambda x: x - x % 64, (self.parent.rectlist[i].w, self.parent.rectlist[i].h))
            self.parent.parent.parent.widgets['unicontrol'].w.with_inpaint.setCheckState(Qt.CheckState.Checked)
            self.parent.w = self.parent.rectlist[i].w
            self.parent.h = self.parent.rectlist[i].h

            self.parent.change_rect_resolutions()
            self.parent.reusable_outpaint(self.uid)
            self.parent.parent.parent.sessionparams.params.with_inpaint = True
            #self.parent.parent.parent.update_ui_from_params()
            #self.parent.signals.outpaint_signal.emit()
        self.parent.parent.parent.task_switcher()
        #self.parent.parent.parent.update_ui_from_params()
        self.prompt_destroy()
    def prompt_destroy(self):
        self.prompt_destroy_action()
    def prompt_destroy_action(self):
        try:
            self.widget.w.delbutton.disconnect()
        except:
            pass
        #self.parent.parent.widgets[self.parent.parent.current_widget].w.prompts = self.backup
        self.animation = QtCore.QPropertyAnimation(self.fx, b"opacity")
        self.animation.setDuration(250)
        self.animation.setStartValue(1)
        self.animation.setEndValue(0)
        self.kreanimation = QtCore.QPropertyAnimation(self.parent.krea.w, b"windowOpacity")
        self.kreanimation.setDuration(250)
        self.kreanimation.setStartValue(self.parent.krea.w.windowOpacity())
        self.kreanimation.setEndValue(0)
        self.kreanimation.start()
        self.animation.start()
        QtCore.QTimer.singleShot(500, self.finish_prompt_destroy)
    def finish_prompt_destroy(self):
        self.parent.uid = self.uid
        self.parent.removeitem_by_uid_from_scene()
        #self.parent.scene.removeItem(self.proxy)
        self.proxy = None
        self.prompt = None
        self.parent.krea.w.hide()
        self.parent.scene.removeItem(self.parent.krea_proxy)
        #
        #self.widget.destroy()
        #self.parent.parent.delete_outpaint_frame()
    def prompt_destroy_with_frame(self):
        self.parent.selected_item = self.uid
        self.parent.parent.parent.delete_outpaint_frame()
        self.prompt_destroy_action()
    def load_img(self):
        self.parent.selected_item = self.uid
        self.parent.load_img_into_rect()
        self.prompt_destroy()
    @Slot(str)
    def got_krea_prompts(self, prompts):

        self.parent.scene.addItem(self.parent.krea_proxy)
        print(self.rect_index, self.parent.rectlist[0].x)

        rect = QRect(self.parent.rectlist[self.rect_index].x + self.parent.rectlist[self.rect_index].w,
                     self.parent.rectlist[self.rect_index].y,
                     self.parent.rectlist[self.rect_index].w * 2,
                     self.parent.rectlist[self.rect_index].h,)

        #rect.translate(self.parent.rectlist[self.rect_index].w, 0)
        self.parent.krea_proxy.setGeometry(rect)
        self.parent.krea.w.setGeometry(rect)

        #self.fx_2 = QGraphicsOpacityEffect()
        #self.parent.krea.w.setGraphicsEffect(self.fx_2)
        self.kreanimation = QtCore.QPropertyAnimation(self.parent.krea.w, b"windowOpacity")
        self.kreanimation.setDuration(125)
        self.kreanimation.setStartValue(self.parent.krea.w.windowOpacity())
        self.kreanimation.setEndValue(0)
        self.promptlist = prompts.split("\n\n")
        self.kreanimation.start()
        QTimer.singleShot(250, self.finish_got_krea_prompts)
    def finish_got_krea_prompts(self):
        x = 0
        #while x < 5:
        #    getattr(krea.w, f'krea_prompt_{x + 1}').setText(insert_newlines(promptlist[x], 64))
        #    x += 1
        try:
            self.parent.krea.w.krea_prompt_1.setText(insert_newlines(self.promptlist[0], 64))
            self.parent.krea.w.krea_prompt_1.setVisible(True)
            self.parent.krea.w.krea_prompt_2.setText(insert_newlines(self.promptlist[1], 64))
            self.parent.krea.w.krea_prompt_2.setVisible(True)

            self.parent.krea.w.krea_prompt_3.setText(insert_newlines(self.promptlist[2], 64))
            self.parent.krea.w.krea_prompt_3.setVisible(True)

            self.parent.krea.w.krea_prompt_4.setText(insert_newlines(self.promptlist[3], 64))
            self.parent.krea.w.krea_prompt_4.setVisible(True)

            self.parent.krea.w.krea_prompt_5.setText(insert_newlines(self.promptlist[4], 64))
            self.parent.krea.w.krea_prompt_5.setVisible(True)

        except:
            pass
        self.kreanimation = QtCore.QPropertyAnimation(self.parent.krea.w, b"windowOpacity")
        self.kreanimation.setDuration(125)
        self.kreanimation.setStartValue(self.parent.krea.w.windowOpacity())
        self.kreanimation.setEndValue(1)
        self.kreanimation.start()

    def keyPressEvent(self, e):
        super(MyProxyWidget, self).keyPressEvent(e)
        if e.key() == 32:
            if e.isAutoRepeat():
                pass
            else:
                self.parent.parent.parent.prompt_fetcher.w.input.setText(self.widget.prompts.toPlainText())
                self.parent.parent.parent.prompt_fetcher_ui.run_get_krea_prompts()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Enter:
            # Handle the QEvent::Enter event here
            self.parent.parent.parent.prompt_fetcher_ui.signals.got_krea_prompts.connect(self.got_krea_prompts)
            print("filtering")

        elif event.type() == QEvent.Leave:
            print("filter off")
            self.parent.parent.parent.prompt_fetcher_ui.signals.got_krea_prompts.disconnect(self.got_krea_prompts)


        # Return False to continue processing the event
        return False


def insert_newlines(string, every=64):
    lines = []
    for i in range(0, len(string), every):
        lines.append(string[i:i+every])
    return '\n'.join(lines)
