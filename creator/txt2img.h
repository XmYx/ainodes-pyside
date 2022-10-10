#ifndef TXT2IMG_H
#define TXT2IMG_H

#include <QWidget>

namespace Ui {
class txt2img;
}

class txt2img : public QWidget
{
    Q_OBJECT

public:
    explicit txt2img(QWidget *parent = nullptr);
    ~txt2img();

private:
    Ui::txt2img *ui;
};

#endif // TXT2IMG_H
