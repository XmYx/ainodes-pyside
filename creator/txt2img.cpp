#include "txt2img.h"
#include "ui_txt2img.h"

txt2img::txt2img(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::txt2img)
{
    ui->setupUi(this);
}

txt2img::~txt2img()
{
    delete ui;
}
