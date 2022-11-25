# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # choose GPU if you are on a multi GPU server
import clip
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from backend.singleton import singleton
gs = singleton
#####  This script will predict the aesthetic score for this image file:

img_path = "test.jpg"





# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def load_aestetics_prediction_model():
    if 'apm' not in gs.models:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

        s = torch.load("src/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo

        model.load_state_dict(s)

        model.to(device)
        model.eval()
        gs.models['apm'] = model
        del model
        gs.models['clip'], gs.prediction = clip.load("ViT-L/14", device=device)


def get_aestetics_score(img_path):
    load_aestetics_prediction_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #model2, preprocess = clip.load("ViT-L/14", device=device)  #RN50x64

    pil_image = Image.open(img_path)

    image = gs.prediction(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = gs.models['clip'].encode_image(image)

    im_emb_arr = normalized(image_features.cpu().detach().numpy() )

    prediction = gs.models['apm'](torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))

    return prediction
