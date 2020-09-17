"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl


class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
        """
        super(KeypointModel, self).__init__()
        self.hparams = hparams
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        ########################################################################


        self.model = nn.Sequential(
            # 32*92*92 -> 32*46*46
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),  
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.25),

            # 64*44*44 -> 64*22*22
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),  
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.25),

            # 128*22*22 -> 128*11*11
            nn.Conv2d(64, 128, 1),
            nn.ReLU(),  
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.25),

            nn.Flatten(),
            nn.Linear(128*11*11, self.hparams["n_hidden"]),
            nn.BatchNorm1d(self.hparams["n_hidden"],self.hparams["n_hidden"]),
            nn.ReLU(),
            nn.Dropout(p=0.25),        
            nn.Linear(self.hparams["n_hidden"], 30),
            
            # nn.Conv2d(1, 32, 5),
            # nn.ReLU(),  
            # nn.MaxPool2d(4, 4),
            # nn.Flatten(),
            # nn.Linear(32*23*23, self.hparams["n_hidden"]),
            # nn.BatchNorm1d(self.hparams["n_hidden"],self.hparams["n_hidden"]),
            # nn.ReLU(),
            # # nn.Dropout(p=0.25),        
            # nn.Linear(self.hparams["n_hidden"], 30),
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints                                    #
        ########################################################################

        # x.shape = [batch_size, 3, 32, 32] -> flatten the image first
        # x = x.view(x.shape[0], -1)

        # feed x into model!
        x = self.model(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x
    
    def configure_optimizers(self):

        optim = torch.optim.Adam(self.model.parameters(), lr=self.hparams["learning_rate"], weight_decay=self.hparams["reg"])

        return optim

    def training_step(self, batch, batch_idx):
        criterion = torch.nn.MSELoss()
        image, keypoints = batch["image"], batch["keypoints"]
        predicted_keypoints = self.forward(image).view(-1,15,2)
        loss = criterion(keypoints,predicted_keypoints)
        return {'loss': loss}

    # def validation_step(self, batch, batch_idx):
    #     criterion = torch.nn.MSELoss()
    #     image, keypoints = batch["image"], batch["keypoints"]
    #     predicted_keypoints = self.forward(image).view(-1,15,2)
    #     loss = criterion(keypoints,predicted_keypoints)
    #     return {'loss': loss}

    # def validation_end(self, outputs):
    #     # average over all batches aggregated during one epoch
    #     avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
    #     total_correct = torch.stack([x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
    #     return avg_loss

    # def validation_end(self, outputs):
    #     avg_loss, acc = self.general_end(outputs, "val")
    #     #print("Val-Acc={}".format(acc))
    #     tensorboard_logs = {'val_loss': avg_loss}
    #     return {'val_loss': avg_loss, 'val_acc': acc, 'log': tensorboard_logs}


class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
