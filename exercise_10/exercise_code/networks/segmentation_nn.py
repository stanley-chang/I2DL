"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl

# class unetUp(nn.Module):
#     def __init__(self, in_size, out_size, is_deconv):
#         super(unetUp, self).__init__()
#         self.conv = unetConv2(in_size, out_size, False)
#         if is_deconv:
#             self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
#         else:
#             self.up = nn.UpsamplingBilinear2d(scale_factor=2)

#     def forward(self, inputs1, inputs2):
#         outputs2 = self.up(inputs2)
#         offset = outputs2.size()[2] - inputs1.size()[2]
#         padding = 2 * [offset // 2, offset // 2]
#         outputs1 = F.pad(inputs1, padding)
#         return self.conv(torch.cat([outputs1, outputs2], 1))

class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.hparams = hparams
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        # filters = [64, 128, 256, 512, 1024]
        filters = [32, 64, 128, 256, 512]
        self.model = nn.Sequential(
            # 64*236*236 -> 64*118*118
            nn.Conv2d(3, filters[0], 3,padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),  
            nn.MaxPool2d(2, 2),
            # nn.Dropout(p=0.25),

            # 128*116*116 -> 128*58*58
            nn.Conv2d(filters[0], filters[1], 3,padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(),  
            nn.MaxPool2d(2, 2),

            # 256*56*56 -> 256*28*28
            nn.Conv2d(filters[1], filters[2], 3,padding=1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(),  
            nn.MaxPool2d(2, 2),

            # 512*26*26 -> 512*13*13
            nn.Conv2d(filters[2], filters[3], 3,padding=1),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(),  
            nn.MaxPool2d(2, 2),

            # 1024*12*12 -> 1024*6*6
            nn.Conv2d(filters[3], filters[4], 3,padding=1),
            nn.BatchNorm2d(filters[4]),
            nn.ReLU(),  

            # ------------ start up-sampling ------------#

            # 1024*12*12 -> 512*11*11
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(filters[4], filters[3], 1),
            nn.ReLU(),

            # 512*22*22 -> 256*20*20
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(filters[3], filters[2], 1),
            nn.ReLU(),

            # 256*40*40 -> 128*38*38
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(filters[2], filters[1], 1),
            nn.ReLU(),

            # 128*76*76 -> 64*74*74
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(filters[1], filters[0], 1),
            nn.ReLU(),

            nn.Conv2d(filters[0], num_classes, 1)
        )



        # filters = [32, 64, 128, 256, 512]
        # self.model = nn.Sequential(
        #     # 32*236*236
        #     nn.Conv2d(3, filters[0], 5),
        #     nn.ReLU(),  
            
        #     # 32*236*236 -> 128*58*58
        #     nn.Conv2d(filters[0], filters[1], 3),
        #     nn.ReLU(),  
        #     nn.Linear(128*11*11, self.hparams["n_hidden"]),

        #     # 64*236*236 -> 64*118*118
        #     nn.Conv2d(3, filters[0], 5,padding=2),
        #     nn.ReLU(),  
            
        #     # 128*116*116 -> 128*58*58
        #     nn.Conv2d(filters[0], filters[1], 3,padding=1),
        #     nn.ReLU(),  
        #     nn.Linear(128*11*11, self.hparams["n_hidden"]),

        #     # 64*236*236 -> 64*118*118
        #     nn.Conv2d(3, filters[0], 5,padding=2),
        #     nn.ReLU(),  
            
        #     # 128*116*116 -> 128*58*58
        #     nn.Conv2d(filters[0], filters[1], 3,padding=1),
        #     nn.ReLU(),  
        #     nn.MaxPool2d(2, 2),

        #     # ------------ start up-sampling ------------#
            
        #     # 1024*12*12 -> 512*11*11
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        #     nn.Linear(128*11*11, self.hparams["n_hidden"]),
        #     nn.Conv2d(filters[4], filters[3], 3,padding=1),
        #     nn.ReLU(),

        #     # 512*22*22 -> 256*20*20
        #     nn.Conv2d(filters[3], filters[2], 3,padding=1),
        #     nn.ReLU(),
        #     nn.Linear(128*11*11, self.hparams["n_hidden"]),

        #     # 256*40*40 -> 128*38*38
        #     nn.Conv2d(filters[2], filters[1], 3,padding=1),
        #     nn.ReLU(),

        #     # 128*76*76 -> 64*74*74
        #     nn.Conv2d(filters[1], filters[0], 3,padding=1),
        #     nn.ReLU(),
        #     nn.Linear(128*11*11, self.hparams["n_hidden"]),

        #     nn.Conv2d(filters[2], filters[1], 3,padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(filters[2], filters[1], 3,padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(filters[0], num_classes, 1)
        # )

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        x = self.model(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
