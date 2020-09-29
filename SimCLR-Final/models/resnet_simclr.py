import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.customnet import CustomNet
from models.res import ResNet


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False),
                            "grayresnet":ResNet(),
                            "custom":CustomNet()}


        # base_model='resnet18'

        resnet = self._get_basemodel(base_model)
  


        if base_model=='custom':
            num_ftrs = resnet.layer.out_features
            self.features=resnet

           


        else: 
            num_ftrs = resnet.fc.in_features
            self.features = nn.Sequential(*list(resnet.children())[:-1])

        


    

        #This is special.. SimCLR have two hidden layers , win the inferece they use only the 1st hidden layer
        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):


        h = self.features(x)

        h = h.squeeze()


        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)

    
        return h, x
