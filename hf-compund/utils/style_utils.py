import torch
import torchvision
from torch import nn


def get_vgg_encoder():
    vgg_net = torchvision.models.vgg16(pretrained=True)
    vgg_encoder = nn.Sequential(*list(vgg_net.features))
    return vgg_encoder

vgg_style_layers, vgg_content_layers = [0, 5, 10, 19, 28], [25]



def compute_gram(x):
    #x [batch_size,channels,h,w]
    x = x.reshape(x.shape[0],x.shape[1],-1) #[batch_size,channels,h*w]
    gram = x@x.transpose(1,2) #[batch_size,channels,channels]
    return gram/(x.shape[2])
def style_distance(x1,x2):
    #return [batch_size]
    gram_diff = (compute_gram(x1)-compute_gram(x2))**2
    return torch.mean(gram_diff,dim=(-1,-2))
# print(style_distance(style_features[0],style_features[0]))
