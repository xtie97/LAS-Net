import torch 
import numpy as np
import monai
from lasnet import LASNet
#from modules import SwinTransformer

def get_network(img_size, in_channels, n_class, use_checkpoint=True):
    # get the multiplication of img_size
    model = LASNet(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=n_class,
            feature_size=48,
            num_heads=[3, 6, 12, 24],
            spatial_dims=3,
            deep_supr_num=3,
            use_checkpoint=use_checkpoint,
            )
    return model

if __name__ == "__main__":
    # get the multiplication of img_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = [112, 112, 112]
    in_channels = 2
    n_class = 2
    model = get_network(img_size, in_channels, n_class).to(device)
    
    # print number of parameters
    print(f"Number of params in the model is {sum([np.prod(p.size()) for p in model.parameters()])}")

    # create a dummy input
    input = torch.randn(1, 4, *img_size).to(device)
    # run the model
    out, out_ref = model(input)
    for i in out:
        print(i.size())

    for i in out_ref:
        print(i.size())

    # save the model as .pt
    torch.save(model.state_dict(), "model.pt")
    '''
    SwinTransformer:
    torch.Size([1, 32, 56, 56, 56])
    torch.Size([1, 64, 28, 28, 28])
    torch.Size([1, 128, 14, 14, 14])
    torch.Size([1, 256, 7, 7, 7])
    torch.Size([1, 512, 4, 4, 4])
    torch.Size([1, 32, 56, 56, 56])
    torch.Size([1, 64, 28, 28, 28])
    torch.Size([1, 128, 14, 14, 14])
    torch.Size([1, 256, 7, 7, 7])
    torch.Size([1, 512, 4, 4, 4])
    '''

