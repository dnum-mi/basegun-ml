
import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as Model
import matplotlib.pyplot as plt
from torchvision import transforms
from prepare_data import ConvertRgb, Rescale, RandomPad


def build_model(model: Model) -> Model:
    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model.classifier[1].in_features
    # to try later : add batch normalization and dropout
    model.classifier[1] = nn.Linear(num_ftrs, len(CLASSES))
    model = model.to(device)
    return model


trained_model = build_model(MODEL_TORCH())
# Initialize model with the pretrained weights
trained_model.load_state_dict(
    torch.load('models/B5_2022-02-09_13/B5_2022-02-09_13.pth', map_location=device)['model_state_dict']
)

class EfficientNet(nn.Module):
    def __init__(self, model):
        super(EfficientNet, self).__init__()
        
        # get the pretrained VGG19 network
        self.net = model
        
        # disect the network to access its last convolutional layer
        self.features_conv = self.net.features
        
        # get the last pool of the features stem
        self.pool = self.net.avgpool
        
        # get the classifier of the vgg19
        self.classifier = self.net.classifier
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)


model = EfficientNet(trained_model)
model.eval()


def gradcam_heatmap(in_tensor: torch.Tensor, pred_index: int) -> np.array:
    preds = model(in_tensor)

    if pred_index is None:
        pred_index = preds.argmax(dim=1)
    else:
        assert pred_index < len(CLASSES)
    print(f"GradCam for class {CLASSES[pred_index]}, ranked {1+list(torch.argsort(preds, dim=1, descending=True)[0]).index(pred_index)}/{len(CLASSES)}")
        
    # get the gradient of the output with respect to the parameters of the model
    preds[:, 3].backward()

    # pull the gradients out of the model
    gradients = model.get_activations_gradient()

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # get the activations of the last convolutional layer
    activations = model.get_activations(in_tensor).detach()

    # weight the channels by corresponding gradients
    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()

    # relu on top of the heatmap
    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap, 0)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)
    # show with plt.matshow(heatmap)
    return np.uint8(255 * heatmap.numpy())


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

loader =  transforms.Compose([
            ConvertRgb(),
            Rescale(456),
            RandomPad(456),
            transforms.ToTensor(),
            transforms.Normalize(mean= mean, std= std)
        ])


def load_efficientnet_image(path: str) -> torch.Tensor:
    im = Image.open(path)
    image = loader(im).float()
    image = image.unsqueeze(0)
    return image

inv_normalize = transforms.Normalize(
   mean= [-m/s for m, s in zip(mean, std)],
   std= [1/s for s in std]
)

def get_image_from_efficientnet_tensor(in_tensor: torch.Tensor) -> np.array:
    inv_tensor = inv_normalize(in_tensor)
    img = inv_tensor.squeeze(0).permute(1,2,0)
    return np.uint8(255 * img.numpy())


def show_heatmap_on_image(heatmap: np.array, img: np.array) -> None:
    import cv2
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)[:,:,::-1] # cv2 is in BGR instead of RGB
    superimposed_img = np.uint8(np.clip(heatmap * 0.4 + img, 0, 255))
    plt.imshow(superimposed_img)