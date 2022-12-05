import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class ConvertRgb(object):
    """Converts an image to RGB
    """

    def __init__(self):
        pass

    def __call__(self, image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image


class Rescale(object):
    """Rescale the image in a sample to a given size while keeping ratio

    Args:
        output_size (int): Desired output size. The largest of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, image):
        w, h = image.size
        if w > h:
            new_h, new_w = self.output_size * h / w, self.output_size
        else:
            new_h, new_w = self.output_size, self.output_size * w / h
        new_h, new_w = int(new_h), int(new_w)
        return transforms.functional.resize(image, (new_h, new_w))


class RandomPad(object):
    """Pad an image to reach a given size

    Args:
        output_size (int): Desired output size. We pad all edges
                        symmetrically to reach a size x size image.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, image):
        w, h = image.size
        pads = {'horiz': [self.output_size - w,0,0],
        'vert': [self.output_size - h,0,0]}
        if pads['horiz'][0] >= 0 and pads['vert'][0] >= 0:
            for direction in ['horiz', 'vert'] :
                pads[direction][1] = pads[direction][0] // 2
                if pads[direction][0] % 2 == 1: # if the size to pad is odd, add a random +1 on one side
                    pads[direction][1] += np.random.randint(0,1)
                pads[direction][2] = pads[direction][0] - pads[direction][1]

            return transforms.functional.pad(image,
                [pads['horiz'][1], pads['vert'][1], pads['horiz'][2], pads['vert'][2]],
                fill = int(np.random.choice([0, 255])) # border randomly white or black
            )
        else:
            return image


def load_datasets(data_dir: str, input_size: int):
    # Data augmentation and normalization for training,
    # just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            Rescale(input_size),
            RandomPad(input_size),
            transforms.RandomRotation(degrees=(-5,5)),
            transforms.RandomPerspective(distortion_scale=0.2),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize based on ImageNet
        ]),
        'val': transforms.Compose([
            Rescale(input_size),
            RandomPad(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                        data_transforms[x]) for x in ['train', 'val']}
    return image_datasets


def get_dataloaders(dataset, batch_size: int):
    """Load data in memory by batch

    Args:
        dataset (torchvision.datasets): transformed images
        batch_size (int): number of images in one batch

    Returns:
        Dict[torchvision.utils.data.DataLoader]:
    """
    shuffle = {'train': True, 'val': False}
    dataloaders = {x: DataLoader(dataset[x], batch_size=batch_size,
                                shuffle=shuffle[x], num_workers=4) for x in ['train', 'val']}
    return dataloaders


def get_classes(dataset) -> list:
    return dataset['train'].classes


def get_sizes(dataset):
    return {x: len(dataset[x]) for x in ['train', 'val']}


def show_batch():
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    # Get a batch of training data
    inputs, classids = next(iter(dataloaders['train']))
    # Make a grid from batch
    out = make_grid(inputs)
    # Show result
    inp = out.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(20,20))
    plt.imshow(inp)
    plt.title([CLASSES[x] for x in classids])
    plt.pause(0.001)