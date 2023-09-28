import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

##############################
#       TRANSFORMERS         #
##############################
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


##############################
#   DATA LOADING FUNCTIONS   #
##############################

def load_dataset(data_dir: str, input_size: int, mode='trainval'):
    # force correct size on dataset images and normalization based on ImageNet
    val_transforms = transforms.Compose([
        Rescale(input_size),
        RandomPad(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if mode=='trainval':
        # Dataset separated in train/val (using for training)
        # train: data augmentation on top of val transforms
        data_transforms = {
            'train': transforms.Compose([
                Rescale(random.randint(int(input_size*1.2), int(input_size*2))),
                transforms.RandomCrop(input_size, pad_if_needed=True),
                transforms.RandomRotation(degrees=(-5,5)),
                transforms.RandomPerspective(distortion_scale=0.2),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': val_transforms,
        }
        # ImageFolder automatically converts images to RGB
        image_dataset= {x: datasets.ImageFolder(os.path.join(data_dir, x),
                            data_transforms[x]) for x in ['train', 'val']}
    else:
        # Single folder dataset (used for testing)
        image_dataset = datasets.ImageFolder(data_dir, val_transforms)
    return image_dataset, transforms


def get_dataloader(dataset, batch_size: int):
    """Load data in memory by batch

    Args:
        dataset (torchvision.datasets): transformed images
        batch_size (int): number of images in one batch

    Returns:
        torchvision.utils.data.DataLoader: batches of data ready for model input
    """
    if type(dataset) == dict:
        shuffle = {'train': True, 'val': False}
        dataloader = {x: DataLoader(dataset[x], batch_size=batch_size,
                                    shuffle=shuffle[x], num_workers=4) for x in ['train', 'val']}
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4)
    return dataloader


def get_classes(dataset) -> list:
    if type(dataset) == dict:
        return dataset['train'].classes
    else:
        return dataset.classes


def get_size(dataset):
    if type(dataset) == dict:
        return {x: len(dataset[x]) for x in ['train', 'val']}
    else:
        return len(dataset)


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


class SingleFolderDataSet(Dataset):
    # Dataset where there is no subfolder for classes
    # (can be useful for testing batch of images)
    def __init__(self, main_dir, transform):
        self.transform = transform
        all_imgs = [os.path.join(main_dir, x) for x in os.listdir(main_dir)]
        self.imgs = sorted(all_imgs)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_loc = self.imgs[idx]
        # ImageFolder automatically converts images to RGB
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image


def load_singlefolder_dataset(folder: str, input_size: int):
    data_transforms = transforms.Compose([
        Rescale(input_size),
        RandomPad(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize based on ImageNet
    ])
    image_dataset = SingleFolderDataSet(folder, data_transforms)
    return image_dataset