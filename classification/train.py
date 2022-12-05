import time
from datetime import datetime
import os
import shutil
import glob
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as Model
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from prepare_data import load_datasets, get_dataloaders, get_classes, get_sizes


NUM_EPOCHS = 30
BATCH_SIZE = 128
MODEL_NAME = 'EffB7'

NETS = {
    'EffB0': {'input_size': 224, 'model': Model.efficientnet_b0},
    'EffB1': {'input_size': 240, 'model': Model.efficientnet_b1},
    'EffB2': {'input_size': 288, 'model': Model.efficientnet_b2},
    'EffB3': {'input_size': 300, 'model': Model.efficientnet_b3},
    'EffB4': {'input_size': 380, 'model': Model.efficientnet_b4},
    'EffB5': {'input_size': 456, 'model': Model.efficientnet_b5},
    'EffB6': {'input_size': 528, 'model': Model.efficientnet_b6},
    'EffB7': {'input_size': 600, 'model': Model.efficientnet_b7},
    'Res18': {'input_size': 224, 'model': Model.resnet18},
    'Dense169': {'input_size': 224, 'model': Model.densenet169},
    'Dense201': {'input_size': 224, 'model': Model.densenet201}
    }

MODEL_TORCH = NETS[MODEL_NAME]['model']
INPUT_SIZE = NETS[MODEL_NAME]['input_size']

datasets = load_datasets('/workspace/data', input_size=INPUT_SIZE)
classes = get_classes(datasets)
dataset_sizes = get_sizes(datasets)
dataloaders = get_dataloaders(datasets, batch_size=BATCH_SIZE)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def build_model(model: Model) -> Model:
    """Setup a model for new training:
        freeze first layers, adapt last layer to number of classes

    Args:
        model (Model): torchvision model to setup

    Returns:
        Model: model ready for training
    """
    # freeze first layers
    for param in model.parameters():
        # for name, param in model.named_parameters():
        # if ("bn"not in name):
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    if 'Eff' in MODEL_NAME:
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(classes))
    elif 'Dense' in MODEL_NAME:
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, len(classes))
    elif 'Res' in MODEL_NAME:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(classes))
    else:
        raise ValueError(f'Unknown model {MODEL_NAME}')

    model = model.to(device)
    return model


def train_model(model: Model,
                criterion,
                optimizer,
                scheduler,
                checkpoint: str = None) -> Model:
    """Main function for training model

    Args:
        model (Model): torchvision model (ex: AlexNet, VGG, ...)
        criterion: loss function
        optimizer: optimization function
        scheduler: learning rate function
        checkpoint (str, optional): path to checkpoints if training to resume.
                                    Defaults to None.

    Returns:
        Model: trained model
    """

    since = time.time()

    if checkpoint: # resuming training
        assert os.path.splitext(checkpoint)[1] == '.pth'
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        sch_dict = checkpoint['scheduler_state_dict']
        if 'total_steps' in sch_dict.keys(): # for OneCycleLR
            sch_dict['total_steps'] = NUM_EPOCHS * len(dataloaders['train'])
        scheduler.load_state_dict(sch_dict)
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        training_name = os.path.splitext(os.path.basename(checkpoint))[0]
    else: # starting from scratch
        start_epoch = 1
        best_acc = 0.0
        training_name = MODEL_NAME + '_' + datetime.now().isoformat("_", "hours")

    writer = SummaryWriter(f"models/{training_name}") # logs for tensorboard
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print('Epoch {}/{}'.format(epoch, NUM_EPOCHS))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to inference mode

            running_loss = 0.0
            running_corrects = 0

            # iterate over data
            for batch, labels in tqdm(dataloaders[phase]):
                batch = batch.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(batch)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step() # OneCycleLR is updated after each batch

                # statistics
                running_loss += loss.item() * batch.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'Epoch {epoch}/{NUM_EPOCHS} {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            # log in tensorboard
            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)

            # deep copy and save the model when better than before
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                os.makedirs("models", exist_ok=True)
                torch.save({
                    'model_state_dict': best_model_wts,
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_acc
                    }, f'models/{training_name}/{training_name}.pth')

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, training_name


def test_model(model: Model, training_name: str) -> None:
    """ Computes metrics to evaluate model
        Prints: accuracy, precision, recall
        Writes: confusion matrix

    Args:
        model (Model): torch model to evaluate
        training_name (str): id l'entraînement
    """
    # Set model to inference mode
    model.eval()

    # Initialize the prediction and label lists(tensors)
    predlist=torch.zeros(0,dtype=torch.long, device='cpu')
    lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
    outputlist=torch.zeros(0,dtype=torch.long, device='cpu')

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Append batch prediction results
            predlist=torch.cat([predlist,preds.view(-1).cpu()])
            lbllist=torch.cat([lbllist,labels.view(-1).cpu()])
            outputlist=torch.cat([outputlist,torch.nn.functional.softmax(outputs, dim=1).cpu().detach()])

    # Confusion matrix
    y_test, y_pred = lbllist.numpy(), predlist.numpy()
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    df_cm = pd.DataFrame(cm, index = [i for i in classes], columns = [i for i in classes])
    df_cm.to_csv(f'models/{training_name}/confusion-matrix.csv') # visualize with sn heatmap

    # Other scores
    acc = accuracy_score(y_test, y_pred, normalize=True)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    with open(f'models/{training_name}/details.txt', 'w') as outfile:
        outfile.write(f'Accuracy = {round(acc, 3)}\n')
        outfile.write(f'Precision = {round(prec, 3)}\n')
        outfile.write(f'Recall = {round(rec, 3)}\n')

    # Details of predictions probabilities
    probas = outputlist.numpy().transpose() # each line is the probas for this class
    all_lines = {'filename': [x[0] for x in datasets['val'].imgs],
                'label': [classes[x[1]] for x in datasets['val'].imgs],
                'max_pred': [classes[x] for x in y_pred]}
    for i in range(len(classes)):
        all_lines[classes[i]] = probas[i]
    df_prob = pd.DataFrame(all_lines)
    df_prob.to_csv(f'models/{training_name}/probas_val.csv', index=False)


def get_model_params():
    if 'Eff' in MODEL_NAME:
        classif_layer = model.classifier[1]
    elif 'Dense' in MODEL_NAME:
        classif_layer = model.classifier
    elif 'Res' in MODEL_NAME:
        classif_layer = model.fc
    else:
        raise ValueError(f'Unknown model {MODEL_NAME}')
    return classif_layer.parameters()


def delete_last_training():
    list_of_folders = glob.glob('models/*')
    latest = max(list_of_folders, key=os.path.getctime)
    res = input(f"Are you ok to delete {latest} ?")
    if res in ["yes", "y", "ok"]:
        shutil.rmtree(latest)
    else:
        print('Did not delete.')


if __name__=="__main__":
    print('Classes ', classes)

    # FROM START: add pretrained=True, RESUME: no parameter
    model = build_model(MODEL_TORCH(pretrained=True))

    # define parameters for training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(get_model_params(), lr=0.001)
    lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005,
                        steps_per_epoch=len(dataloaders['train']), epochs=NUM_EPOCHS)

    # train model (WHEN RESUMING: add checkpoint path)
    model, training_name = train_model(model, criterion, optimizer, lr_scheduler, checkpoint=None)
    # evaluate
    test_model(model, training_name)

    with open(f'models/{training_name}/details.txt', 'a') as outfile:
        outfile.write('\n')
        outfile.write(f'Loss = {criterion.__class__.__name__}\n')
        outfile.write(f'Optimizer = {optimizer.__class__.__name__} lr=0.001\n')
        outfile.write(f'Lr = {lr_scheduler.__class__.__name__}\n')
        outfile.write(f'Epochs = {NUM_EPOCHS}\n')
        outfile.write(f'Base = {MODEL_NAME}\n')
        outfile.write(f'Batch size = {BATCH_SIZE}\n')
        outfile.write('Trained layers = last\n')