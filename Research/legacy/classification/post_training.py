import os
from PIL import Image
from torch import Tensor
from seaborn import heatmap
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.onnx
import torchvision.models as Model
from torchvision import transforms
import onnx
import onnxruntime
from prepare_data import ConvertRgb, Rescale, RandomPad


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

CLASSES = ['autre_epaule', 'autre_pistolet', 'epaule_a_levier_sous_garde',
        'epaule_a_percussion_silex', 'epaule_a_pompe', 'epaule_a_un_coup', 'epaule_a_verrou',
        'pistolet_a_percussion_silex', 'pistolet_semi_auto_moderne', 'revolver']


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

loader =  transforms.Compose([
            ConvertRgb(),
            Rescale(INPUT_SIZE),
            RandomPad(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def build_model(model: Model) -> Model:
    # freeze first layers
    for param in model.parameters():
        param.requires_grad = False
    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model.classifier[1].in_features
    # to try later : add batch normalization and dropout
    model.classifier[1] = torch.nn.Linear(num_ftrs, len(CLASSES))
    model = model.to(device)
    return model


def load_model_inference(state_dict_path: str) -> Model:
    model = build_model(MODEL_TORCH())
    # Initialize model with the pretrained weights
    model.load_state_dict(torch.load(state_dict_path, map_location=device)['model_state_dict'])
    model.to(device)
    # set the model to inference mode
    model.eval()
    return model


def test_image(model, path):
    im = Image.open(path)
    image = loader(im).float()
    image = image.unsqueeze(0).to(device)
    output = model(image)
    probs = torch.nn.functional.softmax(output, dim=1).detach().numpy()[0]
    res = [(CLASSES[i], round(probs[i]*100,2)) for i in range(len(CLASSES))]
    res.sort(key=lambda x:x[1], reverse=True)
    # display(im.resize((300,int(300*im.size[1]/im.size[0])))) # display image in notebook
    return res


def show_confusion_matrix(matrix_path: str):
    df = pd.read_csv(matrix_path, index_col=0)
    fig, ax = plt.subplots(figsize=(10,8))
    heatmap(df, annot=True)


def show_images_of_label_predicted_as(in_df: pd.DataFrame, true_label: str, pred_label: str, limit=20):
    df = in_df[(in_df.label==true_label) & (in_df.max_pred==pred_label)]
    df = df.sort_values(by=pred_label, ascending=False)
    df = df.reset_index()
    print(len(df), 'images found')

    columns = 3
    plt.figure(figsize=(20,25*(limit//columns)))
    for index, row in df.head(limit).iterrows():
        path = os.path.join('/workdir/data/val', row['label'], row['filename'])
        im = Image.open(path)
        plt.subplot(len(df) / columns + 1, columns, index + 1).set_title(round(row[pred_label],3))
        plt.axis('off')
        plt.imshow(np.asarray(im))


def convert_to_onnx(state_dict_path: str):
    if not torch.cuda.is_available():
        print('EfficientNet needs GPU computing for onnx export !')
        return

    print('Loading model...')
    model = load_model_inference(state_dict_path)

    print('Saving model to ONNX ...')
    onnx_path = f'{os.path.splitext(state_dict_path)[0]}.onnx'

    in_tensor = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, requires_grad=True)
    in_tensor = in_tensor.to(device)
    pth_out = model(in_tensor) # compute result of pth model inference on tensor

    # export the model
    torch.onnx.export(model,               # model being run
                    in_tensor,                         # model input (or a tuple for multiple inputs)
                    onnx_path,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input and output names
                    output_names = ['output'])

    # checks that the onnx model has a valid structure
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # check the result of export is same
    ort_session = onnxruntime.InferenceSession(onnx_path)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: in_tensor.detach().cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(pth_out.detach().cpu().numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")



def roc_from_probas_file(csvpath: str):
    df = pd.read_csv(csvpath)

    fpr = {}
    tpr = {}
    thresh ={}
    y_test = list(df['label'])
    label_to_index = {CLASSES[i]:i for i in range(len(CLASSES))}
    y_test = [label_to_index[x] for x in y_test]

    score = roc_auc_score(y_test, df[[*CLASSES]].to_numpy(), multi_class='ovr')
    print(f'AUC score one vs all: {score}\n')


    COLORS = [col[4:] for col in mcolors.TABLEAU_COLORS]
    # plotting
    plt.figure(figsize=(15,10))
    for i in range(len(CLASSES)):
        fpr, tpr, thresh = roc_curve(y_test, df[CLASSES[i]].to_numpy(), pos_label=i)
        plt.plot(fpr, tpr, linestyle='--',color=COLORS[i], label=f'Class {CLASSES[i]} vs Rest')
        # calculate the g-mean for each threshold
        gmeans = np.sqrt(tpr * (1-fpr))
        # locate the index of the largest g-mean
        ix = np.argmax(gmeans) # G-Mean={gmeans[ix]:.3f}
        print(f'Best threshold={thresh[ix]:.4f} for class {CLASSES[i]}')
        plt.scatter(fpr[ix], tpr[ix], marker='o', color=COLORS[i])

    plt.title('Multiclass ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')


if __name__=="__main__":
    model = load_model_inference('models/2022-02-07.pth')
    test_image(model, 'test_images/test.jpg')
    # convert_to_onnx('models/my-model.pth')
