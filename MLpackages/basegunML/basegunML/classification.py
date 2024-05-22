from io import BytesIO
from typing import Union
from ultralytics import YOLO
from PIL import Image
from basegunML import modelClassif


CLASSES = [
    "autre_pistolet",
    "epaule_a_levier_sous_garde",
    "epaule_a_pompe",
    "epaule_a_un_coup_par_canon",
    "epaule_a_verrou",
    "epaule_mecanisme_ancien",
    "epaule_semi_auto_style_chasse",
    "epaule_semi_auto_style_militaire_milieu_20e",
    "pistolet_mecanisme_ancien",
    "pistolet_semi_auto_moderne",
    "revolver",
    "semi_auto_style_militaire_autre",
]


def load_model_inference(model_path: str):
    """Load model structure and weights

    Args:
        model_path (str): path to model (.pt file)

    Returns:
        Model: loaded model ready for prediction
    """
    model=YOLO(model_path)
    return model


def get_typology(img: bytes) -> Union[str, float,str]:
    """Run the model prediction on an image

    Args:
        img (bytes): input image in bytes

    Returns:
        Union[str, float,str]: (label, confidence score,confidence level) of best class predicted
    """
    im = Image.open(BytesIO(img))
    results = modelClassif(im,verbose=False)
    predicted_class=results[0].probs.top5[0]
    label=CLASSES[predicted_class]
    confidence=float(results[0].probs.top5conf[0])
    return (label, confidence,confLevel(confidence))


def confLevel(conf:float)-> str:
    """determine a confidence level for the prediction

    Args:
        conf (float): classification confidence score

    Returns:
        str: confidence level "high, medium, low"
    """
    if conf< 0.76:
        return "low"
    elif conf < 0.98:
        return "medium"
    else:
        return "high"
    
def listTypologies()-> list:
    """determine a confidence level for the prediction

    Returns:
        lsit: list of typologies
    """
    return CLASSES