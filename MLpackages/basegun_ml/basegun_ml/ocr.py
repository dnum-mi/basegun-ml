from basegun_ml import model_ocr
from fuzzysearch import find_near_matches
import pyiqa
import torch
import io
import PIL.Image as Image
import numpy as np


THRESHOLD=35

def get_text(results):
    """extracts raw text from PaddleOCR output
    Args:
        results: raw result from PaddleOCR

    Returns:
        text: A string with the text extracted from the image
    """
    text=" "
    for result in results:
        text=text+result[0][1][0]+" "
    return text.lower()


def is_in(word,phrase):
    """Check if a word is in a word using fuzzysearch algorithm for a tolerance error
    Args:
        word: word seek in the text
        phrase: text to explore

    Returns:
        boolean: true if word is in phrase
    """
    res=find_near_matches(word, phrase, max_l_dist=1)
    return len(res)>0
    
def is_alarm_model(text):
    """determine if the text is from an alarm model weapon image using rules defined with weapon experts
    Args:
        text: string of the extract text

    Returns:
        boolean: true if the an alarm model is recognized
    """
    #fuzzy search for words but exat value for model number
    zoraki=["r2", "925","906","2906","918"]
    
    #Blow
    if is_in("blow",text):
        if any(word in text for word in ["f92","c75"]):
            return True
        else:
            return False
    #Zoraki
    elif is_in("zoraki",text):
        if any(word in text for word in zoraki):
            return True
        else:
            return False
    
    elif is_in("kimar",text):
        if "auto"in text:
            if any(word in text for word in ["75","92"]):
                return True
            else:
                return False
        elif "911" in text:
            return True
        else:
            return False
    elif is_in("alarm",text): #Sur ce type de modèle il arrive que le mot kimar soit remplacé par le logo
            if any(is_in(word,text) for word in ["competitive","power"]):
                return True
            else:
                return False
    
    else:
        return False

def is_pak(text):
    """determine if the text is from an alarm model weapon image with a PAK engraving
    Args:
        text: string of the extract text

    Returns:
        boolean: true if the PAK engraving is recognized
    """
    if any(word in text for word in ["pak ","p.a.k","pak."," pak"]):
        return True
    else:
        return False
    
def brisque_eval(img):
    """Evaluate the brisque metric for image quality and compare it to a defined threshold
    Args:
        img: PIL image

    Returns:
        boolean: true if the image has a good quality (score<threshold)
    """
    device = torch.device("cpu")
    brisque = pyiqa.create_metric('brisque', device=device)
    res=brisque(img)
    print(res)
    return res<THRESHOLD


def is_alarm_weapon(image_bytes):
    """Global pipeline for determining if the weapon is an alarm gun using OCR
    Args:
        image_bytes: Bytes image from Basegun

    Returns:
        string: User feedback on image quality or on alarm gun assessment
    """
    img = Image.open(io.BytesIO(image_bytes))

    if brisque_eval(img):
        results = model_ocr.ocr(np.asarray(img), cls=True)
        print(results)
        if results!=[None]:
            text=get_text(results)
            if is_alarm_model(text):
                return "alarm weapon from model"
            elif is_pak(text):
                return "alarm weapon PAK"
            else:
                return "Not an alarm weapon"
        else:
            return "Text not detected please get closer to the weapon"
    else:
        return "The photo does not seem to have a good quality please take another photo"