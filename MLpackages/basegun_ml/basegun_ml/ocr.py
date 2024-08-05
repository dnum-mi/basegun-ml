from basegun_ml import model_ocr, metric_iqa
from fuzzysearch import find_near_matches
import io
import PIL.Image as Image
import numpy as np
from basegun_ml.exception import MissingText, LowQuality


QUALITY_THRESHOLD = 0.50


def get_text(results):
    """extracts raw text from PaddleOCR output
    Args:
        results: raw result from PaddleOCR

    Returns:
        text: A string with the text extracted from the image
    """
    text = " "
    for result in results:
        text = text + result[1][0] + " "
    return text.lower()


def is_in(word, phrase):
    """Check if a word is in a word using fuzzysearch algorithm for a tolerance error
    Args:
        word: word seek in the text
        phrase: text to explore

    Returns:
        boolean: true if word is in phrase
    """
    res = find_near_matches(word, phrase, max_l_dist=1)
    return len(res) > 0


def is_alarm_model(text):
    """determine if the text is from an alarm model weapon image using rules defined with weapon experts
    Args:
        text: string of the extract text

    Returns:
        boolean: true if the an alarm model is recognized
    """
    # fuzzy search for words but exat value for model number
    zoraki = ["r2", "925", "92s", "906", "2906", "918", "9o6", "29o6"]

    # Blow
    if is_in("blow", text):
        if any(word in text for word in ["f92", "c75"]):
            return True
        else:
            return False
    # Zoraki
    elif is_in("zoraki", text):
        if any(word in text for word in zoraki):
            return True
        else:
            return False
    # Kimar
    elif is_in("kimar", text):
        if is_in("auto", text):
            if "75" in text:
                return True
            else:
                return False
        elif "911" in text:
            return True
        else:
            return False
    elif is_in("auto", text):
        if any(word in text for word in ["92", "85"]):
            return True
        else:
            return False
    elif is_in("lady k", text):
        return True
    elif is_in("python", text):
        return True
    elif "pk4" in text:
        return True
    elif is_in(
        "alarm", text
    ):  # Sur ce type de modèle il arrive que le mot kimar soit remplacé par le logo
        if any(is_in(word, text) for word in ["competitive", "power"]):
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
    if any(
        word in text
        for word in [
            "pak ",
            "p.a.k",
            "pak.",
            " pak",
            "pa.k",
            "p.ak",
            "knall",
            "P.A.Knall",
        ]
    ):
        return True
    else:
        return False


def quality_eval(img):
    """Evaluate the CNNIQA for image quality and compare it to a defined threshold
    Args:
        img: PIL image

    Returns:
        boolean: true if the image has a good quality (score<threshold)
    """
    width, height = img.size
    ratio = 640 / width
    newsize = (640, int(height * ratio))
    im1 = img.resize(newsize)
    res = metric_iqa(im1)
    print(res)
    return res > QUALITY_THRESHOLD


def is_alarm_weapon(image_bytes, quality_check=True):
    """Global pipeline for determining if the weapon is an alarm gun using OCR
    Args:
        image_bytes: Bytes image from Basegun

    Returns:
        string: User feedback on image quality or on alarm gun assessment
    """

    img = Image.open(io.BytesIO(image_bytes))
    if (
        quality_check
    ):  # possibilité ne pas prendre en compte la verification de qualité d'image
        eval = quality_eval(img)
    else:
        eval = True

    if eval:
        results = model_ocr.ocr(np.asarray(img), cls=True)
        if (
            results != [None]
        ):  # The results with recongition and detection confidence below 0.5 are filtered by paddle, the thresholds values can be changed
            text = get_text(results[0])
            if is_alarm_model(text):
                return "alarm weapon from model"
            elif is_pak(text):
                return "alarm weapon PAK"
            else:
                return "Not an alarm weapon"
        else:
            raise MissingText
    else:
        raise LowQuality
