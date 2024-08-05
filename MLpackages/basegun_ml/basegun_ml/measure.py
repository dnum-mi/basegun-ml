import cv2
import numpy as np
from basegun_ml import model_card, model_keypoints
from basegun_ml.utils import rotate, distanceCalculate, scalarproduct
from basegun_ml.exception import MissingCard, MissingGun


def get_card(image, model):
    """Predict the keypoints on the image
    Args:
        image (opencv matrix): image after CV2.imread(path)
        modelCard (model): model after load_models call

    Returns:
        Prediction: Oriented boundng box(x,y,x,y,x,y,x,y ,CONF_THRES, NMS_THRES)
    """
    return model.run(image)


def get_keypoints(image, model):
    """Predict the keypoints on the image
    Args:
        image (opencv matrix): image after CV2.imread(path)
        modelWeapon (model): model after load_models call

    Returns:
        Prediction: keypoints coordinates [[KP1x,KP1y],[KP2x,KP2y],[KP3x,KP3y],[KP4x,KP4y]]
    """
    results = model(image, verbose=False)
    return results[0].keypoints.data[0]


def get_lengths(imagebytes, draw=True, output_filename="result.jpg"):
    """Predict the keypoints on the image
    Args:
        image (bytes):
        draw (Boolean): whether the result image need to be drawed and saved
        output_filename: Filename and location for the image output

    Returns:
        Length (list): Overall Length, Barrel Length, Card detection confidence score
    """
    image = cv2.imdecode(np.asarray(bytearray(imagebytes)), cv2.IMREAD_COLOR)
    image = rotate(image)

    keypoints = get_keypoints(image, model_keypoints)
    if len(keypoints) == 0:
        raise MissingGun
    if keypoints[3][0] < keypoints[0][0]:  # Weapon upside down
        image = cv2.rotate(image, cv2.ROTATE_180)
        keypoints = get_keypoints(image, model_keypoints)

    cards = get_card(image, model_card)
    if len(cards) == 0:
        raise MissingCard
    card = cards[0]
    confCard = card[8]
    CardP = distanceCalculate((card[0], card[1]), (card[4], card[5]))
    CardP = distanceCalculate((card[2], card[3]), (card[6], card[7]))
    CardR = (8.56**2 + 5.398**2) ** 0.5

    factor = CardR / CardP
    canonP = distanceCalculate(
        (int(keypoints[2][0]), int(keypoints[2][1])),
        (int(keypoints[3][0]), int(keypoints[3][1])),
    )
    canonR = round(canonP * factor, 2)

    totalP1 = scalarproduct(keypoints[0] - keypoints[3], keypoints[2] - keypoints[3])
    totalP2 = scalarproduct(keypoints[1] - keypoints[3], keypoints[2] - keypoints[3])

    totalP = float(max(totalP1, totalP2))

    totalR = round(totalP * factor, 2)

    if draw:
        img2 = image
        for keypoint in keypoints:
            img2 = cv2.circle(
                img2,
                (int(keypoint[0]), int(keypoint[1])),
                radius=5,
                color=(0, 0, 255),
                thickness=20,
            )

        img2 = cv2.line(
            img2,
            (int(card[0]), int(card[1])),
            (int(card[2]), int(card[3])),
            color=(255, 0, 0),
            thickness=15,
        )
        img2 = cv2.line(
            img2,
            (int(card[4]), int(card[5])),
            (int(card[2]), int(card[3])),
            color=(255, 0, 0),
            thickness=15,
        )
        img2 = cv2.line(
            img2,
            (int(card[6]), int(card[7])),
            (int(card[4]), int(card[5])),
            color=(255, 0, 0),
            thickness=15,
        )
        img2 = cv2.line(
            img2,
            (int(card[0]), int(card[1])),
            (int(card[6]), int(card[7])),
            color=(255, 0, 0),
            thickness=15,
        )
        img2 = cv2.line(
            img2,
            (int(card[0]), int(card[1])),
            (int(card[4]), int(card[5])),
            color=(255, 255, 0),
            thickness=15,
        )

        cv2.imwrite(output_filename, img2)

    return (totalR, canonR, confCard)
