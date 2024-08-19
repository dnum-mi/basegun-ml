from basegun_ml.ocr import is_alarm_weapon
from basegun_ml.exceptions import LowQuality, MissingText
import os
import pytest


this_dir, this_filename = os.path.split(__file__)


def to_bytes(img):
    with open(img, "rb") as file:
        image_bytes = file.read()
    return image_bytes


class TestOCR:
    def test_LowQuality(self):
        with pytest.raises(LowQuality):
            is_alarm_weapon(
                to_bytes(this_dir + "/tests_images/test_ocr/bad_quality.JPG")
            )

    def test_NoText(self):
        with pytest.raises(MissingText):
            is_alarm_weapon(to_bytes(this_dir + "/tests_images/test_ocr/no_text.JPG"))

    def test_LowQualityBypass(self):
        pred = is_alarm_weapon(
            to_bytes(this_dir + "/tests_images/test_ocr/bad_quality.JPG"),
            quality_check=False,
        )
        assert pred == "Not_alarm"

    def test_NotAlarm(self):
        pred = is_alarm_weapon(
            to_bytes(this_dir + "/tests_images/test_ocr/not_alarm.JPG")
        )
        assert pred == "Not_alarm"

    def test_PAK(self):
        pred = is_alarm_weapon(to_bytes(this_dir + "/tests_images/test_ocr/PAK.JPG"))
        assert pred == "PAK"

    def test_AlarmModel(self):
        pred = is_alarm_weapon(
            to_bytes(this_dir + "/tests_images/test_ocr/alarm_model.JPG")
        )
        assert pred == "Alarm_model"
