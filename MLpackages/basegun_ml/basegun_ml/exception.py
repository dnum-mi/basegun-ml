class MissingGun(Exception):
    "Raised when the gun is not detected in the measure module"
    pass

class MissingCard(Exception):
    "Raised when the card is not detected in the measure module"
    pass

class LowQuality(Exception):
    "Raised when the image does not have a sufficient quality"
    pass

class MissingText(Exception):
    "Raised when text is not detected in the reading module"
    pass