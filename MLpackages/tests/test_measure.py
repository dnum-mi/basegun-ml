from basegun_ml.measure import get_lengths
import os


this_dir, this_filename = os.path.split(__file__)


def to_bytes(img):
    with open(img, "rb") as file:
        image_bytes = file.read()
    return image_bytes


def equalMarg(a, b, margin):
    #measure if the predicted length is close enough to the true length
    return abs(b - a) < margin


class TestMeasure:
    def test_Noweapon(self):
        pred = get_lengths(
            to_bytes(this_dir + "/tests_images/test_measure/noWeapon.JPG"), draw=False
        )
        assert pred == (0, 0, 0)

    def test_NoCard(self):
        pred = get_lengths(
            to_bytes(this_dir + "/tests_images/test_measure/noCard.jpg"), draw=False
        )
        assert pred == (0, 0, 0)

    def test_perfverrou(self):
        pred = get_lengths(
            to_bytes(this_dir + "/tests_images/test_measure/verrou.jpg"), draw=False
        )
        assert equalMarg(pred[0], 111, 4) and equalMarg(pred[1], 56.5, 4)

    def test_perfuncoup(self):
        pred = get_lengths(
            to_bytes(this_dir + "/tests_images/test_measure/uncoup.jpg"), draw=False
        )
        assert equalMarg(pred[0], 93, 4) and equalMarg(pred[1], 51, 4)

    def test_perflevier(self):
        pred = get_lengths(
            to_bytes(this_dir + "/tests_images/test_measure/levier.jpg"), draw=False
        )
        assert equalMarg(pred[0], 97, 4) and equalMarg(pred[1], 51, 4)

    def test_perfsemiChasse(self):
        pred = get_lengths(
            to_bytes(this_dir + "/tests_images/test_measure/semiChasse.jpg"), draw=False
        )
        assert equalMarg(pred[0], 107, 4) and equalMarg(pred[1], 50, 4)

    def test_perfsemiMil(self):
        pred = get_lengths(
            to_bytes(this_dir + "/tests_images/test_measure/mil_autre.jpg"), draw=False
        )
        assert equalMarg(pred[0], 108, 4) and equalMarg(pred[1], 51, 4)

    def test_perfsemiMil20(self):
        pred = get_lengths(
            to_bytes(this_dir + "/tests_images/test_measure/mil20.jpg"), draw=False
        )
        assert equalMarg(pred[0], 91, 4) and equalMarg(pred[1], 42, 4)

    def test_perfpompe(self):
        pred = get_lengths(
            to_bytes(this_dir + "/tests_images/test_measure/pompe.jpg"), draw=False
        )
        assert equalMarg(pred[0], 117, 4) and equalMarg(pred[1], 59, 4)
