import os

from basegun_ml.classification import get_typology

this_dir, this_filename = os.path.split(__file__)


def to_bytes(img):
    with open(img, "rb") as file:
        image_bytes = file.read()
    return image_bytes


class TestTypology:
    def test_verrou(self):
        pred = get_typology(to_bytes(this_dir + "/tests_images/test_typo/a_verrou.jpg"))
        assert pred[0] == "epaule_a_verrou"

    def test_autre_pistolet(self):
        pred = get_typology(
            to_bytes(this_dir + "/tests_images/test_typo/autrepistolet.jpg")
        )
        assert pred[0] == "autre_pistolet"

    def test_epaule_ancien(self):
        pred = get_typology(
            to_bytes(this_dir + "/tests_images/test_typo/epaule_ancien.jpg")
        )
        assert pred[0] == "epaule_mecanisme_ancien"

    def test_levier_sous_garde(self):
        pred = get_typology(
            to_bytes(this_dir + "/tests_images/test_typo/levier_sous_garde.jpg")
        )
        assert pred[0] == "epaule_a_levier_sous_garde"

    def test_pistolet_ancien(self):
        pred = get_typology(
            to_bytes(this_dir + "/tests_images/test_typo/pistolet_ancien.jpg")
        )
        assert pred[0] == "pistolet_mecanisme_ancien"

    def test_pistolet_semi_auto(self):
        pred = get_typology(
            to_bytes(this_dir + "/tests_images/test_typo/pistolet_semi_auto.jpg")
        )
        assert pred[0] == "pistolet_semi_auto_moderne"

    def test_pompe(self):
        pred = get_typology(to_bytes(this_dir + "/tests_images/test_typo/pompe.jpg"))
        assert pred[0] == "epaule_a_pompe"

    def test_revolver(self):
        pred = get_typology(to_bytes(this_dir + "/tests_images/test_typo/revolver.jpg"))
        assert pred[0] == "revolver"

    def test_semi_auto_chasse(self):
        pred = get_typology(
            to_bytes(this_dir + "/tests_images/test_typo/semi_auto_chasse.jpg")
        )
        assert pred[0] == "epaule_semi_auto_style_chasse"

    def test_semi_auto_mil20(self):
        pred = get_typology(
            to_bytes(this_dir + "/tests_images/test_typo/semi_auto_mil_20.jpg")
        )
        assert pred[0] == "epaule_semi_auto_style_militaire_milieu_20e"

    def test_semi_auto_mil_autre(self):
        pred = get_typology(
            to_bytes(this_dir + "/tests_images/test_typo/semi_auto_mil_autre.jpg")
        )
        assert pred[0] == "semi_auto_style_militaire_autre"

    def test_un_coup(self):
        pred = get_typology(to_bytes(this_dir + "/tests_images/test_typo/un_coup.jpg"))
        assert pred[0] == "epaule_a_un_coup_par_canon"


class TestConfidence:
    def test_verrou(self):
        pred = get_typology(to_bytes(this_dir + "/tests_images/test_typo/a_verrou.jpg"))
        assert pred[2] == "high"

    def test_autre_pistolet(self):
        pred = get_typology(
            to_bytes(this_dir + "/tests_images/test_typo/autrepistolet.jpg")
        )
        assert pred[2] == "high"

    def test_epaule_ancien(self):
        pred = get_typology(
            to_bytes(this_dir + "/tests_images/test_typo/epaule_ancien.jpg")
        )
        assert pred[2] == "high"

    def test_levier_sous_garde(self):
        pred = get_typology(
            to_bytes(this_dir + "/tests_images/test_typo/levier_sous_garde.jpg")
        )
        assert pred[2] == "high"

    def test_pistolet_ancien(self):
        pred = get_typology(
            to_bytes(this_dir + "/tests_images/test_typo/pistolet_ancien.jpg")
        )
        assert pred[2] == "high"

    def test_pistolet_semi_auto(self):
        pred = get_typology(
            to_bytes(this_dir + "/tests_images/test_typo/pistolet_semi_auto.jpg")
        )
        assert pred[2] == "high"

    def test_pompe(self):
        pred = get_typology(to_bytes(this_dir + "/tests_images/test_typo/pompe.jpg"))
        assert pred[2] == "high"

    def test_revolver(self):
        pred = get_typology(to_bytes(this_dir + "/tests_images/test_typo/revolver.jpg"))
        assert pred[2] == "high"

    def test_semi_auto_chasse(self):
        pred = get_typology(
            to_bytes(this_dir + "/tests_images/test_typo/semi_auto_chasse.jpg")
        )
        assert pred[2] == "high"

    def test_semi_auto_mil20(self):
        pred = get_typology(
            to_bytes(this_dir + "/tests_images/test_typo/semi_auto_mil_20.jpg")
        )
        assert pred[2] == "high"

    def test_semi_auto_mil_autre(self):
        pred = get_typology(
            to_bytes(this_dir + "/tests_images/test_typo/semi_auto_mil_autre.jpg")
        )
        assert pred[2] == "high"

    def test_un_coup(self):
        pred = get_typology(to_bytes(this_dir + "/tests_images/test_typo/un_coup.jpg"))
        assert pred[2] == "high"

    def test_autre(self):
        pred = get_typology(to_bytes(this_dir + "/tests_images/test_typo/DTNUM.png"))
        assert pred[2] == "low"
