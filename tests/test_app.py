"""Tests du module Flask principal: mapping features et routes HTTP."""

import os
import sys

import pytest
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from app import build_feature_vector, prepare_input              

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


                                                    

class TestBuildFeatureVector:

    def test_numeric_fields(self):
        form = {"Age": "10", "BMI": "18.5", "Appendix_Diameter": "8",
                "Body_Temperature": "38.2", "WBC_Count": "15000", "CRP": "50"}
        vec = build_feature_vector(form)
        assert vec["Age"] == 10.0
        assert vec["BMI"] == 18.5
        assert vec["CRP"] == 50.0

    def test_wbc_crp_ratio(self):
        form = {"WBC_Count": "10000", "CRP": "99.9"}
        vec = build_feature_vector(form)
        expected = 10000 / (99.9 + 0.1)
        assert abs(vec["WBC_CRP_Ratio"] - expected) < 0.01

    def test_wbc_crp_ratio_zero_crp(self):
        form = {"WBC_Count": "10000", "CRP": "0"}
        vec = build_feature_vector(form)
        expected = 10000 / 0.1
        assert abs(vec["WBC_CRP_Ratio"] - expected) < 0.01

    def test_sex_male(self):
        vec = build_feature_vector({"Sex": "male"})
        assert vec["Sex_male"] == 1

    def test_sex_female(self):
        vec = build_feature_vector({"Sex": "female"})
        assert vec["Sex_male"] == 0

    def test_binary_yes(self):
        fields = ["Migratory_Pain", "Lower_Right_Abd_Pain",
                   "Contralateral_Rebound_Tenderness", "Coughing_Pain",
                   "Nausea", "Loss_of_Appetite", "Neutrophilia",
                   "Psoas_Sign", "Ipsilateral_Rebound_Tenderness",
                   "Appendix_on_US", "Free_Fluids"]
        for field in fields:
            vec_yes = build_feature_vector({field: "yes"})
            vec_no = build_feature_vector({field: "no"})
            assert vec_yes[f"{field}_yes"] == 1, f"{field}_yes devrait être 1"
            assert vec_no[f"{field}_yes"] == 0, f"{field}_yes devrait être 0"

    def test_peritonitis_local(self):
        vec = build_feature_vector({"Peritonitis": "local"})
        assert vec["Peritonitis_local"] == 1
        assert vec["Peritonitis_no"] == 0

    def test_peritonitis_no(self):
        vec = build_feature_vector({"Peritonitis": "no"})
        assert vec["Peritonitis_local"] == 0
        assert vec["Peritonitis_no"] == 1

    def test_peritonitis_generalized(self):
        vec = build_feature_vector({"Peritonitis": "generalized"})
        assert vec["Peritonitis_local"] == 0
        assert vec["Peritonitis_no"] == 0

    def test_missing_fields_default_to_zero(self):
        vec = build_feature_vector({})
        assert vec["Age"] == 0.0
        assert vec["Sex_male"] == 0
        assert vec["Nausea_yes"] == 0

    def test_empty_string_defaults(self):
        vec = build_feature_vector({"Age": "", "WBC_Count": ""})
        assert vec["Age"] == 0.0
        assert vec["WBC_Count"] == 0.0

    def test_output_has_21_model_features(self):
        form = {"Age": "12", "BMI": "20", "Sex": "male", "Peritonitis": "no",
                "Body_Temperature": "38", "WBC_Count": "12000", "CRP": "30",
                "Appendix_Diameter": "9"}
        vec = build_feature_vector(form)
        feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
        for feat in feature_names:
            assert feat in vec, f"Feature manquante : {feat}"

    def test_prepare_input_shape(self):
        form = {"Age": "10", "BMI": "18", "WBC_Count": "10000", "CRP": "20"}
        vec = build_feature_vector(form)
        X = prepare_input(vec)
        feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
        assert X.shape == (1, len(feature_names))

    def test_end_to_end_prediction(self):
        form = {"Age": "12", "BMI": "20", "Sex": "male",
                "Body_Temperature": "38.5", "WBC_Count": "15000", "CRP": "80",
                "Appendix_Diameter": "10", "Nausea": "yes",
                "Lower_Right_Abd_Pain": "yes", "Peritonitis": "local",
                "Appendix_on_US": "yes", "Free_Fluids": "yes"}
        vec = build_feature_vector(form)
        X = prepare_input(vec)
        model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
        pred = model.predict(X)
        proba = model.predict_proba(X)
        assert pred[0] in [0, 1]
        assert 0 <= proba[0][1] <= 1


                            

class TestFlaskApp:
    """Validation des routes accessibles et des routes protégées."""
                                 

    @pytest.fixture(autouse=True)
    def _setup_client(self, tmp_path):
        from app import app
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.app = app
        self.client = app.test_client()

    def _login(self, username="admin", password="admin123"):
        return self.client.post("/login", data={
            "username": username, "password": password,
        }, follow_redirects=True)

    def test_home_page(self):
        resp = self.client.get("/")
        assert resp.status_code == 200
        assert b"PediAppend" in resp.data

    def test_diagnosis_requires_login(self):
        resp = self.client.get("/diagnosis")
        assert resp.status_code == 302

    def test_login_page_renders(self):
        resp = self.client.get("/login")
        assert resp.status_code == 200

    def test_register_page_renders(self):
        resp = self.client.get("/register")
        assert resp.status_code == 200

    def test_login_with_valid_credentials(self):
        resp = self._login()
        assert resp.status_code == 200

    def test_login_with_invalid_credentials(self):
        resp = self.client.post("/login", data={
            "username": "admin", "password": "wrongpassword",
        }, follow_redirects=True)
        assert b"incorrect" in resp.data

    def test_diagnosis_page_after_login(self):
        self._login()
        resp = self.client.get("/diagnosis")
        assert resp.status_code == 200

    def test_predict_route(self):
        self._login()
        resp = self.client.post("/predict", data={
            "patient_first_name": "Test",
            "patient_last_name": "Patient",
            "Age": "10",
            "BMI": "18.5",
            "Sex": "male",
            "Body_Temperature": "38.0",
            "WBC_Count": "12000",
            "CRP": "30",
            "Appendix_Diameter": "7",
            "Nausea": "yes",
            "Lower_Right_Abd_Pain": "yes",
            "Migratory_Pain": "no",
            "Contralateral_Rebound_Tenderness": "no",
            "Coughing_Pain": "no",
            "Loss_of_Appetite": "yes",
            "Neutrophilia": "yes",
            "Psoas_Sign": "no",
            "Ipsilateral_Rebound_Tenderness": "no",
            "Appendix_on_US": "yes",
            "Free_Fluids": "no",
            "Peritonitis": "no",
        }, follow_redirects=True)
        assert resp.status_code == 200
        assert b"appendicite" in resp.data.lower()

    def test_history_page(self):
        self._login()
        resp = self.client.get("/history")
        assert resp.status_code == 200

    def test_logout(self):
        self._login()
        resp = self.client.get("/logout", follow_redirects=True)
        assert resp.status_code == 200
        resp2 = self.client.get("/diagnosis")
        assert resp2.status_code == 302

    def test_profile_requires_login(self):
        resp = self.client.get("/profile")
        assert resp.status_code == 302

    def test_admin_requires_admin(self):
        self._login()
        resp = self.client.get("/admin")
        assert resp.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
