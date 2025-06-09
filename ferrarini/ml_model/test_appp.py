import pytest
from ferrarini.ml_model.app import app as flask_app


@pytest.fixture()
def client():
    flask_app.config.update({"TESTING": True})
    with flask_app.test_client() as client:
        yield client





def test_infer_prediction(client):
    # Inserisci tutti i 44 parametri richiesti, esempio con valori dummy
    payload = {
        'age': 80,  # int
        'sex': '0',  # mantengo stringa per categorie
        'marital_status': "Married (including common law)",
        'Race recode (White, Black, Other)': "White",
        'Race recode (with detailed Asian and Native Hawaiian other PI)': "White",
        'Origin recode NHIA (Hispanic, Non-Hisp)': '0',
        'median_household_income_adj_2023': 115000.0,  # float
        'rural_urban_continuum': "Counties in metropolitan areas of 250,000 to 1 million pop",
        'primary_site': "C34.1-Upper lobe, lung",
        'Schema ID (2018+)': "Lung",
        'ICD-O-3 Hist/behav': "8140/3: Adenocarcinoma, NOS",
        'clinical_grade': 3,
        'diagnostic_confirmation': "Positive histology",
        'tumor_size_summary': 14,
        'eod_t': "T1mi",
        'eod_n': "N0",
        'eod_m': "M0",
        'eod_stage_group': "1A1",
        'eod_primary_tumor': '100',
        'eod_regional_nodes': '0',
        'eod_mets': '0',
        'n_sentinel_lymph_nodes': "Not available (not breast or melanoma skin schemas)",
        'mets_at_bone': "No",
        'mets_at_brain': "No",
        'mets_at_liver': "No",
        'mets_at_lung': "No",
        'mets_at_dx_distand_ln': "None; no lymph node metastases",
        'mets_at_dx_other': "None; no other metastases",
        'E_R_binary': None,  # NaN -> None
        'pr_binary': None,  # NaN -> None
        'her2_binary': None,  # NaN -> None
        'days_from_diagnosis_to_treatment': 29.449965,  # float
        'rx_summ_surg_prim_site': 0,  # numeric
        'rx_summ_scope_reg_ln_sur': "Unknown or not applicable",
        'rx_summ_surg_oth_reg_dis': "None; diagnosed at autopsy",
        'rx_summ_surg_rad_seq': "No radiation and/or no surgery; unknown if surgery and/or radiation given",
        'reason_no_surgery': "Not recommended, contraindicated due to other conditions",
        'radiation': "None/Unknown",
        'chemo_yes_no': 0,
        'rx_summ_systemic_sur_seq': "No systemic therapy and/or surgical procedures",
        'first_malignant_tumor': 0,
        'n_benign_borderline_tumors': 0,
        'n_in_situ_malignant_tumors': 2,
        'report_source': "Hospital inpatient/outpatient or clinic"
    }


    response = client.post("/infer", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert data["prediction"]>0  # oppure altro tipo se è regressione