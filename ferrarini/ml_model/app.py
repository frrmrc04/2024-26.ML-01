from flask import Flask, request, jsonify
import joblib
import pandas as pd

app=Flask(__name__)

@app.route('/infer', methods=['POST'])
def hello():
    data = request.get_json()
    param1=data.get('age')
    param2=data.get('sex')
    param3=data.get('marital_status')   
    param4=data.get('Race recode (White, Black, Other)')
    param5=data.get('Race recode (with detailed Asian and Native Hawaiian other PI)')
    param6=data.get('Origin recode NHIA (Hispanic, Non-Hisp)')
    param7=data.get('median_household_income_adj_2023')
    param8=data.get('rural_urban_continuum')
    param9=data.get('primary_site')
    param10=data.get('Schema ID (2018+)')
    param11=data.get('ICD-O-3 Hist/behav')
    param12=data.get('clinical_grade')
    param13=data.get('diagnostic_confirmation')
    param14=data.get('tumor_size_summary')
    param15=data.get('eod_t')
    param16=data.get('eod_n')
    param17=data.get('eod_m')
    param18=data.get('eod_stage_group')
    param19=data.get('eod_primary_tumor')
    param20=data.get('eod_regional_nodes')
    param21=data.get('eod_mets')
    param22=data.get('n_sentinel_lymph_nodes')
    param23=data.get('mets_at_bone')
    param24=data.get('mets_at_brain')
    param25=data.get('mets_at_liver')
    param26=data.get('mets_at_lung')
    param27=data.get('mets_at_dx_distand_ln')
    param28=data.get('mets_at_dx_other')
    param29=data.get('E_R_binary')
    param30=data.get('pr_binary')
    param31=data.get('her2_binary')
    param32=data.get('days_from_diagnosis_to_treatment')
    param33=data.get('rx_summ_surg_prim_site')
    param34=data.get('rx_summ_scope_reg_ln_sur')
    param35=data.get('rx_summ_surg_oth_reg_dis')
    param36=data.get('rx_summ_surg_rad_seq')
    param37=data.get('reason_no_surgery')
    param38=data.get('radiation')
    param39=data.get('chemo_yes_no')
    param40=data.get('rx_summ_systemic_sur_seq')
    param41=data.get('first_malignant_tumor')
    param42=data.get('n_benign_borderline_tumors')
    param43=data.get('n_in_situ_malignant_tumors')
    param44=data.get('report_source')
    
    parameters = pd.DataFrame({
        'age': [param1],
        'sex': [param2],
        'marital_status': [param3],
        'Race recode (White, Black, Other)': [param4],
        'Race recode (with detailed Asian and Native Hawaiian other PI)': [param5],
        'Origin recode NHIA (Hispanic, Non-Hisp)': [param6],
        'median_household_income_adj_2023': [param7],
        'rural_urban_continuum': [param8],
        'primary_site': [param9],
        'Schema ID (2018+)': [param10],
        'ICD-O-3 Hist/behav': [param11],
        'clinical_grade': [param12],
        'diagnostic_confirmation': [param13],
        'tumor_size_summary': [param14],
        'eod_t': [param15],
        'eod_n': [param16],
        'eod_m': [param17],
        'eod_stage_group': [param18],
        'eod_primary_tumor': [param19],
        'eod_regional_nodes': [param20],
        'eod_mets': [param21],
        'n_sentinel_lymph_nodes': [param22],
        'mets_at_bone': [param23],
        'mets_at_brain': [param24],
        'mets_at_liver': [param25],
        'mets_at_lung': [param26],
        'mets_at_dx_distand_ln': [param27],
        'mets_at_dx_other': [param28],
        'E_R_binary': [param29],
        'pr_binary': [param30],
        'her2_binary': [param31],
        'days_from_diagnosis_to_treatment': [param32],
        'rx_summ_surg_prim_site': [param33],
        'rx_summ_scope_reg_ln_sur': [param34],
        'rx_summ_surg_oth_reg_dis': [param35],
        'rx_summ_surg_rad_seq': [param36],
        'reason_no_surgery': [param37],
        'radiation': [param38],
        'chemo_yes_no': [param39],
        'rx_summ_systemic_sur_seq': [param40],
        'first_malignant_tumor': [param41],
        'n_benign_borderline_tumors': [param42],
        'n_in_situ_malignant_tumors': [param43],
        'report_source': [param44]
    })
    modello=joblib.load('linear_regression_model.joblib')
    
    print("###################### PREDICT PARAMETERS NEXT LINE ############################")
    print(parameters)

    print("###################### PREDICT NEXT LINE ############################")
    result=modello.predict(parameters)

    print("########################################################################")
    response_data = {"prediction": result[0]}
    print(response_data)
    print("########################################################################")
    return jsonify(response_data)

