# import librerie e dataset

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import sklearn as sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

df_original=pd.read_csv(r'df_clean_5.csv')
sklearn.set_config(transform_output='pandas')


# filtraggio colonne utili

df_original = df_original[[
    # Dati demografici e socioeconomici
    'age', 
    'sex',
    'marital_status',
    'Race recode (White, Black, Other)',
    'Race recode (with detailed Asian and Native Hawaiian other PI)',
    'Origin recode NHIA (Hispanic, Non-Hisp)',
    'median_household_income_adj_2023', 'rural_urban_continuum',
    
    # Caratteristiche del tumore
    'primary_site', 'Schema ID (2018+)', 'ICD-O-3 Hist/behav',
    'clinical_grade', 'diagnostic_confirmation',
    'tumor_size_summary',
    
    # Stadio
    'eod_t', 'eod_n', 'eod_m', 'eod_stage_group',
    'eod_primary_tumor', 'eod_regional_nodes', 'eod_mets',
    'n_sentinel_lymph_nodes',
    
    # Metastasi
    'mets_at_bone', 'mets_at_brain', 'mets_at_liver', 'mets_at_lung', 
    'mets_at_dx_distand_ln', 'mets_at_dx_other',
    
    # Biomarcatori
    'E_R_binary', 'pr_binary', 'her2_binary',
    
    # Trattamento
    'days_from_diagnosis_to_treatment',
    'rx_summ_surg_prim_site', 'rx_summ_scope_reg_ln_sur', 'rx_summ_surg_oth_reg_dis',
    'rx_summ_surg_rad_seq', 'reason_no_surgery',
    'radiation', 'chemo_yes_no', 'rx_summ_systemic_sur_seq',
    
    # Storia clinica 
    'first_malignant_tumor',
    'n_benign_borderline_tumors', 'n_in_situ_malignant_tumors',
    'survival_months',
    
    # Fonte 
    'report_source'
]]


#sampling dataframe
df = df_original.sample(frac=1, random_state=42)

# colonne numeriche e categoriche

# Colonne categoriche
cat_cols = [
    # Dati demografici
    'sex', 
    'marital_status',
    'Race recode (White, Black, Other)',
    'Race recode (with detailed Asian and Native Hawaiian other PI)',
    'Origin recode NHIA (Hispanic, Non-Hisp)',
    'rural_urban_continuum',
    
    # Caratteristiche del tumore
    'primary_site',
    'Schema ID (2018+)',
    'ICD-O-3 Hist/behav',
    'clinical_grade',
    'diagnostic_confirmation',
    'tumor_size_summary',
    
    # Stadiazione
    'eod_t', 'eod_n', 'eod_m', 'eod_stage_group',
    'eod_primary_tumor', 'eod_regional_nodes', 'eod_mets',
    'n_sentinel_lymph_nodes',
    
    # Metastasi
    'mets_at_bone', 'mets_at_brain', 'mets_at_liver', 'mets_at_lung',
    'mets_at_dx_distand_ln', 'mets_at_dx_other',
    
    # Biomarcatori
    'E_R_binary', 'pr_binary', 'her2_binary',
    
    # Trattamento
    'rx_summ_surg_prim_site', 'rx_summ_scope_reg_ln_sur', 'rx_summ_surg_oth_reg_dis',
    'rx_summ_surg_rad_seq', 'reason_no_surgery', 'radiation',
    'chemo_yes_no', 'rx_summ_systemic_sur_seq',
    
    # Storia clinica
    'first_malignant_tumor',
    
    # Fonte dei dati
    'report_source'
]

# Colonne numeriche
num_cols = [
    #'age',
    'days_from_diagnosis_to_treatment',
    'median_household_income_adj_2023',
    'n_in_situ_malignant_tumors',
    'n_benign_borderline_tumors'
]

#train_test_split
x_train, x_test, y_train, y_test=train_test_split(df.drop(columns='survival_months'),df['survival_months'], test_size=0.2, shuffle=True, random_state=42)

# encoder e pipeline
encoding=ColumnTransformer(
    [
        (
            'onehot',
            OneHotEncoder(sparse_output=False, min_frequency=5, handle_unknown='infrequent_if_exist'),
            cat_cols
        )
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
    force_int_remainder_cols=False
)

pipe_rl=Pipeline(
    [
        ('encoder', encoding),
        ('scaler', StandardScaler(with_mean=False)),
        ('linreg', LinearRegression())
    ]
)

pipe_rl.fit(x_train, y_train)
y_test_pred=pipe_rl.predict(x_test)
print(mean_absolute_error(y_test, y_test_pred))

joblib.dump(pipe_rl, 'ferrarini/ml_model/linear_regression_model.joblib')
