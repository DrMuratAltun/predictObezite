import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Eğitim verilerini yükle
train = pd.read_csv('/kaggle/input/playground-series-s3e25/train.csv')

# Bireysel sınıflandırıcıları oluştur
log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
lgbm = LGBMClassifier()
xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Bir Voting Classifier oluştur
voting_clf = VotingClassifier(estimators=[
    ('lr', log_reg),
    ('lgbm', lgbm),
    ('xgb', xgb_classifier)
], voting='soft')

# Eğitim verilerini hazırla
X_train = train[['allelectrons_Total', 'density_Total', 'allelectrons_Average',
                 'val_e_Average', 'atomicweight_Average', 'ionenergy_Average',
                 'el_neg_chi_Average', 'R_vdw_element_Average', 'R_cov_element_Average',
                 'zaratio_Average', 'density_Average']]
y_train = train['Hardness']

# Modeli eğit
voting_clf.fit(X_train, y_train)

# Streamlit başlığı
st.title("Mohs Sertliği Tahmin Uygulaması")

# Kullanıcıdan girdi alma
st.header("Mineral Özelliklerini Girin:")
allelectrons_Total = st.number_input('Toplam Elektron Sayısı')
density_Total = st.number_input('Yoğunluk (Total)')
allelectrons_Average = st.number_input('Ortalama Elektron Sayısı')
val_e_Average = st.number_input('Ortalama Değerlik Elektron Sayısı')
atomicweight_Average = st.number_input('Ortalama Atom Ağırlığı')
ionenergy_Average = st.number_input('Ortalama İyonizasyon Enerjisi')
el_neg_chi_Average = st.number_input('Ortalama Elektron Negatifliği')
R_vdw_element_Average = st.number_input('Ortalama Van der Waals Yarıçapı')
R_cov_element_Average = st.number_input('Ortalama Kovalent Yarıçap')
zaratio_Average = st.number_input('Ortalama Z/A Oranı')
density_Average = st.number_input('Ortalama Yoğunluk')

# Tahmin butonu
if st.button('Sertliği Tahmin Et'):
    input_data = np.array([[allelectrons_Total, density_Total, allelectrons_Average,
                            val_e_Average, atomicweight_Average, ionenergy_Average,
                            el_neg_chi_Average, R_vdw_element_Average, R_cov_element_Average,
                            zaratio_Average, density_Average]])
    prediction = voting_clf.predict(input_data)
    st.subheader(f"Tahmin Edilen Mohs Sertliği: {prediction[0]:.2f}")
