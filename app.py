import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

# Eğitim verilerini yükle
train = pd.read_csv('train.csv')

# Verileri ön işleme tabi tut
categorical_columns = ['CAEC', 'CALC', 'MTRANS']
hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_data = hot_encoder.fit_transform(train[categorical_columns])
feature_names = hot_encoder.get_feature_names_out(categorical_columns)
encoded_df = pd.DataFrame(encoded_data, columns=feature_names)

# Etiketleri sayısal değerlere dönüştür
train['Gender'] = train['Gender'].map({'Male': 1, 'Female': 0})
train['family_history_with_overweight'] = train['family_history_with_overweight'].map({'yes': 1, 'no': 0})
train['FAVC'] = train['FAVC'].map({'yes': 1, 'no': 0})
train['SMOKE'] = train['SMOKE'].map({'yes': 1, 'no': 0})
train['SCC'] = train['SCC'].map({'yes': 1, 'no': 0})

# Veri çerçevesini birleştir
train = pd.concat([train.drop(columns=categorical_columns), encoded_df], axis=1)

# Özellik ölçeklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(train.drop(columns=['NObeyesdad']))
y_train = train['NObeyesdad']

# Label encoding for target
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

# Modeli oluştur ve eğit
log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
lgbm = LGBMClassifier()
xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

voting_clf = VotingClassifier(estimators=[
    ('lr', log_reg),
    ('lgbm', lgbm),
    ('xgb', xgb_classifier)
], voting='soft')

voting_clf.fit(X_train, y_train)

# Streamlit başlığı
st.title("Obezite Risk Tahmin Uygulaması")

# Kullanıcıdan girdi alma formu
st.header("Kişisel Bilgilerinizi Girin:")
gender = st.selectbox('Cinsiyet', ['Male', 'Female'])
family_history_with_overweight = st.selectbox('Ailede fazla kilo geçmişi var mı?', ['yes', 'no'])
FAVC = st.selectbox('Yüksek kalorili yiyecekler tüketiyor musunuz?', ['yes', 'no'])
SMOKE = st.selectbox('Sigara içiyor musunuz?', ['yes', 'no'])
SCC = st.selectbox('Kendi kalorilerinizi kontrol ediyor musunuz?', ['yes', 'no'])
CAEC = st.selectbox('Yemek yeme sıklığınız nedir?', ['no', 'Sometimes', 'Frequently', 'Always'])
CALC = st.selectbox('Alkol tüketim sıklığınız nedir?', ['no', 'Sometimes', 'Frequently', 'Always'])
MTRANS = st.selectbox('Ulaşım aracı tercihiniz nedir?', ['Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking'])
# Diğer özellikler
age = st.number_input('Yaş', min_value=1, max_value=100, value=25)
height = st.number_input('Boy (cm)', min_value=100, max_value=250, value=170)
weight = st.number_input('Kilo (kg)', min_value=30, max_value=200, value=70)

# Tahmin butonu
if st.button('Obezite Riskini Tahmin Et'):
    input_data = pd.DataFrame({
        'Gender': [1 if gender == 'Male' else 0],
        'family_history_with_overweight': [1 if family_history_with_overweight == 'yes' else 0],
        'FAVC': [1 if FAVC == 'yes' else 0],
        'SMOKE': [1 if SMOKE == 'yes' else 0],
        'SCC': [1 if SCC == 'yes' else 0],
        'CAEC': [CAEC],
        'CALC': [CALC],
        'MTRANS': [MTRANS],
        'Age': [age],
        'Height': [height],
        'Weight': [weight]
    })

    # Kategorik sütunları encode et
    encoded_data = hot_encoder.transform(input_data[categorical_columns])
    encoded_df = pd.DataFrame(encoded_data, columns=feature_names)
    input_data = pd.concat([input_data.drop(columns=categorical_columns), encoded_df], axis=1)

    # Veriyi ölçeklendir
    input_data = scaler.transform(input_data)

    # Tahmin yap
    prediction = voting_clf.predict(input_data)
    prediction_label = label_encoder.inverse_transform(prediction)
    
    st.subheader(f"Tahmin Edilen Obezite Riski: {prediction_label[0]}")
