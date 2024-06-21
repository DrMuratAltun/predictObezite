import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, PowerTransformer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Eğitim verilerini yükle
train = pd.read_csv('train.csv')

# Kategorik sütunları belirle
categorical_columns = ['CAEC', 'CALC', 'MTRANS']
label_column = 'NObeyesdad'

# Kategorik sütunları OneHotEncode et
hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_data = hot_encoder.fit_transform(train[categorical_columns])
feature_names = hot_encoder.get_feature_names_out(categorical_columns)
encoded_df = pd.DataFrame(encoded_data, columns=feature_names)

# Orijinal DataFrame'i güncelle
train = pd.concat([train.drop(columns=categorical_columns), encoded_df], axis=1)

# Diğer dönüşümler
train['Gender'] = train['Gender'].map({'Male': 1, 'Female': 0})
train['family_history_with_overweight'] = train['family_history_with_overweight'].map({'yes': 1, 'no': 0})
train['FAVC'] = train['FAVC'].map({'yes': 1, 'no': 0})
train['SMOKE'] = train['SMOKE'].map({'yes': 1, 'no': 0})
train['SCC'] = train['SCC'].map({'yes': 1, 'no': 0})

# Özellikler ve hedef değişkeni ayır
X = train.drop(columns=[label_column])
y = train[label_column]

# Label Encoding for target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Verileri ölçeklendirme ve dönüştürme
scaler = StandardScaler()
X = scaler.fit_transform(X)

pt = PowerTransformer()
X = pt.fit_transform(X)

# Modeli eğit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Streamlit başlığı
st.title("Obezite Risk Tahmin Uygulaması")

# Kullanıcıdan girdi alma
st.header("Bireysel Özellikleri Girin:")

gender = st.selectbox('Cinsiyet', ['Male', 'Female'])
age = st.number_input('Yaş', min_value=0, max_value=120, value=25)
height = st.number_input('Boy (cm)', min_value=50, max_value=250, value=170)
weight = st.number_input('Kilo (kg)', min_value=10, max_value=300, value=70)
family_history = st.selectbox('Ailede aşırı kilo geçmişi var mı?', ['yes', 'no'])
favc = st.selectbox('Yüksek kalorili yiyecek tercih ediyor musunuz?', ['yes', 'no'])
smoke = st.selectbox('Sigara içiyor musunuz?', ['yes', 'no'])
scc = st.selectbox('Her gün kaloriyi kontrol ediyor musunuz?', ['yes', 'no'])
caec = st.selectbox('Yemek sıklığınız nedir?', ['Always', 'Frequently', 'Sometimes', 'no'])
calc = st.selectbox('Alkol tüketim sıklığınız nedir?', ['Always', 'Frequently', 'Sometimes', 'no'])
mtrans = st.selectbox('Ulaşım şekliniz nedir?', ['Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking'])

# Tahmin butonu
if st.button('Obezite Riskini Tahmin Et'):
    # Kullanıcı girdilerini ön işleme tabi tut
    user_data = {
        'Gender': 1 if gender == 'Male' else 0,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'family_history_with_overweight': 1 if family_history == 'yes' else 0,
        'FAVC': 1 if favc == 'yes' else 0,
        'SMOKE': 1 if smoke == 'yes' else 0,
        'SCC': 1 if scc == 'yes' else 0,
        'CAEC': caec,
        'CALC': calc,
        'MTRANS': mtrans
    }

    user_df = pd.DataFrame(user_data, index=[0])
    user_encoded = hot_encoder.transform(user_df[categorical_columns])
    user_encoded_df = pd.DataFrame(user_encoded, columns=feature_names)
    user_df = pd.concat([user_df.drop(columns=categorical_columns), user_encoded_df], axis=1)

    # Özellik adlarını kontrol et
    user_df = user_df[X_train.columns]

    # Ölçeklendirme ve dönüşüm
    user_df = scaler.transform(user_df)
    user_df = pt.transform(user_df)

    # Tahmin yap
    prediction = model.predict(user_df)
    prediction_label = label_encoder.inverse_transform(prediction)
    
    st.subheader(f"Tahmin Edilen Obezite Riski: {prediction_label[0]}")
