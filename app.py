import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo
model = joblib.load('mimodelo.pkl')

# Cargar columnas del dataset para construir el formulario dinámicamente
df = pd.read_csv('Medicaldataset.csv')

# Quitar columnas objetivo
X = df.drop(['Result'], axis=1)
X = pd.get_dummies(X)  # Asegurarse que las columnas categóricas estén codificadas
input_cols = X.columns

# Título de la app
st.title("Demo de Predicción Médica")
st.write("Ingrese los datos del paciente para predecir el resultado (positivo o negativo).")

# Crear formulario de entrada
user_input = {}
for col in input_cols:
    if 'Age' in col:
        user_input[col] = st.slider('Edad', 0, 100, 25)
    elif 'Gender' in col:
        # Checkbox para gender_Female, gender_Male, etc.
        if 'gender_Female' in input_cols and 'gender_Male' in input_cols:
            gender = st.radio("Género", ['Femenino', 'Masculino'])
            user_input['gender_Female'] = 1 if gender == 'Femenino' else 0
            user_input['gender_Male'] = 1 if gender == 'Masculino' else 0
    elif 'Blood Pressure' in col:
        user_input[col] = st.selectbox("Presión Arterial", ['Normal', 'High', 'Low'])
    elif 'Cholesterol' in col:
        user_input[col] = st.selectbox("Colesterol", ['Normal', 'High'])
    elif 'Na_to_K' in col:
        user_input[col] = st.number_input("Relación Sodio/Potasio", min_value=0.0, max_value=50.0, value=15.0)
    elif col not in user_input:
        user_input[col] = st.number_input(col, value=0.0)

# Convertir a DataFrame
input_df = pd.DataFrame([user_input])

# Rellenar columnas faltantes con 0 (por ejemplo, si una categoría no se seleccionó)
for col in input_cols:
    if col not in input_df.columns:
        input_df[col] = 0

# Ordenar columnas según las del modelo
input_df = input_df[input_cols]

# Botón para predecir
if st.button("Predecir"):
    prediction = model.predict(input_df)[0]
    resultado = "Positivo" if prediction == 1 else "Negativo"
    st.success(f"Resultado predicho: {resultado}")
