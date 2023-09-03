# Importar las librerías necesarias
import streamlit as st
import pandas as pd
import pickle
  
# Cargar el archivo .pkl que contiene el modelo de Random Forest
pkl_filename = "./Modelo_Random_Forest.pkl"
with open(pkl_filename, 'rb') as file:
    model_rf = pickle.load(file)

# Cargar el archivo local que contiene el dataset
ruta_archivo = './DATA/Data_Final_egresos_y_camillas.csv'
df = pd.read_csv(ruta_archivo, delimiter=';')

# Añadir un título y una descripción al inicio del formulario
st.title('Diagnóstico Inteligente: ')
st.markdown('Esta aplicación te permite la clasificar la mejor opcion segun la clase de establecimiento según los datos de ingreso de los pacientes en los establecimientos de salud.')
( ' Solo tienes que introducir los valores de las características categóricas y pulsar el botón "Diagnosticar". El modelo usa el algoritmo Random Forest, que se basa en la probabilidad condicional de cada clase.')
# Añadir una imagen o un logo relacionado con el tema o el proyecto
st.image('https://www.salud.gob.ec/wp-content/uploads/2017/03/logo-gabo-01.jpg', width=600)

# Agrupar los widgets de entrada en columnas o en un sidebar
sidebar = st.sidebar # Crear una variable para referirse al sidebar
sidebar.header('Introduce los valores de las características categóricas') # Añadir un encabezado al sidebar
provincia = sidebar.selectbox('Provincia de ubicación', df['Provincia de ubicación'].unique())
sector = sidebar.selectbox('Sector del establecimento', df['Sector del establecimento'].unique())
tipo = sidebar.selectbox('Tipo de establecimiento', df['Tipo de establecimiento'].unique())
mes =  sidebar.selectbox('Mes_Ingreso', df['Mes_Ingreso'].unique())
dias = sidebar.number_input('Dias_Estadia', min_value=0)
sexo = sidebar.selectbox('Sexo', df['Sexo'].unique())
edad = sidebar.number_input('Edad', min_value=0, max_value=120)
camas = sidebar.number_input('Total Dotación de camas del Establecimiento', min_value=0)
clasificacion = st.selectbox('Clasificacion_enfermedad', df['Clasificacion_enfermedad'].unique())

# Crear una variable que almacene los valores de entrada en un diccionario
input_data = {
    'Provincia de ubicación': provincia,
    'Sector del establecimento': sector,
    'Tipo de establecimiento': tipo,
    'Mes_Ingreso': mes,
    'Dias_Estadia': dias,
    'Sexo': sexo,
    'Edad': edad,
    'Total Dotación de camas del Establecimiento': camas,
    'Clasificacion_enfermedad': clasificacion
}

# Convertir el diccionario en un DataFrame
input_df = pd.DataFrame([input_data])

# Aplicar la misma transformación de one-hot encoding que usaste para entrenar el modelo
input_df = pd.get_dummies(input_df)

# Usar el atributo feature_names_in_ para obtener los nombres de las columnas que usó el modelo al entrenarse, y rellenar el DataFrame de entrada con ceros en las columnas faltantes.
cols_when_model_builds = model_rf.feature_names_in_
input_df = input_df.reindex(columns=cols_when_model_builds, fill_value=0)

# Crear un botón de acción para ejecutar el modelo
if st.button('Clasificar'):
    # Hacer la predicción con el DataFrame de entrada
    prediction_rf = model_rf.predict(input_df)[0]
    # Mostrar el resultado de la predicción
    st.success(f'La mejor opcion para la clase de establecimiento es: {prediction_rf}')
