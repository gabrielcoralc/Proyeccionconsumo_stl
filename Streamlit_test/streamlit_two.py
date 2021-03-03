# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 20:36:11 2021

@author: Gabri
"""

import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import pandas as pd
from Functions import Utilities
import base64
from os import listdir
from os.path import isfile, join

def filedownload(df,filename):
    filename=filename + ".csv"
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download=%s>Download %s File</a>'%(filename,filename)
    return href

st.set_option('deprecation.showPyplotGlobalUse', False)

###HEADER
st.title("""Mi Primera app de proyeccion de consumos""")
st.write("Como archivo principal para todos los procesos se requiere la informacion de consumos actual con la cual se comparara las predicciones")
data_out=pd.DataFrame()
###CARGA DE ARCHIVO CONSUMOS REALES A COMPARAR CON LA PREDICCION
uploaded_file_real = st.file_uploader('Load real energy consume to compare against the prediction',type=['xlsx'])
if uploaded_file_real is not None:
    if Utilities.check_dtype_data(pd.read_excel(uploaded_file_real)):
        data_out=pd.read_excel(uploaded_file_real).drop(columns=["Unnamed: 0",1]).drop_duplicates()
        st.dataframe(data_out.head(10))
    else:
        st.warning("El archivo no cumple con el formato, favor revisar los archivos de ejemplo")

    
example_data=pd.read_excel("Examples/Consumo_Mes_dia_Tx2_aenc01.xlsx")
example_frts=pd.read_excel("Examples/FRTS.xlsx")
st.write('Descargar los archivos de ejemplo para saber como se debe cargar la informacion. Para el caso del archivo referente a los datos de consumo, no hay limite en los datos de consumo que van por columnas, pero el archivo debe seguir con la misma estructura que el ejemplo.')
st.markdown(filedownload(example_data,"Ejemplo_datos_consumo_energia"), unsafe_allow_html=True)
st.markdown(filedownload(example_frts,"Ejemplo_listado_FRTS"), unsafe_allow_html=True)

##PRIMERA SECCION
st.write("""## 1. Training models and forecasting (Massive)
                 """)
st.write("Utilizar esta funcion para volver a calcular los modelos de prediccion de las fronteras deseadas")

Frts=pd.DataFrame()
#CARGA DE LISTADO DE CODIGOS A REALIZAR MODELOS
uploaded_files_Frts = st.file_uploader('Load Frts to predict',type=['xlsx'])
if uploaded_files_Frts is not None:
    if Utilities.check_dtype_frts(pd.read_excel(uploaded_files_Frts)):
        Frts=pd.read_excel(uploaded_files_Frts)
        st.dataframe(Frts)
    else:
        st.warning("El archivo no cumple con el formato, favor revisar los archivos de ejemplo")
        
##CARGA DATA DE ENTRENAMIENTO    
uploaded_files_Training = st.file_uploader('Load training data to predict',accept_multiple_files=1,type=['xlsx'])
count_files_training=0
data=pd.DataFrame()
if uploaded_files_Training is not None:
    for uploaded_file_training in uploaded_files_Training:
        if Utilities.check_dtype_data(pd.read_excel(uploaded_file_training)):
            if count_files_training != 0:
                prov_data=pd.read_excel(uploaded_file_training).drop(columns=["Unnamed: 0",1]).drop_duplicates()
                data=pd.merge(left= data, right=prov_data,how="outer",left_on=0,right_on=0)
                st.write(uploaded_file_training.name)
            if count_files_training==0:
                data=pd.read_excel(uploaded_file_training).drop(columns=["Unnamed: 0",1]).drop_duplicates()
                count_files_training=count_files_training+1
                st.write(uploaded_file_training.name)
        else:
            st.warning("El archivo  " + uploaded_file_training.name + " no cumple con el formato, favor revisar los archivos de ejemplo")

if len(data) != 0:
    st.dataframe(data.head(10))    
st.write("If you are using multiple files for the training, please upload thes files in ascending order according to date")        


##SLIDERS PARA DECIDIR PARAMENTROS DE PREDICCION
if (len(data) != 0) & (len(data_out) != 0) & (len(Frts) != 0):
    colss=data_out.shape[1]-1
    step_forecast = st.slider('How many step do you want to forescast?', colss, 90, colss)
    z_score = st.slider('Number of standar desviation for the anomalies analysis', 1, 4, 1)

if st.button('Start training models and predicction'):
    if (len(data) != 0) & (len(data_out) != 0) & (len(Frts) != 0):
        
        data_pred,check_pred = Utilities.Massive_training_pred(Frts,data_out,data,step_forecast,z_score)
        st.write('Prediction complete, check the links below to download the results')
        st.markdown(filedownload(data_pred,"Resultados_Predicion"), unsafe_allow_html=True)
        st.markdown(filedownload(check_pred,"Frts_Revisar"), unsafe_allow_html=True)

    else:
        st.warning("Please check that all neccesary files have been upload")
        
st.write("""## 2. Forecasting (Massive) for models already trained
                 """)

onlyfiles = [f for f in listdir("Models/") if isfile(join("Models/", f))]
onlyfiles= [f[0:8] for f in onlyfiles]
        
        
Frts_trained=st.multiselect("Select Frts to forecast",onlyfiles)      


    
if (len(data_out) != 0) & (len(Frts_trained) != 0):
    colss=data_out.shape[1]-1
    step_forecast2 = st.slider('How many steps in the future do you want to forescast?', colss, 90, colss)
    z_score2 = st.slider('Number of standar desviations for the anomalies analysis', 1, 4, 1)
    
if st.button('Start predicction'):
    if (len(data_out) != 0) & (len(Frts_trained) != 0):
        
        data_pred2,check_pred2= Utilities.Massive_pred(Frts_trained,data_out,step_forecast2,z_score2)
        st.write('Prediction complete, check the links below to download the results')
        st.markdown(filedownload(data_pred2,"Resultados_Predicion"), unsafe_allow_html=True)
        st.markdown(filedownload(check_pred2,"Frts_Revisar"), unsafe_allow_html=True)

    else:
        st.warning("Please check that all neccesary files have been upload")
        
st.write("""## 3. Plot training, Forecasting and actual data
                 """)
                 
Frt_trained=st.selectbox("Select Frt to plot",onlyfiles) 
if (len(data_out) != 0) :
    colss=data_out.shape[1]-1
    step_forecast3 = st.slider('How many steps in the future do you want to forescast? ', colss, 90, colss)
    z_score3 = st.slider('Number of standar desviations for the anomalies analysis ', 1, 4, 1)

if st.button('Show Plot'):
    if (len(data_out) != 0):
        
        Utilities.plot_forecast(Frt_trained, data_out,step_forecast3,z_score3)
    else:
        st.warning("Please check that all neccesary files have been upload or check if you select a valid Frt")
        


