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


def actual_data():
    ###HEADER
    #################################
    st.title("""Mi Primera app de proyeccion de consumos""")
    st.write("Como archivo principal para todos los procesos se requiere la informacion de consumos actual con la cual se comparara las predicciones")
    data_out=pd.DataFrame()
    ###CARGA DE ARCHIVO CONSUMOS REALES A COMPARAR CON LA PREDICCION
    uploaded_file_real = st.file_uploader('Load real energy consume to compare against the prediction',type=['xlsx'])
    if uploaded_file_real is not None:
        if Utilities.check_dtype_data(pd.read_excel(uploaded_file_real)):
            data_out=pd.read_excel(uploaded_file_real).drop(columns=["Unnamed: 0",1]).drop_duplicates()
        else:
            st.warning("El archivo no cumple con el formato, favor revisar los archivos de ejemplo")
    example_data=pd.read_excel("Examples/Consumo_Mes_dia_TxR_aenc01.xlsx")
    example_frts=pd.read_excel("Examples/FRTS.xlsx")
    st.write('Descargar los archivos de ejemplo para saber como se debe cargar la informacion. Para el caso del archivo referente a los datos de consumo, no hay limite en los datos de consumo que van por columnas, pero el archivo debe seguir la misma estructura que el ejemplo.')
    st.markdown(filedownload(example_data,"Ejemplo_datos_consumo_energia"), unsafe_allow_html=True)
    st.markdown(filedownload(example_frts,"Ejemplo_listado_FRTS"), unsafe_allow_html=True)
    return data_out


def training_massive(data_out):
    #######################################
    
    ##PRIMERA SECCION
    #######################################
    st.write("""## 1. Training models and forecasting (Massive)
                     """)
    st.write("Utilizar esta funcion para volver a calcular los modelos de prediccion de las fronteras deseadas")
    
    Frts=pd.DataFrame()
    
    
    #CARGA DE LISTADO DE CODIGOS A REALIZAR MODELOS
    uploaded_files_Frts = st.file_uploader('Load Frts to predict',type=['xlsx'])
    
    ##CARGA DATA DE ENTRENAMIENTO    
    uploaded_files_Training = st.file_uploader('Load training data to predict',accept_multiple_files=1,type=['xlsx'])
    
    ##Data a retornar
    data_pred,check_pred = None,None
    
    ##SLIDERS PARA DECIDIR PARAMENTROS DE PREDICCION
    if (len(data_out) != 0):
        colss=data_out.shape[1]-2
        step_forecast = st.slider('How many step do you want to forescast?', colss, 90, colss)
        z_score = st.slider('Number of standar desviation for the anomalies analysis', 1, 4, 1)
    
    if st.button('Start training models and predicction'):
        #CARGA DE LISTADO DE CODIGOS A REALIZAR MODELOS
        if uploaded_files_Frts is not None:
            if Utilities.check_dtype_frts(pd.read_excel(uploaded_files_Frts)):
                Frts=pd.read_excel(uploaded_files_Frts)
            else:
                st.warning("El archivo no cumple con el formato, favor revisar los archivos de ejemplo")
            
        ##CARGA DATA DE ENTRENAMIENTO    
        if uploaded_files_Training:
            data,warnings = Utilities.load_training(uploaded_files_Training)
            if len(warnings)!=0:
                for w in warnings:
                    st.warning(w)
        
        
        if (len(data) != 0) & (len(data_out) != 0) & (len(Frts) != 0):
            st.success('All files have uploaded successfully')
            data_pred,check_pred = Utilities.Massive_training_pred(Frts,data_out,data,step_forecast,z_score)
        else:
            st.warning("Please check that all neccesary files have been upload")
            
            
    return data_pred,check_pred
    


def forecasting(data_out):

#######################################

#SEGUNDA SECCION
#######################################
    st.write("""## 2. Forecasting (Massive) for models already trained
                     """)
    
    onlyfiles = [f for f in listdir("Models/") if isfile(join("Models/", f))]
    onlyfiles= [f.split('_')[0] for f in onlyfiles]
    
    if st.button('Add All option'):
        Frts_trained=st.multiselect("Select Frts to forecast",onlyfiles,default=onlyfiles,key="Frts")
    else:
        Frts_trained=st.multiselect("Select Frts to forecast",onlyfiles,key="Frts")   
    
    
    
    if (len(data_out) != 0) & (len(Frts_trained) != 0):
        Models_dates=pd.read_excel("Results/Fechas_Modelos.xlsx")
        Models_dates=Models_dates[Models_dates.CODIGO_SIC.isin(Frts_trained)]
        st.write("Tener en cuenta las fechas con las que se creo el modelo")
        st.dataframe(Models_dates)
        colss=data_out.shape[1]-2
        step_forecast2 = st.slider('How many steps in the future do you want to forescast?', colss, 90, colss)
        z_score2 = st.slider('Number of standar desviations for the anomalies analysis', 1, 4, 1)
     
    data_pred2,check_pred2= None, None    
    if st.button('Start predicction'):
        if (len(data_out) != 0) & (len(Frts_trained) != 0):
            
            data_pred2,check_pred2= Utilities.Massive_pred(Frts_trained,data_out,step_forecast2,z_score2)

    
        else:
            st.warning("Please check that all neccesary files have been upload")
            
    return data_pred2,check_pred2


def plotting(data_out):
    #######################################
    
    #TERCERA SECCION
    #######################################
    st.write("""## 3. Plot training, Forecasting and actual data
                     """)
    
    data_forecast=None
    onlyfiles = [f for f in listdir("Models/") if isfile(join("Models/", f))]
    onlyfiles= [f.split('_')[0] for f in onlyfiles]
    Frt_trained=st.selectbox("Select Frt to plot",onlyfiles) 
    if (len(data_out) != 0) :
        colss=data_out.shape[1]-2
        step_forecast3 = st.slider('How many steps in the future do you want to forescast? ', colss, 90, colss)
        z_score3 = st.slider('Number of standar desviations for the anomalies analysis ', 1, 4, 1)
    
    if st.button('Show Plot'):
        if (len(data_out) != 0):
    
            fig,forecast_mean,upper,lower=Utilities.plot_forecast(Frt_trained, data_out,step_forecast3,z_score3)
            st.plotly_chart(fig)
            data_forecast=pd.DataFrame([["Prediccion_Media"]+forecast_mean.tolist(),["Prediccion_Inferior"]+upper.tolist(),["Prediccion_Superior"]+lower.tolist()])
        else:
            st.warning("Please check that all neccesary files have been upload or check if you select a valid Frt")
    
    return data_forecast
    #######################################

def trends_massive():
    #######################################
    
    ##CUARTA SECCION
    #######################################
    st.write("""## 4. Calculate trends for all Frts
             """)
    st.write("Calcula las tendencias en diferentes rangos de 30 dias para todos los Frts")
    
    ##CARGA DATA DE ENTRENAMIENTO    
    uploaded_files_Trend = st.file_uploader('Load training data to calculate trends',accept_multiple_files=1,type=['xlsx'])   
    ##Data a retornar
    data_pred=None 
    if uploaded_files_Trend:
        if st.button('Get trends'):
            ##CARGA DATA DE ENTRENAMIENTO    
            
            data,warnings = Utilities.load_training(uploaded_files_Trend)
            if len(warnings)!=0:
                for w in warnings:
                    st.warning(w)
            if (len(data) != 0):
                st.success('All files have uploaded successfully')
                data_pred= Utilities.Massive_trends(data)
            else:
                st.warning("Please check that all neccesary files have been upload")
    
    return data_pred




def main():
    
    data_out=actual_data()
    data_pred,check_pred = training_massive(data_out)
    if data_pred is not None and check_pred is not None:
        st.write('Prediction complete, check the links below to download the results')
        st.markdown(filedownload(data_pred,"Resultados_Predicion"), unsafe_allow_html=True)
        st.markdown(filedownload(check_pred,"Frts_Revisar"), unsafe_allow_html=True)
    data_pred2,check_pred2 = forecasting(data_out)
    if data_pred2 is not None and check_pred2 is not None:
        st.write('Prediction complete, check the links below to download the results')
        st.markdown(filedownload(data_pred2,"Resultados_Predicion"), unsafe_allow_html=True)
        st.markdown(filedownload(check_pred2,"Frts_Revisar"), unsafe_allow_html=True)
    data_forecast=plotting(data_out)
    if data_forecast is not None:
        st.write('Forecast calculation complete, check the links below to download the results')
        st.markdown(filedownload(data_forecast,"Resultados_predicciones"), unsafe_allow_html=True)
    data_trend=trends_massive()
    if data_trend is not None:
        st.write('Trends calculation complete, check the links below to download the results')
        st.markdown(filedownload(data_trend,"Resultados_Tendencias"), unsafe_allow_html=True)
        
st.set_option('deprecation.showPyplotGlobalUse', False)
main()

