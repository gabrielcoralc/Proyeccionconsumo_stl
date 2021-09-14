# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 15:58:45 2021

@author: Gabri
"""



import plotly.graph_objects as go

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp


from tensorflow_probability import distributions as tfd
from tensorflow_probability import sts
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from itertools import islice



import streamlit as st
tf.enable_v2_behavior()

def trend_(list_of_lists_sums,cut):
    """La funcion calcula la tendencia de una serie de tiempo y la divide entre el promedio de la misma serie, 
        retornando asi el valor porcentaje de variancion entre la pendiente y el promedio
    Parameters
    ----------
    list_of_lists_sums : list
        lista que contiene la serie de tiempo
    cut : int
        El numero de valores que se va a tomar de la lista
    Returns
    -------
    float:
        Retorna la variancion entre la pendiente y el promedio de la serie de tiempo
    """
    try:
        model = LinearRegression()
        return model.fit(np.arange(len(list_of_lists_sums[-cut:])).reshape(-1, 1),np.array(list_of_lists_sums[-cut:])).coef_[0]*100/np.average(np.array(list_of_lists_sums[-cut:]))
    except:
        return 0
    
def group_list(timesr,days):
    
    """La funcion separa en diferentes listas la serie de tiempo. el largo de cada lista esta definido por el valor
        de days y luego las suma, dando asi valores agregados como por ejemplo semanalmente, quincenal o mensual
    Parameters
    ----------
    timesr : list
        lista que contiene la serie de tiempo
    days : int
        El numero de dias que define los intervalos de agregacion
    Returns
    -------
    list:
        Retorna la serie de tiempo con los valores agregados segun un intervalo
    """
    list_of_lists = [ list(islice(reversed(timesr), i*days, (i+1)*days)) for i in range(int(len(timesr) / days ))]
    list_of_lists_sums=list(reversed([np.sum(x) for x in list_of_lists]))
    
    return list_of_lists_sums
    
def group_trends_(timesr,days,groups=1):
    
    """La funcion hace uso de las funciones group_list y la funcion trend_, para calcular multiples tendencias.
    Un ejemplo es, se tiene una serie de tiempo con informacion diaria, pero tu quieres conocer la tendencia de consumo mensual,
    de los ultimos 3 meses, por lo que se usaria group_list(timesr,30) para generar la serie de tiempo mensual y por ultimo
    ingresamos groups=1, para que calcule la tendencia de los ultimos 3 meses.
    La tendencia minima se puede realizar solamente con 3 valores, sino el algortimo lo valida en la funcion tred_ y retona un 0
    en el valor de la tendencia o si los consumos son todos ceros
    Parameters
    ----------
    timesr : list
        lista que contiene la serie de tiempo
    days : int
        El numero de dias que define los intervalos de agregacion
    groups : int
        Representa al numero de valores con los que se va a calcular la tendencia, si es mayor 1 retorna multiples tendencia,
        dando a entender que la primera tendencia calculada corresponde a 3 valores y las tendencia que continuan tendran un valor
        adicional con el cual fueron calculadas.
    Returns
    -------
    list:
        Retorna una lista de tendencias
    """
    list_of_lists_sums = group_list(timesr,days)
    trends = [np.round(trend_(list_of_lists_sums,i),2) for i in range(2,groups+2)]
    return trends


def build_model(observed_time_series):
    
    """Se crea el modelo con el cual se va a predecir el siguiente valor de una serie de tiempo.
        Utilizamos la libreria de tensorflow_probability para esto
    Parameters
    ----------
    observed_time_series : list
        lista que contiene la serie de tiempo
    Returns
    -------
    object:
        Retorna un objecto creado con la clase tensorflow_probability.sts"""
    
    day_of_month_effect = sts.SmoothSeasonal(
      period=30,frequency_multipliers= [1,2,3],
      observed_time_series=observed_time_series,
      name='day_of_month_effect')
    
    day_of_week_effect = sts.Seasonal(
      num_seasons=7,
      observed_time_series=observed_time_series,
      name='day_of_week_effect')
    
    autoregressive = sts.Autoregressive(
      order=1,
      observed_time_series=observed_time_series,
      name='autoregressive')    
    model = sts.Sum([day_of_week_effect,autoregressive,day_of_month_effect], observed_time_series=observed_time_series)
    
    return model

def train(energy_model,training):
    
    """Entrenamos el modelo creado apartir de la funcion build_model()
    Parameters
    ----------
    energy_model : obj
        modelo de tensor_flow
    training : list
        Serie de tiempo
    Returns
    -------
    object:
        elbo_loss_curve : list
            Retorna la curva de entrenamiento del modelo
        variational_posteriors : obj
            Objeto utilizado por tensor flow para hacer las predicciones, podemos considerarlo como nuestro modelo entrenado
        Retorna un objecto creado con la clase tensorflow_probability.sts"""
    # Allow external control of optimization to reduce test runtimes.
    num_variational_steps = 100 # @param { isTemplate: true}
    num_variational_steps = int(num_variational_steps)
    
    variational_posteriors = tfp.sts.build_factored_surrogate_posterior(
    model=energy_model)
    
    optimizer = tf.optimizers.Adam(learning_rate=.1)
    elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
    target_log_prob_fn=energy_model.joint_log_prob(
        observed_time_series=training),
    surrogate_posterior=variational_posteriors,
    optimizer=optimizer,
    num_steps=num_variational_steps)
    return elbo_loss_curve, variational_posteriors



def Massive_training_pred(Frts,data_out,data,step_forecast=30,z_score=1):
        
    """Entrenamos el multiples modelos creado apartir de la funcion build_model() y hacemos la prediccion de valores posibles a futuro
    Parameters
    ----------
    Frts : list
        Una lista que contiene los codigos que se van a buscar en el dataframe 
    data_out : dataframe
        Un dataframe con las series de tiempo que se van a comparar con las predicciones
    data : dataframe
        Un dataframe con las series de tiempo que se utilizaran para entrenar los modelos
    step_forecast: int
        numero de pasos en el futuro que se van a predecir
    z_score: int
        numero de desviaciones estandar en la prediccion para la deteccion de anomalias
    Returns
    -------
    object:
        data_pred_df : dataframe
            Retorna un  dataframe con algunos caculos realizados a partir de las predicciones hechar
        check_pred_df : dataframe
            dataframe con las observaciones pertinentes cuando no pudo realizar las prediccion en alguno 
            de los codigos seleccionados
        Adicionalmente se guardan los datos de entrenamiento y los modelos calculados en sus repectivas rutas"""
    Frts["FRTS"]=Frts["FRTS"].apply(lambda x: x.lower())
    frts=Frts["FRTS"].values.tolist()
    
    data_out[0]=data_out[0].apply(lambda x: x.lower())
    data[0]=data[0].apply(lambda x: x.lower())
    #step_forecast=30 #Numero de dias a predecir, en un futuro hay que validar que esto sea menor o igual a la cantidad 
                     #de dias reales a comparar con la prediccion
    
    data_pred=[]
    check_pred=[]
    
    #z_score=1 ##Rango para la deteccion de anomalia segun la desviacion estandar
    my_bar = st.progress(0)
    len_frts=len(frts)
    count_frts=0
    my_bar.progress(int(1))
    Fechas_Modelos=[]

    placeholder = st.empty()
    with st.spinner('Working on models and predictions...'):
        for Frt in frts:

            placeholder.text("Tiempo estimado de ejecucion %f minutos"%(float((len_frts-count_frts)*32/60)))
            observation=""
            try:
                real=data_out[data_out[0]==Frt].values[0][2:]
            except:
                observation=observation + "Usuario no se encuentra en la tabla de datos reales, "
            try:
                training = data[data[0]==Frt].values.tolist()[0][3:]
                training=np.array(training)
            except:
                observation=observation + "Usuario no se encuentra en la tabla de datos de entrenamiento, "            
    
            try:
                energy_model = build_model(training)
        
                # Build the variational surrogate posteriors `qs`.
        
        
                elbo_loss_curve, surrogate_posterior = train(energy_model,training)
                #@title Minimize the variational loss.
        
                # Draw samples from the variational posterior.
        
        
        
                q_samples_energy_ = surrogate_posterior.sample(50)
                energy_forecast_dist = tfp.sts.forecast(
                    energy_model,
                    observed_time_series=training,
                    parameter_samples=q_samples_energy_,
                    num_steps_forecast=step_forecast)
        
        
                E_forecast_mean, E_forecast_scale, E_forecast_samples = (
                    energy_forecast_dist.mean().numpy()[..., 0],
                    energy_forecast_dist.stddev().numpy()[..., 0],
                    energy_forecast_dist.sample(10).numpy()[..., 0])
                
                
        
                filename = 'Models/' + Frt + '_qsamp.sav'
                joblib.dump(q_samples_energy_, filename)
                filename = 'Training/' + Frt + '_training.sav'
                joblib.dump(training, filename)
        
                Num_anomalies=np.sum((real-E_forecast_mean[:real.shape[-1]])/E_forecast_scale[:real.shape[-1]]<-z_score)
                Dif_pred=np.round(np.sum(real-E_forecast_mean[:real.shape[-1]])*100/np.sum(E_forecast_mean[:real.shape[-1]]),2)
                timesr=np.append(training,real)
                tendecia_mensual=group_trends_(timesr,30,3)
                tendecia_trimestral=group_trends_(timesr,90,1)
                promedio_mensual=np.round(np.average(np.array(group_list(timesr,30))),2)
                data_pred.append([Frt,promedio_mensual,Dif_pred,Num_anomalies]+tendecia_mensual+tendecia_trimestral)
                Fechas_Modelos.append([Frt,data[data[0]==Frt].values.tolist()[0][1],data[data[0]==Frt].values.tolist()[0][2]])
            except:
                observation = observation +"No fue posible generar modelo de prediccion para el usuario, verificar datos de entrenamiento"
                check_pred.append([Frt,observation])
            count_frts=count_frts+1
            my_bar.progress(int(count_frts*100/len_frts))
    placeholder.empty()
    data_pred_df=pd.DataFrame(data_pred,columns=["CODIGO_SIC","Consumo_Promedio","Diferencia_Real_Prediccion_%","Cantidad_Anomalias","Tendencia_mes_2","Tendencia_mes_3","Tendencia_mes_4","Tendencia_trimes_2"])
    check_pred_df=pd.DataFrame(check_pred,columns=["CODIGO_SIC","Observacion"])
    Fechas_Modelos_df=pd.DataFrame(Fechas_Modelos,columns=["CODIGO_SIC","Fecha_Inicial","Fecha_Final"])
    Fechas_Modelos_df.Fecha_Final=pd.to_datetime(Fechas_Modelos_df.Fecha_Final)
    Fechas_Modelos_df.Fecha_Inicial=pd.to_datetime(Fechas_Modelos_df.Fecha_Inicial)
    try:   
        Fechas_Modelos_prov=pd.read_excel("Results/Fechas_modelos.xlsx")
        Fechas_Modelos_df=Fechas_Modelos_df.append(Fechas_Modelos_prov,ignore_index=True)
        Fechas_Modelos_df.drop_duplicates(subset=["CODIGO_SIC"],inplace=True)
    except:
        st.write("Se ha creado el archivo Fechas_modelos.xlsx por primera vez")
    Fechas_Modelos_df.to_excel("Results/Fechas_modelos.xlsx",index=False)
    data_pred_df.to_excel("Results/Resultados_prediccion.xlsx",index=False)
    check_pred_df.to_excel("Results/Frts_Revisar.xlsx",index=False)
    return data_pred_df,check_pred_df

def Massive_pred(frts,data_out,step_forecast=30,z_score=1):
    
    """Predecimos n pasos posibles valores en el futuro de modelos previamente entrenados
    Parameters
    ----------
    Frts : list
        Una lista que contiene los codigos que se van a buscar en el dataframe 
    data_out : dataframe
        Un dataframe con las series de tiempo que se van a comparar con las predicciones
    step_forecast: int
        numero de pasos en el futuro que se van a predecir
    z_score: int
        numero de desviaciones estandar en la prediccion para la deteccion de anomalias
    Returns
    -------
    object:
        data_pred_df : dataframe
            Retorna un  dataframe con algunos caculos realizados a partir de las predicciones hechar
        check_pred_df : dataframe
            dataframe con las observaciones pertinentes cuando no pudo realizar las prediccion en alguno 
            de los codigos seleccionados"""
    data_out[0]=data_out[0].apply(lambda x: x.lower())
    
    #step_forecast=30 #Numero de dias a predecir, en un futuro hay que validar que esto sea menor o igual a la cantidad 
                     #de dias reales a comparar con la prediccion
    
    data_pred=[]
    check_pred=[]
    
    my_bar = st.progress(0)
    len_frts=len(frts)
    count_frts=0
    my_bar.progress(int(1))
    placeholder1=st.empty()
    with st.spinner('Working on predictions...'):
        for Frt in frts:
            placeholder1.text("Tiempo estimado de ejecucion %2.f minutos"%(float((len_frts-count_frts)*32/60)))
            observation=""
            try:
                real=data_out[data_out[0]==Frt].values[0][2:]
            except:
                observation=observation + "Usuario no se encuentra en la tabla de datos reales, "
            training = joblib.load('Training/' + Frt + '_training.sav')
            
            try:
                energy_model = build_model(training)
        
        
                # Draw samples from the variational posterior.
        
        
                q_samples_energy_=joblib.load('Models/' + Frt + '_qsamp.sav')

                energy_forecast_dist = tfp.sts.forecast(
                    energy_model,
                    observed_time_series=training,
                    parameter_samples=q_samples_energy_,
                    num_steps_forecast=step_forecast)
        
        
                E_forecast_mean, E_forecast_scale, E_forecast_samples = (
                    energy_forecast_dist.mean().numpy()[..., 0],
                    energy_forecast_dist.stddev().numpy()[..., 0],
                    energy_forecast_dist.sample(10).numpy()[..., 0])
        
                
        
                Num_anomalies=np.sum((real-E_forecast_mean[:real.shape[-1]])/E_forecast_scale[:real.shape[-1]]<-z_score)
                Dif_pred=np.round(np.sum(real-E_forecast_mean[:real.shape[-1]])*100/np.sum(E_forecast_mean[:real.shape[-1]]),2)
                timesr=np.append(training,real)
                tendecia_mensual=group_trends_(timesr,30,3)
                tendecia_trimestral=group_trends_(timesr,90,1)
                promedio_mensual=np.round(np.average(np.array(group_list(timesr,30))),2)
                data_pred.append([Frt,promedio_mensual,Dif_pred,Num_anomalies]+tendecia_mensual+tendecia_trimestral)
                
            except:
                observation=observation + "No fue posible generar prediccion, revisar datos y modelo del usuario"
                check_pred.append([Frt,observation])
            count_frts=count_frts+1
            my_bar.progress(int(count_frts*100/len_frts))
    placeholder1.empty()
    data_pred_df=pd.DataFrame(data_pred,columns=["CODIGO_SIC","Consumo_Promedio","Diferencia_Real_Prediccion_%","Cantidad_Anomalias","Tendencia_mes_2","Tendencia_mes_3","Tendencia_mes_4","Tendencia_trimes_2"])
    check_pred_df=pd.DataFrame(check_pred,columns=["CODIGO_SIC","Observacion"])
    return data_pred_df,check_pred_df




def plot_forecast(frt, data_out,step_forecast=30,z_score=1):
    
    """Entrega una figura interactiva creada en plotly de los valores de entrenamiento,
        prediccion y anomalias detectadas
    ----------
    Frts : list
        Una lista que contiene los codigos que se van a buscar en el dataframe 
    data_out : dataframe
        Un dataframe con las series de tiempo que se van a comparar con las predicciones
    step_forecast: int
        numero de pasos en el futuro que se van a predecir
    z_score: int
        numero de desviaciones estandar en la prediccion para la deteccion de anomalias
    Returns
    -------
    object:
        fig : Object
            Una figura creada a partir de plotly
        forecast_mean : list
            lista con la media de los valores que obtuvo la prediccion
        upper : list
            lista con el limite superior de los valores que obtuvo la prediccion
        lower : list
            lista con el limite inferior de los valores que obtuvo la prediccion
            de los codigos seleccionados"""
    try:
        data_out[0]=data_out[0].apply(lambda x: x.lower())#Esta es la data de los valores reales que se compara con la prediccion
        real=data_out[data_out[0]==frt].values[0][2:]
        training = joblib.load('Training/' + frt + '_training.sav')
        observed_time_series=training
        energy_model = build_model(training)
    
    
        # Draw samples from the variational posterior.
    
    
        q_samples_energy_=joblib.load('Models/' + frt + '_qsamp.sav')
    
        energy_forecast_dist = tfp.sts.forecast(
            energy_model,
            observed_time_series=training,
            parameter_samples=q_samples_energy_,
            num_steps_forecast=step_forecast)
    
    
        forecast_mean, forecast_scale, forecast_samples = (
            energy_forecast_dist.mean().numpy()[..., 0],
            energy_forecast_dist.stddev().numpy()[..., 0],
            energy_forecast_dist.sample(10).numpy()[..., 0])
        
    
        
        fig = go.Figure()
        
        observed_time_series=np.append(observed_time_series,real[0])
    
        num_steps = observed_time_series.shape[-1]
        num_steps_forecast = forecast_mean.shape[-1]
        num_steps_train = num_steps
        num_real_steps=real.shape[-1]
        
        initial_date=pd.read_excel(r"Results\Fechas_Modelos.xlsx")
        initial_date=initial_date[initial_date.CODIGO_SIC==frt].values[0][1]
        
        observed_time_series_step=pd.date_range(initial_date,periods=num_steps)
        forecast_steps=pd.date_range(observed_time_series_step[-1],periods=num_steps_forecast)
        real_steps=pd.date_range(observed_time_series_step[-1],periods=num_real_steps)
    
        c1, c2 = 'rgb(0.12, 0.47, 0.71)', 'rgb(1.0, 0.5, 0.05)'
        fig.add_trace(go.Scatter(
        x=observed_time_series_step, y=observed_time_series,
        line_color=c1,
        name='ground truth',))
    
    
    
        #plt.plot(forecast_steps, forecast_samples.T, lw=1, color=c2, alpha=0.1)
        
        upper=forecast_mean - z_score * forecast_scale
        lower=forecast_mean + z_score * forecast_scale
        
        fig.add_trace(go.Scatter(
        x=forecast_steps.tolist()+forecast_steps.tolist()[::-1],
        y=upper.tolist()+lower.tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
        name='std',
        ))
        
        fig.add_trace(go.Scatter(
        x=forecast_steps, y=forecast_mean,
        line_color=c2, line = dict( width=2, dash='dash'),
        name='forecast',))
            
        fig.add_trace(go.Scatter(
        x=real_steps, y=real,
        line_color='blue', line = dict( width=2, dash='dot'),
        name='real',))
        
    
        
        anomalies=(real-forecast_mean[:real.shape[-1]])/forecast_scale[:real.shape[-1]]<-z_score
        
        fig.add_trace((go.Scatter(mode="markers", x=real_steps[anomalies], y=real[anomalies], marker_symbol='x',
                               marker_line_color="midnightblue", marker_color="red",
                               marker_line_width=2, marker_size=8,name="Anomalies")))
        fig.update_layout(title='Prediccion de consumo '+frt, autosize=False,
                      width=800, height=400,
                      margin=dict(l=40, r=40, b=40, t=40))
        
        return fig,forecast_mean,upper,lower
    except:
        st.warning("El Frt seleccionado no se encuentra en el archivo de consumos actuales, favor revisar")

@st.cache
def check_dtype_data(df):
    
    """La funcion verifica que el dataframe cargado cumpla ciertas caracteristicas para que sea valid
    ----------
    df : dataframe
        dataframe con la informacion de las series de tiempo de cada codigo a utilizar

    Returns
    -------
    boolean:
        Retorna True o False si el dataframe cumple ciertas caracteristicas
    """
    cols=df.columns
    types=df.dtypes
    try:
        if (types[cols[0]].name == 'int64') & (types[cols[1]].name == 'object') & (types[cols[2]].name == 'object') & (types[cols[3]].name == 'datetime64[ns]'):
            check=0
            num_cols=4
            while (num_cols<len(cols)):
                if (types[cols[num_cols]].name == 'float64'):
                     num_cols=num_cols+1
                else:
                    check=1
                    break
            if check==0:
                return True
            else:
                return False
        else:
            return False
    except:
        return False
                            
@st.cache
def check_dtype_frts(df):
    
    """La funcion verifica que el dataframe cargado cumpla ciertas caracteristicas para que sea valid
    ----------
    df : dataframe
        dataframe con la informacion de los codigos que se quiere utilizar

    Returns
    -------
    boolean:
        Retorna True o False si el dataframe cumple ciertas caracteristicas
    """
    cols=df.columns
    types=df.dtypes
    try: 
        if (len(cols)==1) & (cols[0]=='FRTS') & (types[cols[0]].name == 'object'):
            return True
        else:
            return False
    except:
        return False
    
    
def load_training(uploaded_files_Training):
    
    """La funcion verifica el orden que se deberian unir los archivos con las series de tiempo cargados
        de esta manera se asegura que a pesar de que se suban archivos en desorden, la informacion este ordenada
    ----------
    uploaded_files_Training : object
        Un objeto creado por streamlit, con la informacion de todos los archivos creados

    Returns
    -------
    data : dataframe
        Retorna un dataframe con toda la informacion ordenada por fecha
    warnings : object
        Un objecto de streamlit que dispara una advertencia cuando un archivo esta mal subido
    """
    data=pd.DataFrame()
    sort_date=[]
    check_files=0
    count_files_training=0
    sort_date_df=pd.DataFrame(columns=["File","Fecha"])
    warnings=[]
    for uploaded_file_training in uploaded_files_Training:
        if check_dtype_data(pd.read_excel(uploaded_file_training)):    
            
            datedf=pd.read_excel(uploaded_file_training)
            datedf['Fecha']=pd.to_datetime(datedf['Fecha'])
            prov_date=datedf['Fecha'].values.tolist()[0]
            
            sort_date.append([uploaded_file_training,prov_date])
        else:
            warnings+=["El archivo  " + uploaded_file_training.name + " no cumple con el formato, favor revisar los archivos de ejemplo"]
            check_files=1
    if check_files==0:
        sort_date_df=pd.DataFrame(sort_date,columns=["File","Fecha"])
        sort_date_df['Fecha']=pd.to_datetime(sort_date_df['Fecha'])        
        sort_date_df.sort_values(by=['Fecha'],inplace=True)
        for uploaded_file_training in sort_date_df['File']:
                if count_files_training != 0:
                    
                    prov_data=pd.read_excel(uploaded_file_training)
                    Fecha_Final=prov_data['Fecha'].values.tolist()[0]
                    prov_data=prov_data.drop(columns=["Unnamed: 0",1,'Fecha']).drop_duplicates()
                    data=pd.merge(left= data, right=prov_data,how="outer",left_on=0,right_on=0)
                    st.write(uploaded_file_training.name)
                if count_files_training==0:
                    
                    data=pd.read_excel(uploaded_file_training)
                    Fecha_inicial=data['Fecha'].values.tolist()[0]
                    Fecha_Final=Fecha_inicial
                    data=data.drop(columns=["Unnamed: 0",1,'Fecha']).drop_duplicates()
                    count_files_training=count_files_training+1
                    st.write(uploaded_file_training.name)
    
        data.insert(1,"Fecha_Inicial",Fecha_inicial)
        data.insert(2,"Fecha_Final",Fecha_Final)
        data.Fecha_Inicial=pd.to_datetime(data.Fecha_Inicial)
        data.Fecha_Final=pd.to_datetime(data.Fecha_Final)
        data.fillna(0,inplace=True)
        
    return data,warnings

def Massive_trends(data):
    
    """La funcion realiza calculos de tendecias sobre la informacion cargada sin tener en cuenta las predicciones
    ----------
    data : dataframe
        dataframe que contiene todas las series de tiempo de los codigos

    Returns
    -------
    data_pred_df : dataframe
        Retorna un dataframe con los calculos de tendencias de todos los codigos
    """
    
    data.fillna(0,inplace=True)
    data[0]=data[0].apply(lambda x: x.lower())
    #step_forecast=30 #Numero de dias a predecir, en un futuro hay que validar que esto sea menor o igual a la cantidad 
                     #de dias reales a comparar con la prediccion
    frts=data[0].values.tolist()
    data_pred=[]

    for Frt in frts:

        training = data[data[0]==Frt].values.tolist()[0][3:]
        training=np.array(training)
        tendecia_mensual=group_trends_(training,30,11)
        tendecia_mensual=[tendecia_mensual[i] for i in [1,4,9]]
        tendecia_trimestral=group_trends_(training,90,1)
        promedio_mensual=np.round(np.average(np.array(group_list(training,30))),2)
        data_pred.append([Frt,promedio_mensual]+tendecia_mensual+tendecia_trimestral)
    data_pred_df=pd.DataFrame(data_pred,columns=["CODIGO_SIC","Consumo_Promedio","Tendencia_mes_3","Tendencia_mes_6","Tendencia_mes_12","Tendencia_trimestral"])
    data_pred_df.fillna(0,inplace=True)
    return data_pred_df