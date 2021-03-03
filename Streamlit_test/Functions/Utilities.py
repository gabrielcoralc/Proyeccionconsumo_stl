# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 15:58:45 2021

@author: Gabri
"""


from matplotlib import pylab as plt
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

import time

import streamlit as st
tf.enable_v2_behavior()

def trend_(list_of_lists_sums,cut):
    try:
        model = LinearRegression()
        return model.fit(np.arange(len(list_of_lists_sums[-cut:])).reshape(-1, 1),np.array(list_of_lists_sums[-cut:])).coef_[0]*100/np.average(np.array(list_of_lists_sums[-cut:]))
    except:
        return 0
    
def group_list(timesr,days):
    list_of_lists = [ list(islice(reversed(timesr), i*days, (i+1)*days)) for i in range(int(len(timesr) / days ))]
    list_of_lists_sums=list(reversed([np.sum(x) for x in list_of_lists]))
    return list_of_lists_sums
    
def group_trends_(timesr,days,groups=1):
    list_of_lists_sums = group_list(timesr,days)
    trends = [np.round(trend_(list_of_lists_sums,i),2) for i in range(2,groups+2)]
    return trends


def build_model(observed_time_series):
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
        
        return st.plotly_chart(fig)
    except:
        st.warning("El Frt seleccionado no se encuentra en el archivo de consumos actuales, favor revisar")

def check_dtype_data(df):

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
                            

def check_dtype_frts(df):

    cols=df.columns
    types=df.dtypes
    try: 
        if (len(cols)==1) & (cols[0]=='FRTS') & (types[cols[0]].name == 'object'):
            return True
        else:
            return False
    except:
        return False