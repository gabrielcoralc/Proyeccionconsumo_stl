# Proyeccionconsumo_stl
Proyecto para realizar predicciones de consumos y deteccion de anomalias a series de tiempo con informacion diara.

## Librerias 
- Streamlit: Se utiliza la libreria de streamlit para la generacion de una pagina web, dado a su facilidad en implementacion
- Plotly: Plotly genera graficos interactivos y brinda muchas opciones al usuario final
- tensorflow_probability: La libreria de estadistica de tensorflow, nos permite crear de manera sencilla modelos de alta precision para prediccion en series de tiempo.
- sklearn: Se utiliza esta libreria para el calculo de tendencias en series de tiempo con la clase LinearRegression
- numpy
- pandas
- joblib
- base64
- os

## Funcionamiento

Es importante tener instaladas previamente todas las librerias necesarias para evitar cualquier inconveniente al ejecutar el codigo (Se recomienda la creacion de un ambiente virtual).
Una vez instaladas las librerias se tiene que ingresar a la ventana del terminal y ubicarce en la direccion donde se encuentran los archivos, ya ubicado en la ruta se debe ejecutar el siguiente comando:
```Python
streamlit run streamlit_first.py
```
Una vez ejecutado sin errores, se deberia generar el servicio de manera local en la siguiente direcion http://localhost:8501/.

![alt-text](https://github.com/gabrielcoralc/Proyeccionconsumo_stl/blob/main/gifs/fullpage.gif)

Una vez iniciado solo es necesario seguir las instrucciones de cada seccion, hay links para descargar los archivos de excel de ejemplo que el programa puede recibir.

Cada seccion realiza un proceso diferente pero casi todas dependen de que se carge el primer archivo el cual se supone son los valores reales que se van a comparar con la prediccion para asi obtener las anomalias en ese rango.

Se tienen en total 4 secciones:

- Training models and forecasting (Massive): Realiza el proceso de entrenamiento de modelos para prediccion, de todos los codigos y series de tiempo que se relacione en los archivos cargados.
- Forecasting (Massive) for models already trained : Esta seccion es para que realice predicciones de modelos ya entrenados, dando la posibilidad de modificar los parametros de numero de pasos a futuro a predecir, y cantidad de desviaciones estadar para el analisis de anomalias.
- Plot training, Forecasting and actual data: Se presenta el resultado grafico de las series de tiempo utilizadas para entrenamiento, serie de tiempo de la prediccion y anomalias detectadas.
- Calculate trends for all Frts: este es un proceso independiente de lo modelos de prediccion, solo calcula las tendencias de consumo.

De igual manera todas las secciones al terminar su proceso dan un link donde se puede descargar la informacion que dio como resultado el proceso.


### Resultado grafico prediccion

Finalmente la parte que visualmente la mas llamativa de la aplicacion y que permite mucha interaccion con el usuario.

![alt-text](https://github.com/gabrielcoralc/Proyeccionconsumo_stl/blob/main/gifs/prediction_plot.gif)

## Conclusion

Finalmente dejo el comentario de que el codigo tiene bastante espacio para ser mejorado y ser mucho mas eficiente, pero para un trabajo basico realiza lo que se pretenden con buenos resultados.
