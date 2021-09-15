# Proyeccionconsumo_stl
Proyecto para realizar predicciones de consumos a series de tiempo con informacion diara.

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

![alt-text](link)
