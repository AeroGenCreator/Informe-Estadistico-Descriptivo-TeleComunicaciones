# Informe-Estadistico-Descriptivo-Y-Modelos-De-Clasificacion-Megaline

📊 Análisis de Datos y Modelo de Clasificación de Clientes de Telecomunicaciones (Megaline)

Este proyecto se divide en dos fases principales: Análisis Estadístico de Datos de las tarifas prepago Surf y Ultimate de Megaline, y el desarrollo de un Modelo de Clasificación para recomendar un nuevo plan a los clientes.

## 1. Análisis Estadístico de Datos

El objetivo fue determinar qué tarifa (Surf o Ultimate) generaba más ingresos promedio para la compañía.
🛠️ Preparación y Preprocesamiento de Datos

    Librerías Clave: Se utilizaron principalmente Pandas para la manipulación y NumPy para operaciones numéricas.

    Alineación de Datos: Se combinaron las cinco tablas de datos (users, calls, messages, internet, y plans) para obtener un conjunto de datos unificado.

    Cálculo de Consumo: Se calcularon los totales mensuales de llamadas (minutos), mensajes (SMS) y uso de datos (GB) por cada usuario. Se aplicaron las reglas de redondeo de Megaline (segundos a minutos por llamada individual; MB a GB para el total mensual).

    Cálculo de Ingresos: Se determinaron los ingresos mensuales por usuario, restando el límite del paquete de los totales de consumo y aplicando las tarifas por exceso, sumando finalmente la cuota mensual.

🔎 Análisis del Comportamiento del Cliente

![Image Alt](https://github.com/AeroGenCreator/Informe-Estadistico-Descriptivo-Y-Modelos-De-Clasificacion-Megaline/blob/main/1.png)
![Image Alt](https://github.com/AeroGenCreator/Informe-Estadistico-Descriptivo-Y-Modelos-De-Clasificacion-Megaline/blob/main/2.png)
![Image Alt](https://github.com/AeroGenCreator/Informe-Estadistico-Descriptivo-Y-Modelos-De-Clasificacion-Megaline/blob/main/3.png)


Se examinaron las métricas de consumo (minutos, SMS, GB) para cada tarifa:

    Se calcularon la media, la varianza y la desviación estándar para describir la dispersión del consumo.

    Se generaron histogramas para visualizar las distribuciones del consumo, mostrando que los usuarios de Surf tienden a acercarse más a sus límites de paquete.

🧪 Prueba de Hipótesis Estadística

Se utilizó una prueba t de dos muestras independientes (de scipy.stats) para probar dos hipótesis clave, asumiendo un umbral de significancia (α) de 0.05:

    Hipótesis 1 (Ingresos):

        H0​: El ingreso promedio de los usuarios de las tarifas Ultimate y Surf NO difiere.

        Ha​: El ingreso promedio de los usuarios de las tarifas Ultimate y Surf DIFIERE.

    Hipótesis 2 (Región):

        H0​: El ingreso promedio de los usuarios de la región NY-NJ NO difiere del de otras regiones.

        Ha​: El ingreso promedio de los usuarios de la región NY-NJ DIFIERE del de otras regiones.

Conclusión del Análisis: Los resultados estadísticos (encontrados en los archivos Jupyter) permitieron determinar si la diferencia en los ingresos promedio es estadísticamente significativa, informando al departamento comercial sobre la tarifa más rentable.

## 2. Modelado de Clasificación de Planes

![Image Alt](https://github.com/AeroGenCreator/Informe-Estadistico-Descriptivo-Y-Modelos-De-Clasificacion-Megaline/blob/main/models.png)

El objetivo fue crear un modelo que, basándose en el comportamiento de los usuarios, pudiera predecir y recomendar uno de los nuevos planes de Megaline (Smart o Ultra), alcanzando una exactitud (accuracy) superior a 0.75.
⚙️ Pipeline de Modelado

    Librerías Clave: Scikit-learn (sklearn) fue la base para todos los modelos y métricas.

    Segmentación de Datos: El dataset procesado se segmentó en conjuntos de entrenamiento (60%), validación (20%) y prueba (20%) utilizando train_test_split.

    Características: Las features incluyeron calls, minutes, messages, y mb_used. La target fue is_ultra (1 para Ultra, 0 para Smart).

🤖 Modelos Evaluados

Se investigó el rendimiento de varios modelos de clasificación de sklearn, ajustando hiperparámetros para maximizar la exactitud:
Modelo	Clase de Scikit-learn	Hiperparámetros Clave
Árbol de Decisión	DecisionTreeClassifier	max_depth
Bosque Aleatorio	RandomForestClassifier	n_estimators, max_depth
Regresión Logística	LogisticRegression	solver, random_state
