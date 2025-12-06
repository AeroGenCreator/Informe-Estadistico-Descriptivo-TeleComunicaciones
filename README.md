# Informe-Estadistico-Descriptivo-Y-Modelos-De-Clasificacion-Megaline
Ejercio de practica, pruebas estadisticas y graficos.

📊 Análisis de Ingresos de Planes de Telecomunicaciones (Megaline)

Introducción y Objetivo del Proyecto

Este proyecto se centra en un análisis de datos para Megaline, un operador de telecomunicaciones que ofrece dos planes de prepago: Surf y Ultimate.

El objetivo principal es realizar un análisis exploratorio y estadístico sobre una muestra de 500 clientes para determinar cuál de los dos planes genera mayores ingresos promedio para la empresa. La información obtenida será crucial para el departamento comercial al momento de ajustar el presupuesto de publicidad y optimizar la estrategia de marketing.

🔑 Análisis Desarrollado y Metodología

El proyecto se desarrolla a través de un Jupyter Notebook siguiendo una metodología robusta de análisis de datos:

1. Preprocesamiento y Preparación de Datos

    Inspección y Limpieza: Se realizó una revisión detallada de cinco datasets (users, calls, messages, internet, plans) para identificar y corregir anomalías, valores ausentes o errores.

    Conversión de Tipos: Se ajustaron los tipos de datos (e.g., fechas a formato datetime) según fue necesario.

    Cálculo de Consumo Mensual: Se agregaron los datos para calcular el consumo mensual total de cada usuario en tres métricas clave:

        Número de llamadas y minutos utilizados.

        Cantidad de mensajes de texto (SMS) enviados.

        Volumen de datos (MB) utilizados.

2. Cálculo de Ingresos y Fusión de Datos

    Determinación de Ingresos: Se implementó una función para calcular el ingreso mensual total por cada usuario. Este cálculo incluye la cuota mensual fija y suma los cargos adicionales por el consumo que exceda los límites de los paquetes (Surf o Ultimate), respetando la política de redondeo de Megaline (segundos a minutos; MB total a GB).

    Integración de Datos: Los datos de consumo e ingresos se fusionaron con la información de los usuarios y planes para crear un dataset único listo para el análisis.

3. Análisis Exploratorio de Datos (EDA)
    ![image alt](https://github.com/AeroGenCreator/Informe-Estadistico-Descriptivo-TeleComunicaciones/blob/main/1.png)
    ![image alt](https://github.com/AeroGenCreator/Informe-Estadistico-Descriptivo-TeleComunicaciones/blob/main/2.png)
    ![image alt](https://github.com/AeroGenCreator/Informe-Estadistico-Descriptivo-TeleComunicaciones/blob/main/3.png)
    Se describió el comportamiento de los clientes para cada plan, calculando la media, varianza y desviación estándar del consumo mensual (minutos, SMS, datos) para los usuarios de las tarifas Surf y Ultimate.

    Se generaron histogramas para visualizar las distribuciones del consumo en ambas tarifas, permitiendo una comprensión clara de cómo se utilizan los recursos del plan.

5. Pruebas de Hipótesis Estadísticas

Se emplearon pruebas t de dos muestras (Two-sample t-tests) para validar las siguientes hipótesis estadísticas, utilizando un valor α predefinido (especificado en el notebook):

    Hipótesis 1: Se prueba si el ingreso promedio de los usuarios de la tarifa Ultimate difiere del ingreso promedio de los usuarios de la tarifa Surf.

    Hipótesis 2: Se prueba si el ingreso promedio de los usuarios en el área de Nueva York-Nueva Jersey es diferente al ingreso promedio de los usuarios de otras regiones.

🚀 Conclusión y Resultados

El proyecto culmina con una conclusión general que resume los hallazgos del análisis exploratorio y los resultados de las pruebas de hipótesis. El principal entregable es la recomendación fundamentada sobre cuál de los planes (Surf o Ultimate) genera, en promedio, más ingresos para la compañía Megaline, sirviendo de base para futuras decisiones empresariales.
