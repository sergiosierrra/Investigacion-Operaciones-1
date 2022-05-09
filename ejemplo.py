#Ejemplo de uso de la libreria de SHAP
#Sergio Camilo Sierra Pinilla 
#Codigo:20201020072

# Se importan las librerias que se van a usar
import pandas as pd
import shap
import sklearn

# conjunto de datos de precios de las viviendas de Boston
X,y = shap.datasets.boston()
X100 = shap.utils.sample(X, 100) # 100 instancias para usar como distribución de fondo

# Modelo de la regresion linear simple
model = sklearn.linear_model.LinearRegression()
model.fit(X, y)

# Se muestran los coeficientes de las caracteristicas de las viviendas
print("Model coefficients:\n")
for i in range(X.shape[1]):
    print(X.columns[i], "=", model.coef_[i].round(4))

# Muestra la grafica de la regresion lineal
shap.plots.partial_dependence(
    "RM", model.predict, X100, ice=False,
    model_expected_value=True, feature_expected_value=True
)

# Se calcula los valores SHAP para el modelo lineal
explainer = shap.Explainer(model.predict, X100)
shap_values = explainer(X)

# Se realiza una grafica de dependencia parcial estándar
sample_ind = 18
shap.partial_dependence_plot(
    "RM", model.predict, X100, model_expected_value=True,
    feature_expected_value=True, ice=False,
    shap_values=shap_values[sample_ind:sample_ind+1,:]
)

shap.plots.scatter(shap_values[:,"RM"])

# Se muestra un diagrama de cascada el cualpasamos de shap_values.base_values ​​a model.predict(X)[sample_ind], es decir se pasa a un modelo predictivo
shap.plots.waterfall(shap_values[sample_ind], max_display=14)