#-----------------------------------------------------------------------------------------
# @Autor: Aurélien Vannieuwenhuyze
# @Empresa: Junior Makers Place
# @Libro:
# @Capítulo: 10 - La predicción con neuronas
#
# Módulos necesarios:
#   NUMPY 1.16.3
#   MATPLOTLIB: 3.0.3
#   TENSORFLOW: 1.13.1
#
# Para instalar un módulo:
#   Haga clic en el menú File > Settings > Project:nombre_del_proyecto > Project interpreter > botón +
#   Introduzca el nombre del módulo en la zona de búsqueda situada en la parte superior izquierda
#   Elegir la versión en la parte inferior derecha
#   Haga clic en el botón install situado en la parte inferior izquierda
#-----------------------------------------------------------------------------------------




#-------------------------------------
#    PARAMETROS GENERALES
#-------------------------------------

valores_entradas_X = [[1., 0.], [1., 1.], [0., 1.], [0., 0.]]
valores_a_predecir_Y = [[0.], [1.], [0.], [0.]]



#-------------------------------------
#    PARÁMETROS DE LA RED
#-------------------------------------
import tensorflow as tf


#Variable TensorFLow correspondiente a los valores de neuronas de entrada
tf_neuronas_entradas_X = tf.placeholder(tf.float32, [None, 2])

#Variable TensorFlow correspondiente a la neurona de salida (predicción real)
tf_valores_reales_Y = tf.placeholder(tf.float32, [None, 1])

#-- Peso --
#Creación de una variable TensorFlow de tipo tabla
#que contiene 2 entradas y cada una tiene un peso [2,1]
#Estos valores se inicializan al azar
peso = tf.Variable(tf.random_normal([2, 1]), tf.float32)

#-- Sesgo inicializado a 0 --
sesgo = tf.Variable(tf.zeros([1, 1]), tf.float32)

#La suma ponderada es en la práctica una multiplicación de matrices
#entre los valores en la entrada X y los distintos pesos
#la función matmul se encarga de hacer esta multiplicación
sumaponderada = tf.matmul(tf_neuronas_entradas_X,peso)

#Adición del sesgo a la suma ponderada
sumaponderada = tf.add(sumaponderada,sesgo)

#Función de activación de tipo sigmoide que permite calcular la predicción
prediccion = tf.sigmoid(sumaponderada)

#Función de error de media cuadrática MSE
funcion_error = tf.reduce_sum(tf.pow(tf_valores_reales_Y-prediccion,2))

#Descenso de gradiente con una tasa de aprendizaje fijada a 0,1
optimizador = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(funcion_error)



#-------------------------------------
#    APRENDIZAJE
#-------------------------------------

#Cantidad de epochs
epochs = 10000

#Inicialización de la variable
init = tf.global_variables_initializer()

#Inicio de una sesión de aprendizaje
sesion = tf.Session()
sesion.run(init)

#Para la realización de la gráfica para la MSE
Grafica_MSE=[]


#Para cada epoch
for i in range(epochs):

   #Realización del aprendizaje con actualzación de los pesos
   sesion.run(optimizador, feed_dict = {tf_neuronas_entradas_X: valores_entradas_X, tf_valores_reales_Y:valores_a_predecir_Y})

   #Calcular el error
   MSE = sesion.run(funcion_error, feed_dict = {tf_neuronas_entradas_X: valores_entradas_X, tf_valores_reales_Y:valores_a_predecir_Y})

   #Visualización de la información
   Grafica_MSE.append(MSE)
   print("EPOCH (" + str(i) + "/" + str(epochs) + ") -  MSE: "+ str(MSE))


#Visualización gráfica
import matplotlib.pyplot as plt
plt.plot(Grafica_MSE)
plt.ylabel('MSE')
plt.show()

print("--- VERIFICACIONES ----")

for i in range(0,4):
    print("Observación:"+str(valores_entradas_X[i])+ " - Esperado: "+str(valores_a_predecir_Y[i])+" - Predicción: "+str(sesion.run(prediccion, feed_dict={tf_neuronas_entradas_X: [valores_entradas_X[i]]})))



sesion.close()
