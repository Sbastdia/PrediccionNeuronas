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

import tensorflow as tf
import matplotlib.pyplot as plt
tf=tf.compat.v1
tf.disable_eager_execution()


class Tensorflow:

    #-------------------------------------
    #    PARAMETROS GENERALES
    #-------------------------------------
    def __init__(self):

        self.valores_entradas_X = [[1., 0.], [1., 1.], [0., 1.], [0., 0.]]
        self.valores_a_predecir_Y = [[0.], [1.], [0.], [0.]]



    #-------------------------------------
    #    PARÁMETROS DE LA RED
    #-------------------------------------


    def entrada(self):
        #Variable TensorFLow correspondiente a los valores de neuronas de entrada
        self.tf_neuronas_entradas_X = tf.placeholder(tf.float32, [None, 2])

    def salida(self):
        #Variable TensorFlow correspondiente a la neurona de salida (predicción real)
        self.tf_valores_reales_Y = tf.placeholder(tf.float32, [None, 1])

    def peso(self):
        #-- Peso --
        #Creación de una variable TensorFlow de tipo tabla
        #que contiene 2 entradas y cada una tiene un peso [2,1]
        #Estos valores se inicializan al azar
        self.peso = tf.Variable(tf.random_normal([2, 1]), tf.float32)

    def sesgo(self):
        #-- Sesgo inicializado a 0 --
        self.sesgo = tf.Variable(tf.zeros([1, 1]), tf.float32)

    def sumaPonderada(self):
        #La suma ponderada es en la práctica una multiplicación de matrices
        #entre los valores en la entrada X y los distintos pesos
        #la función matmul se encarga de hacer esta multiplicación
        self.sumaponderada = tf.matmul(self.tf_neuronas_entradas_X,self.peso)

    def adicion(self):
        #Adición del sesgo a la suma ponderada
        self.sumaponderada = tf.add(self.sumaponderada,self.sesgo)

    def fActivacion(self):
        #Función de activación de tipo sigmoide que permite calcular la predicción
        self.prediccion = tf.sigmoid(self.sumaponderada)

    def fError(self):
        #Función de error de media cuadrática MSE
        self.funcion_error = tf.reduce_sum(tf.pow(self.tf_valores_reales_Y-self.prediccion,2))

    def descensoGradiente(self):
        #Descenso de gradiente con una tasa de aprendizaje fijada a 0,1
        self.optimizador = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(self.funcion_error)



    #-------------------------------------
    #    APRENDIZAJE
    #-------------------------------------

    def inicializacion(self):
        #Cantidad de epochs
        self.epochs = 10000

        #Inicialización de la variable
        self.init = tf.global_variables_initializer()

        #Inicio de una sesión de aprendizaje
        self.sesion = tf.Session()
        self.sesion.run(self.init)

    def grafica(self):
        #Para la realización de la gráfica para la MSE
        self.Grafica_MSE=[]

    def aprendizaje1(self):
        #Para cada epoch
        for i in range(self.epochs):

            #Realización del aprendizaje con actualzación de los pesos
            self.sesion.run(self.optimizador, feed_dict = {self.tf_neuronas_entradas_X: self.valores_entradas_X, self.tf_valores_reales_Y:self.valores_a_predecir_Y})

            #Calcular el error
            MSE = self.sesion.run(self.funcion_error, feed_dict = {self.tf_neuronas_entradas_X: self.valores_entradas_X, self.tf_valores_reales_Y:self.valores_a_predecir_Y})

            #Visualización de la información
            self.Grafica_MSE.append(MSE)
            print("EPOCH (" + str(i) + "/" + str(self.epochs) + ") -  MSE: "+ str(MSE))

    def visualizacion(self):
        #Visualización gráfica

        plt.plot(self.Grafica_MSE)
        plt.ylabel('MSE')
        plt.show()

        print("--- VERIFICACIONES ----")

        for i in range(0,4):
            print("Observación:"+str(self.valores_entradas_X[i])+ " - Esperado: "+str(self.valores_a_predecir_Y[i])+" - Predicción: "+str(self.sesion.run(self.prediccion, feed_dict={self.tf_neuronas_entradas_X: [self.valores_entradas_X[i]]})))



        self.sesion.close()

    def inicio(self):
        self.entrada()
        self.salida()
        self.peso()
        self.sesgo()
        self.sumaPonderada()
        self.adicion()
        self.fActivacion()
        self.fError()
        self.descensoGradiente()

    def aprendizaje(self):
        self.inicializacion()
        self.grafica()
        self.aprendizaje1()
        self.visualizacion()

    @staticmethod
    def ejecutar():
        Tensor=Tensorflow()
        Tensor.inicio()
        Tensor.aprendizaje()

if __name__=="__main__":
    Tensorflow.ejecutar()
