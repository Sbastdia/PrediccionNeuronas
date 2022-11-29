#-----------------------------------------------------------------------------------------
# @Autor: Aurélien Vannieuwenhuyze
# @Empresa: Junior Makers Place
# @Libro:
# @Capítulo: 10 - La predicción con neuronas
#
# Módulos necesarios:
#   NUMPY 1.16.3
#   MATPLOTLIB : 3.0.3
#   TENSORFLOW : 1.13.1
#
# Para instalar un módulo:
#   Haga clic en el menú File > Settings > Project:nombre_del_proyecto > Project interpreter > botón +
#   Introduzca el nombre del módulo en la zona de búsqueda situada en la parte superior izquierda
#   Elegir la versión en la parte inferior derecha
#   Haga clic en el botón install situado en la parte inferior izquierda
#-----------------------------------------------------------------------------------------


from numpy import exp, array, random
import matplotlib.pyplot as plt




class Perceptron:


    #-------------------------------------
    #    OBSERVACIONES Y PREDICCIONES
    #-------------------------------------

    def __init__(self):

        self.observaciones_entradas = array([
                                    [1, 0],
                                    [1, 1],
                                    [0, 1],
                                    [0, 0]
                                    ])


        self.predicciones = array([[0],[1], [0],[0]])

    #--------------------------------------
    #      PARAMETRIZACIÓN DEL PERCEPTRÓN
    #--------------------------------------

    def generacionPesos(self):

        #Generación de los pesos en el intervalo [-1;1]
        random.seed(1)
        self.limiteMin = -1
        self.limiteMax = 1

        self.w11 = (self.limiteMax-self.limiteMin) * random.random() + self.limiteMin
        self.w21 = (self.limiteMax-self.limiteMin) * random.random() + self.limiteMin
        self.w31 = (self.limiteMax-self.limiteMin) * random.random() + self.limiteMin

    def sesgo(self):
        #El sesgo
        self.sesgo = 1
        self.wb = 0

    def almacenamiento(self):
        #Almacenamiento de los pesos iniciales, solo para visualización al final del aprendizaje
        self.peso = [self.w11,self.w21,self.w31,self.wb]

    def tasaAprendizaje(self):
        #Tasa de aprendizaje
        self.txAprendizaje = 0.1

    def cantidadEpocas(self):
        #Cantidad de épocas
        self.epochs = 10000


    #--------------------------------------
    #       FUNCIONES ÚTILES
    #--------------------------------------

    @staticmethod
    def suma_ponderada(X1,W11,X2,W21,B,WB):
        return (B*WB+( X1*W11 + X2*W21))

    @staticmethod
    def funcion_activacion_sigmoide(valor_suma_ponderada):
        return (1 / (1 + exp(-valor_suma_ponderada)))

    @staticmethod
    def funcion_activacion_relu(valor_suma_ponderada):
        return (max(0,valor_suma_ponderada))

    @staticmethod
    def error_lineal(valor_esperado, valor_predicho):
        return (valor_esperado-valor_predicho)

    @staticmethod
    def calculo_gradiente(valor_entrada,prediccion,error):
        return (-1 * error * prediccion * (1-prediccion) * valor_entrada)

    @staticmethod
    def calculo_valor_ajuste(valor_gradiente, tasa_aprendizaje):
        return (valor_gradiente*tasa_aprendizaje)

    @staticmethod
    def calculo_nuevo_peso (valor_peso, valor_ajuste):
        return (valor_peso - valor_ajuste)

    @staticmethod
    def calculo_MSE(predicciones_realizadas, predicciones_esperadas):
        i=0;
        suma=0;
        for prediccion in predicciones_esperadas:
            diferencia = predicciones_esperadas[i] - predicciones_realizadas[i]
            cuadradoDiferencia = diferencia * diferencia
            suma = suma + cuadradoDiferencia
        media_cuadratica = 1 / (len(predicciones_esperadas)) * suma
        return media_cuadratica

    #--------------------------------------
    #       GRÁFICA
    #--------------------------------------

    def grafica2(self):
        self.Grafica_MSE=[]


    #--------------------------------------
    #    APRENDIZAJE
    #--------------------------------------

    def aprendizaje2(self):

        for epoch in range(0,self.epochs):
            print("EPOCH ("+str(epoch)+"/"+str(self.epochs)+")")
            self.predicciones_realizadas_durante_epoch = [];
            self.predicciones_esperadas = [];
            self.numObservacion = 0
            for observacion in self.observaciones_entradas:

                #Carga de la capa de entrada
                self.x1 = observacion[0];
                self.x2 = observacion[1];

                #Valor de predicción esperado
                self.valor_esperado = self.predicciones[self.numObservacion][0]

                #Etapa 1: Cálculo de la suma ponderada
                self.valor_suma_ponderada = Perceptron.suma_ponderada(self.x1,self.w11,self.x2,self.w21,self.sesgo,self.wb)


                #Etapa 2: Aplicación de la función de activación
                self.valor_predicho = Perceptron.funcion_activacion_sigmoide(self.valor_suma_ponderada)


                #Etapa 3: Cálculo del error
                self.valor_error = Perceptron.error_lineal(self.valor_esperado,self.valor_predicho)


                #Actualización del peso 1
                #Cálculo ddel gradiente del valor de ajuste y del peso nuevo
                self.gradiente_W11 = Perceptron.calculo_gradiente(self.x1,self.valor_predicho,self.valor_error)
                self.valor_ajuste_W11 =Perceptron.calculo_valor_ajuste(self.gradiente_W11,self.txAprendizaje)
                self.w11 = Perceptron.calculo_nuevo_peso(self.w11,self.valor_ajuste_W11)

                # Actualización del peso 2
                self.gradiente_W21 = Perceptron.calculo_gradiente(self.x2, self.valor_predicho, self.valor_error)
                self.valor_ajuste_W21 = Perceptron.calculo_valor_ajuste(self.gradiente_W21, self.txAprendizaje)
                self.w21 = Perceptron.calculo_nuevo_peso(self.w21, self.valor_ajuste_W21)


                # Actualización del peso del sesgo
                self.gradiente_Wb = Perceptron.calculo_gradiente(self.sesgo, self.valor_predicho, self.valor_error)
                self.valor_ajuste_Wb = Perceptron.calculo_valor_ajuste(self.gradiente_Wb, self.txAprendizaje)
                self.wb = Perceptron.calculo_nuevo_peso(self.wb, self.valor_ajuste_Wb)

                print("     EPOCH (" + str(epoch) + "/" + str(self.epochs) + ") -  Observación: " + str(self.numObservacion+1) + "/" + str(len(self.observaciones_entradas)))

                #Almacenamiento de la predicción realizada:
                self.predicciones_realizadas_durante_epoch.append(self.valor_predicho)
                self.predicciones_esperadas.append(self.predicciones[self.numObservacion][0])

                #Paso a la observación siguiente
                self.numObservacion = self.numObservacion+1

            self.MSE = Perceptron.calculo_MSE(self.predicciones_realizadas_durante_epoch, self.predicciones)
            self.Grafica_MSE.append(self.MSE[0])
            print("MSE: "+str(self.MSE))


    def mostrar(self):
        plt.plot(self.Grafica_MSE)
        plt.ylabel('MSE')
        plt.show()


        print()
        print()
        print ("¡Aprendizaje terminado!")
        print ("Pesos iniciales: " )
        print ("W11 = "+str(self.peso[0]))
        print ("W21 = "+str(self.peso[1]))
        print ("Wb = "+str(self.peso[3]))

        print ("Pesos finales: " )
        print ("W11 = "+str(self.w11))
        print ("W21 = "+str(self.w21))
        print ("Wb = "+str(self.wb))

        print()
        print("--------------------------")
        print ("PREDICCIÓN ")
        print("--------------------------")
        self.x1 = 1
        self.x2 = 1

    def etapa1(self):
        #Etapa 1: Cálculo de la suma ponderada
        self.valor_suma_ponderada = Perceptron.suma_ponderada(self.x1,self.w11,self.x2,self.w21,self.sesgo,self.wb)

    def etapa2(self):
        #Etapa 2: Aplicación de la función de activación
        self.valor_predicho = Perceptron.funcion_activacion_sigmoide(self.valor_suma_ponderada)

        print("Predicción del [" + str(self.x1) + "," + str(self.x2)  + "]")
        print("Predicción = " + str(self.valor_predicho))

    def inicio(self):
        self.generacionPesos()
        self.sesgo()
        self.almacenamiento()
        self.tasaAprendizaje()
        self.cantidadEpocas()

    def aprendizaje(self):
        self.grafica2()
        self.aprendizaje2()
        self.mostrar()
        self.etapa1()
        self.etapa2()

    @staticmethod
    def ejecutar():
        Percept=Perceptron()
        Percept.inicio()
        Percept.aprendizaje()

    


if __name__=="__main__":
    Perceptron.ejecutar()


