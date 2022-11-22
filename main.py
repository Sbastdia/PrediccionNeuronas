from codigo.Programacion_del_perceptron import Perceptron
from codigo.Perceptron_con_Tensorflow import Tensorflow

if __name__=='__main__':

    print('¿Qué archivo quiere ejecutar?')
    eleccion=input('Perceptron sin Tensorflow [1] ó Perceptron con Tensorflow [2]: ')

    if eleccion=='1':
        Perceptron.ejecutar()

    else:
        Tensorflow.ejecutar()