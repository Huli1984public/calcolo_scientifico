import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import random


class VectorAngles:

    def __init__(self, matrix):
            self.angolo = None
            self.tan = None
            self.cos = None
            self.matrix = matrix

    def calcola_retta_due_punti(self):
        '''(x-x1)/(x2-x1)=(y-y1)/(y2-y1)

        A(5, 3)
        A'(7, 6) -> 3x-15 = 2y-6 -> y = 3/2x - 9/2 -> CA = 3/2

        B(4, 1)
        B'(8, 2) -> x-4 = 4y-4 -> y = 1/4x -> CA = 1/4'''
        pass

    def calcola_angolo_due_rette(self):
        '''tan(a) = |m1-m2|/|1+(m1*m2)|
        (a) angolo = arctan(tan(a))

        |3/2-1/4|/|1+3/8| -> (5/4)/(11/8)
        a = circa 42 gradi
        '''
        pass

    def calcola_angolo_vettori(self):
        pass


if __name__ == "__main__":
    '''prima_retta = np.array([[2, 3]])
    seconda_retta = np.array([[4, 1]])
    matrice = np.concatenate((prima_retta, seconda_retta), axis=0)'''
    n = random.randint(0, 50)
    matrice = np.random.rand(n, n)
    # print(f"matrice:\n{matrice}")
    print(matrice.shape)
    print("______\n")
    cos_list = []
    for i in range(len(matrice.T)):
        if i + 1 < len(matrice.T):
            print(i)
            prod_vec = np.dot(matrice.T[i,:], matrice.T[i+1,:])
            norm1 = np.linalg.norm(matrice.T[i,:])
            norm2 = np.linalg.norm(matrice.T[i+1,:])
            cos_vec = prod_vec/(norm1*norm2)
            cos_list.append(cos_vec)
            angle = np.degrees(np.arccos(cos_vec))
            # print(f"colonna 1: {matrice.T[i,:]}\ncolonna 2: {matrice.T[i+1,:]}")

    cos_array = np.array(cos_list)
    print(f"cos array: {cos_array}\nlen list: {len(cos_list)}")

    fig = plt.figure(1)
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(cos_array[:], cos_array[:], cos_array[:])
    plt.show()


