import math
import numpy as np

sigma = np.array({0, 0.001, 0.01, 0.1})

#Carregando as medições correspondentes de p1
from numpy import load
p1 = load("p1.npy")
n = int(len(p1)/2)
print(n)


#FUNCOES

#Matriz n-ésima linha de UNS
def Xesima (x, lin, col): 
    XesimaM = [[0 for x in range(lin)] for y in range(col)] 
    for i in range(lin):
        for j in range(col):
            if i == x-1:
                XesimaM[i][j] = 1
            else:
                XesimaM[i][j] = 0
    return XesimaM
#Devolve a matriz identidade
def Identidade(n):
    I = np.identity(n)
    return I

#Devolve matriz mxn com zeros
def MatrizZero (m, n):
    A = np.zeros((m,n))
    return A

#Devolve matriz mxn com zeros e linha x com UNS

def MatrizNUns (m, n, x):
    B = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            if i == x-1:
                B[i][j] = 1
            else:
                B[i][j] = 0
    return B

def MatrizA (entrada):
    ParteSuperior = np.kron(MatrizNUns(1, entrada, 1), Identidade(entrada))
    ParteInferior = np.kron(Identidade(entrada), MatrizNUns(1, entrada, 1))
    Soma = np.kron(MatrizNUns(2,1,1), ParteSuperior) + np.kron(MatrizNUns(2,1,2), ParteInferior)
    return Soma

def Transposta (A):
    B = A.transpose()
    return B

print (Transposta(MatrizA(4)))
#Carregando as medições correspondentes de p2
# from numpy import load
# p1 = load("p2.npy")
# print(p2)


#Imagem X = NxN

#Vetor f = 2N x 1

# Matriz A = 2N x N²



