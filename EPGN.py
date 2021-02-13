import math
import numpy as np

#MATRIZ A 
def MA(n):
    Resultado = np.kron(np.matrix([[1],[0]]), np.kron(np.ones(n), np.identity(n))) + np.kron(np.matrix([[0],[1]]), np.kron(np.identity(n), np.ones(n)))
    return Resultado

#At*A + delta*I
def DeteterminantePedido(delta, n):
    determinante = np.linalg.det(np.dot(MA(n).transpose(), MA(n)) + delta*np.identity(n*n))
    return determinante

#def solucao(delta, n):
def first(dimensao):
    A = np.zeros((1, dimensao))
    A[0][0]=1
    return A

B = np.zeros((2,9)) 
A = np.kron(np.identity(4), first(4))

B = np.fliplr(np.diag(np.ones(3)))
print(B)

print(A)

