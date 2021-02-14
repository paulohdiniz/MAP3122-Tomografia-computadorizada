import math
import numpy as np

#Solução LU

def Uxy(U, y):
    x = np.zeros_like(y)
    for i in range(len(x), 0, -1):
      x[i-1] = (y[i-1] - np.dot(U[i-1, i:], x[i:])) / U[i-1, i-1]
    return x
def Lyb(L, b):
    y = []
    for i in range(len(b)):
        y.append(b[i])
        for j in range(i):
            y[i]=y[i]-(L[i, j]*y[j])
        y[i] = y[i]/L[i, i]
    return y
def LuDecomposition(A):
    L = np.zeros_like(A,dtype=float)
    U = np.zeros_like(A,dtype=float)
    N = np.size(A,0)
    for k in range(N):
        L[k, k] = 1
        U[k, k] = (A[k, k] - np.dot(L[k, :k], U[:k, k])) / L[k, k]
        for j in range(k+1, N):
            U[k, j] = (A[k, j] - np.dot(L[k, :k], U[:k, j])) / L[k, k]
        for i in range(k+1, N):
            L[i, k] = (A[i, k] - np.dot(L[i, :k], U[:k, k])) / U[k, k]
    return L, U
def LUSolution(L, U, b):
    y = Lyb(L,b)
    x = Uxy(U,y)
    return x
def SolutionLinear(A, b):
    L, U = LuDecomposition(A)
    x = LUSolution(L,U,b)
    return x


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

