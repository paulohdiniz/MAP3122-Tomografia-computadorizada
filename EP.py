import math, matplotlib.pyplot as plt, numpy as np, pandas as pd, seaborn as sns, matplotlib.image as mpimg, pprint

#Fatoração e solução por LU

def backward(U, y):
    x = np.zeros_like(y)
    for i in range(len(x), 0, -1):
      x[i-1] = (y[i-1] - np.dot(U[i-1, i:], x[i:])) / U[i-1, i-1]
    return x
def forward(L, b):
    y = []
    for i in range(len(b)):
        y.append(b[i])
        for j in range(i):
            y[i]=y[i]-(L[i, j]*y[j])
        y[i] = y[i]/L[i, i]
    return y
def FatLU(A):
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
def SolucaoLU(L, U, b):
    y = forward(L,b)
    x = backward(U,y)
    return x
def SolucaoSistema(A, b):
    L, U = FatLU(A)
    x = SolucaoLU(L,U,b)
    return x

# funções básicas

# Identidade ordem n
def Identidade(n):
    I = np.identity(n)
    return I

# Transposta
def Transposta (A):
    B = A.transpose()
    return B

# Determinante
def Determinante(A):
    B = np.linalg.det(A)
    return B

# Inversa
def Inversa (A):
    B = np.linalg.inv(A)
    return B

#Devolve matriz mxn com zeros
def MatrizZero (m, n):
    A = np.zeros((m,n))
    return A

#Diagonal Secundaria de UNS
def DiagSec (ordem):
    A = np.fliplr(np.diag(np.ones(ordem)))
    return A

#funcoes particulares

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

def deslocaMatriz(A):
    m = A.shape[0]
    n = A.shape[1]
    B = MatrizZero(m,n)
    for i in range(m):
        for j in range(n):
            if i == 0 or j ==0:
                B[i][j] = 0
            else:
                B[i][j] = A[i-1][j-1]
    return B

def deslocaMatriz2(A, dimensional):
    m = A.shape[0]
    n = A.shape[1]
    B = MatrizZero(m,n)
    for i in range(m):
        for j in range(n):
            if i < 1 or j < dimensional:
                B[i][j] = 0
            else:
                B[i][j] = A[i-1][j-dimensional]
    return B

#primeiro elemento[0][0] igual a 1 e os demais zeros
def first(dimensao):
    A = np.zeros((1, dimensao))
    A[0][0]=1
    return A

def MatrizBelow(dimensao):
    A = np.kron(np.identity(dimensao), first(dimensao))
    B = np.zeros((dimensao -1,dimensao*dimensao)) 
    C = np.vstack([A, B]) #acrescenta B linhas à matriz A
    R = np.zeros((2*dimensao -1, dimensao*dimensao))
    i = 1
    D = C
    while i < dimensao:
        D = deslocaMatriz(D)
        R = R + D
        i = i + 1
    return R +C

def MatrizUp(dimensao):
    A = DiagSec(dimensao)
    B = np.zeros((dimensao -1,dimensao)) 
    C = np.vstack([A, B]) #acrescenta B linhas à matriz A
    D = np.zeros((2*dimensao - 1, dimensao*(dimensao -1)))
    E = np.hstack([C, D])
    R = np.zeros((2*dimensao -1, dimensao*dimensao))
    i = 1
    F = E
    while i < dimensao:
        F = deslocaMatriz2(F,dimensao)
        R = R + F
        i = i + 1
    return R + E

#Matriz A
def MatrizA (entrada):
    ParteSuperior = np.kron(MatrizNUns(1, entrada, 1), Identidade(entrada))
    ParteInferior = np.kron(Identidade(entrada), MatrizNUns(1, entrada, 1))
    Soma = np.kron(MatrizNUns(2,1,1), ParteSuperior) + np.kron(MatrizNUns(2,1,2), ParteInferior)
    return Soma

def MatrizA2(dimensao):
    A = MatrizA(dimensao)
    B = MatrizUp(dimensao)
    C = MatrizBelow(dimensao)
    D = np.vstack([A, B])
    E = np.vstack([D, C])
    return E



#Matriz Requerida At*A + delta*I
def Requerida (dimensao, delta):
    Req = np.dot(Transposta(MatrizA(dimensao)),MatrizA(dimensao)) + delta*Identidade(dimensao*dimensao)
    return Req


def SolucaoEx1 (dimensao, delta, p):
    Sol = np.linalg.multi_dot([Inversa(Requerida(dimensao, delta)), MatrizA(dimensao), p])
    return Sol



# #plotando gráfico
# plt.imshow(MatrizA2(10), interpolation='none')
# plt.show()

#Carregando as medições correspondentes de p2
# from numpy import load
# p1 = load("p2.npy")
# print(p2)


#Imagem X = NxN

#Vetor f = 2N x 1

# Matriz A = 2N x N²

#Programa mesmo.

sigma = np.array({0, 0.001, 0.01, 0.1})


#Carregando as medições correspondentes de p1
from numpy import load
p1 = load("p1.npy")
n1 = int(len(p1)/2)

plt.imshow(SolucaoEx1(n1, 0, p1), interpolation='none')
plt.show()

#Carrega a imagem original
img = mpimg.imread('im1.png')
imgplot = plt.imshow(img)
plt.show()