import math, matplotlib.pyplot as plt, numpy as np, pandas as pd, seaborn as sns, matplotlib.image as mpimg, pprint, sys, os
from numpy import load

#Fatoração e solução por LU
def getTamanhoDeMatriz(matriz):
	numero_linhas = len(matriz[:, 0]) 
	numero_colunas = len(matriz[0, :])
	return numero_linhas, numero_colunas

def gaussEscalona(matriz):
	array = matriz.copy()
	multiplicadores = np.zeros([len(array[0,:]), len(array[:,0])])
	for i in range(len(array[:, 0])-1):
		for k in range(i+1, len(array[:,0])):
			multiplicadores[k, i]  = array[k, i]/array[i,i]
			linhaTemp = array[i,] * multiplicadores[k, i]
			linhaFinal = array[k,] - linhaTemp
			array[k] = linhaFinal
	for i in range(len(multiplicadores)):
		multiplicadores[i, i] = 1
	return array, multiplicadores

def resolveTriangularInf(matrizTriangular, respostas):
	linhas,colunas = getTamanhoDeMatriz(matrizTriangular)
	R = respostas
	M = matrizTriangular
	X  = np.zeros([linhas, 1])
	for i in range(linhas):
		X[i] = R[i]
		for j in range(i):
			X[i] -= M[i][j]*X[j]
		X[i] /= M[i][i]
	return X

def resolveTriangularSup(matrizTriangular, respostas):
	linhas,colunas = getTamanhoDeMatriz(matrizTriangular)
	R = respostas
	M = matrizTriangular
	X  = np.zeros([linhas, 1])
	for i in range(linhas-1,-1,-1):
		X[i, 0] = R[i, 0]
		for k in range(i+1, linhas):
			X[i, 0] -= M[i][k]*X[k][0]
		X[i, 0] /= M[i][i]
	return X

def achaSolucao(M, R):
	upper, lower  = gaussEscalona(M)
	Y = resolveTriangularInf(lower, R)
	X = resolveTriangularSup(upper, Y)
	return X

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

#Matriz A
def MatrizA (entrada):
    ParteSuperior = np.kron(MatrizNUns(1, entrada, 1), Identidade(entrada))
    ParteInferior = np.kron(Identidade(entrada), MatrizNUns(1, entrada, 1))
    Soma = np.kron(MatrizNUns(2,1,1), ParteSuperior) + np.kron(MatrizNUns(2,1,2), ParteInferior)
    return Soma

#Matriz Requerida At*A + delta*I
def Requerida (dimensao, delta):
    Req = np.dot(Transposta(MatrizA(dimensao)),MatrizA(dimensao)) + delta*Identidade(dimensao*dimensao)
    return Req

#INICIO DO PROGRAMA

#Coletando Entradas
path = sys.argv[1]
if ("im1" == path):
    pathimagem = os.path.join(path, "im1.png")
    pathnpy = os.path.join(path, "p1.npy")
elif ("im2" == path):
    pathimagem = os.path.join(path, "im2.png")
    pathnpy = os.path.join(path, "p2.npy")
elif ("im3" == path):
    pathimagem = os.path.join(path, "im3.png")
    pathnpy = os.path.join(path, "p3.npy")
else:
    print ("Siga o modelo do PDF: python tomo(x).py im(x)")

#Coletando vetor p
p = load(pathnpy)
n = int(len(p)/2) 

#Calculando A e sua transposta 
A = MatrizA(n)
At = Transposta(A)

#Req = At*A + delta*I (para deltas respectivos)
Req1 = Requerida(n, 0.001)
Req2 = Requerida(n, 0.01)
Req3 = Requerida(n, 0.1)

#Solução do sist linear
newp = np.dot(At, p)


f1 = achaSolucao(Req2, newp)
f2 = achaSolucao(Req3, newp)
f3 = achaSolucao(Req4, newp)


m1 = Transposta(f2.reshape(int(math.sqrt(len(f2))),int(math.sqrt(len(f2)))))
m2 = f3.reshape(int(math.sqrt(len(f3))),int(math.sqrt(len(f3))))
m3 = f4.reshape(int(math.sqrt(len(f4))),int(math.sqrt(len(f4))))
