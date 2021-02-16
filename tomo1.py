import math, matplotlib.pyplot as plt, numpy as np, matplotlib.image as mpimg, pprint, sys, os
from numpy import load

# funções básicas/auxiliares

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

#Devolve matriz mxn com zeros
def MatrizZero (m, n):
    A = np.zeros((m,n))
    return A

#Devolve matriz mxn com zeros e linha x com UNS
def MatrizNUns (m, n, x):
    B = MatrizZero(m,n)
    for i in range(m):
        for j in range(n):
            if i == x-1:
                B[i][j] = 1
            else:
                B[i][j] = 0
    return B

#plota o resultado das imagens do programa
def plotaImages(image, sol1, sol2, sol3):
    plt.figure(figsize=(10,6))
    plt.subplot(2,2,1)
    plt.axis('off')
    plt.title("Imagem Original")
    img = mpimg.imread(image)
    imgplot = plt.imshow(img)

    plt.subplot(2,2,2)
    plt.axis('off')
    plt.title("\u03B4: 0.001")
    plt.imshow(sol1)

    plt.subplot(2,2,3)
    plt.axis('off')
    plt.title("\u03B4: 0.01")
    plt.imshow(sol2)

    plt.subplot(2,2,4)
    plt.axis('off')
    plt.title("\u03B4: 0.1")
    plt.imshow(sol3)


    plt.show()

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

#Pega a imagem original e transforma em um vetor
def getImageVector(pathIMG):
    vectorimagens = plt.imread(pathIMG)
    vectorok = vectorimagens.flatten('F')
    return vectorok

#Obtem a % do erro
def getERRO(vectorimg, vector2):
    i = 0
    soma1 = 0
    soma2 = 0
    while i < len(vectorimg):
        dif2 = (vectorimg[i] - vector2[i])**2
        soma1 = soma1 + dif2
        denominador = vectorimg[i]*vectorimg[i]
        soma2 = soma2 + denominador
        i = i + 1
    sqrtsoma1 = math.sqrt(soma1)
    sqrtsoma2 = math.sqrt(soma2)
    return 100*(sqrtsoma1/sqrtsoma2)

#Funções para fatoração e solução por LU
def lenMatrix(M):
	N = len(M[:, 0]) 
	M = len(M[0, :])
	return N, M

def escalonamento(M):
	vector = M.copy()
	cross = MatrizZero(len(vector[0,:]), len(vector[:,0]))
	for i in range(len(vector[:, 0])-1):
		for k in range(i+1, len(vector[:,0])):
			cross[k, i]  = vector[k, i]/vector[i,i]
			temp = vector[i,] * cross[k, i]
			fim = vector[k,] - temp
			vector[k] = fim
	for i in range(len(cross)):
		cross[i, i] = 1
	return vector, cross

def MatL(L, vector):
	linhas,colunas = lenMatrix(L)
	R = vector
	M = L
	X  = MatrizZero(linhas, 1)
	for i in range(linhas):
		X[i] = R[i]
		for j in range(i):
			X[i] -= M[i][j]*X[j]
		X[i] /= M[i][i]
	return X

def MatU(L, vector):
	linhas,colunas = lenMatrix(L)
	R = vector
	M = L
	X  = MatrizZero(linhas, 1)
	for i in range(linhas-1,-1,-1):
		X[i, 0] = R[i, 0]
		for k in range(i+1, linhas):
			X[i, 0] -= M[i][k]*X[k][0]
		X[i, 0] /= M[i][i]
	return X

def SolutionWithLU(M, R):
	resU, resL  = escalonamento(M)
	Y = MatL(resL, R)
	X = MatU(resU, Y)
	return X
#Fim da Fatoração e solução por LU

#INICIO DO PROGRAMA

#Coletando Entradas
path = sys.argv[1]
if ("im1" == path):
    pathimagem = os.path.join(path, "im1.png")
    pathnpy = os.path.join(path, "p1.npy")
elif ("im2" == path):
    pathimagem = os.path.join(path, "im2.png")
    pathnpy = os.path.join(path, "p1.npy")
elif ("im3" == path):
    pathimagem = os.path.join(path, "im3.png")
    pathnpy = os.path.join(path, "p1.npy")
else:
    print ("Siga o modelo do PDF: python tomo(x).py im(x)")

#Coletando vetor p
p = load(pathnpy)
n = int(len(p)/2) 

#Calculando A e sua transposta 
A = MatrizA(n)
At = Transposta(A)

#Req = At*A + delta*I (para deltas respectivos)
Req0 = Requerida(n, 0)
Req1 = Requerida(n, 0.001)
Req2 = Requerida(n, 0.01)
Req3 = Requerida(n, 0.1)

#Calculando os Determinantes requeridos no Ex1
det0 = Determinante(Req0)
det1 = Determinante(Req1)
det2 = Determinante(Req2)
det3 = Determinante(Req3)

#Escreve esses determinantes num txt de saida
file = open("determinantesEx1.txt", "a+")
str_dictionary0 = repr(det0)
str_dictionary1 = repr(det1)
str_dictionary2 = repr(det2)
str_dictionary3 = repr(det3)
file.write("Imagem: " + path + "\n" + "delta: 0 = " + str_dictionary0 + "\n" + "delta: 0.001 = " + str_dictionary1 + "\n" + "delta: 0.01 = " + str_dictionary2 + "\n" + "delta: 0.1 = " + str_dictionary3 + "\n")
file.close()

#Solução do sist linear
newp = np.dot(At, p)
f1 = SolutionWithLU(Req1, newp)
f2 = SolutionWithLU(Req2, newp) 
f3 = SolutionWithLU(Req3, newp)

#calculando erro
vectorimg = getImageVector(pathimagem)
errof1 = getERRO(vectorimg, f1)
errof2 = getERRO(vectorimg, f2)
errof3 = getERRO(vectorimg, f3)

#Escreve esses erros num txt de saida
file = open("errosEx1.txt", "a+")
erro1 = repr(errof1)
erro2 = repr(errof2)
erro3 = repr(errof3)
file.write("Imagem: " + path + "\n" + "delta 0.001, " + "erro: " + erro1 + "\n" + "delta 0.01, " + "erro: " + erro2 + "\n" + "delta 0.1, " + "erro: " + erro3 + "\n" + "\n")
file.close()

#ajeitando a matriz pra o formato ideal
m1 = Transposta(f1.reshape(int(math.sqrt(len(f1))),int(math.sqrt(len(f1)))))
m2 = Transposta(f2.reshape(int(math.sqrt(len(f2))),int(math.sqrt(len(f2)))))
m3 = Transposta(f3.reshape(int(math.sqrt(len(f3))),int(math.sqrt(len(f3)))))

#plotando imagens
plotaImages(pathimagem, m1, m2, m3)

#Fim do programa