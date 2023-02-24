import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import copy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
# 3555.0602015998707 J: Nó 27 com as coordenadas x e y alteradas para 4.185000000000005 e 1.2499999999999991

def force(xa, xb, ua, ub, e, a):
   l0 = np.linalg.norm(xb-xa)
   xaf = xa + ua
   xbf = xb + ub
   lf = np.linalg.norm(xbf-xaf)
   eab = (xbf-xaf)/lf
   return np.linalg.norm(eab*(e*a*(lf-l0)/l0))


def maxforce(conec, coord, desloc, e, a, nb):
   forces = np.zeros(nb)
   ua = np.zeros((2, 1))
   ub = ua
   for k in range(nb):
       ia = int(conec[k][0])-1
       ib = int(conec[k][1])-1
       xa = np.vstack(coord[ia])
       xb = np.vstack(coord[ib])
       ua = np.vstack(np.array([desloc[2*ia-2], desloc[2*ia-1]]))
       ub = np.vstack(np.array([desloc[2*ib-2], desloc[2*ib-1]]))
       forces[k] = force(xa, xb, ua, ub, e, a)
   return forces


def rigidez2(nb, nv, conec, coord, mod):
   kg = np.zeros((2*nv, 2*nv))
   for k in range(nb):
       ia = int(conec[k][0])-1
       ib = int(conec[k][1])-1
       xa = coord[ia]
       xb = coord[ib]
       d = np.vstack(xb-xa)
       l0 = np.linalg.norm(d)
       aux = mod[k]/(l0**3)
       mat = aux*(d*(xb-xa))
       kb = np.array(np.bmat([[mat, -mat], [-mat, mat]]))
       loc = [2*ia, 2*ia+1, 2*ib, 2*ib+1]
       for i in range(4):
           for j in range(4):
               ig = loc[i]
               jg = loc[j]
               kg[ig][jg] = kg[ig][jg] + kb[i][j]
   return kg


# def rigidez3(nb, nv, conec, coord, mod):


def restringe(conecaux, coordaux, a, b):
   if a == 0:  # Restringe colunas
       if b == 0:  # Restringe apenas a coluna da esquerda
           nr = [1]
           for i in conecaux[:, 0]:
               i = int(i)
               if coordaux[i, 0] == 0:
                   i += 1
                   nr.append(i)
       elif b == 1:  # Restringe apenas a coluna da direita
           nr = []
           for i in conecaux[:, 0]:
               i = int(i)
               if coordaux[i, 0] == max(coordaux[:, 0]):
                   i += 1
                   nr.append(i)
       elif b == 2:  # Restringe ambas extremidades horizontais
           nr = [1]
           for i in conecaux[:, 0]:
               i = int(i)
               if coordaux[i, 0] == 0:
                   i += 1
                   nr.append(i)
               elif coordaux[i, 0] == max(coordaux[:, 0]):
                   i += 1
                   nr.append(i)
       elif b == 3:  # Restringe uma coluna específica com base na coordenada x da mesma
           nr = []
           for i in conecaux[:, 0]:
               i = int(i)
               if coordaux[i, 0] == float(input()):
                   i += 1
                   nr.append(i)
   elif a == 1:
       if b == 0:  # Restringe a linha inferior
           nr = [1]
           for i in conecaux[:, 0]:
               i = int(i)
               if coordaux[i, 1] == 0:
                   i += 1
                   nr.append(i)
       elif b == 1:  # Restringe a linha superior
           nr = []
           for i in conecaux[:, 0]:
               i = int(i)
               if coordaux[i, 1] == max(coordaux[:, 1]):
                   i += 1
                   nr.append(i)
       elif b == 2:  # Restringe ambas extremidades verticais
           nr = [1]
           for i in conecaux[:, 0]:
               i = int(i)
               if coordaux[i, 1] == 0:
                   i += 1
                   nr.append(i)
               elif coordaux[i, 1] == max(coordaux[:, 1]):
                   i += 1
                   nr.append(i)
       elif b == 3:  # Restringe uma linha específica com base na coordenada y
           nr = []
           for i in conecaux[:, 0]:
               i = int(i)
               if coordaux[i, 1] == float(input()):
                   i += 1
                   nr.append(i)
   return nr


def plotb(conec, p, m, c):
   nb = len(conec)
   pp = np.zeros((2, 2))
   for i in range(nb):
       pp[0, 0] = p[int(conec[i][0])-1][0]
       pp[0, 1] = p[int(conec[i][0])-1][1]
       pp[1, 0] = p[int(conec[i][1])-1][0]
       pp[1, 1] = p[int(conec[i][1])-1][1]
       plt.plot(pp[:, 0], pp[:, 1], marker=m, color=c)


conec = []
coord = []
# c 0 para treliça quadrada, 1 para triangular
c = 1
if c == 0:
   b = 1  # A base da unidade
   h = 1  # A altura da unidade
   bb = 15  # A base da treliça
   hh = 3  # A altura da treliça
   nb = int(bb//b) + 1  # Número de nós na base
   nh = int(hh//h) + 1
   # Gerando um array que contem as coordenadas X/Y, percorrendo cada linha
   for j in range(nh):
       for i in range(nb):
           coord.append([i*b, j*h])
   for i in range(nb):
       for j in range(nh):
           if i == 0:
               if j == 0:
                   conec.extend([[1, 2], [1, nb+1], [1, nb+2]])
               elif j == nh-1:
                   conec.extend([[nb*j+1, nb*(j-1)+2], [nb*j+1, nb*j+2]])
               else:
                   conec.extend([[nb*j+1, nb*(j-1)+2], [nb*j+1, nb*j+2], [nb*j+1, nb*(j+1)+1], [nb*j+1, nb*(j+1)+2]])
           elif i != nb-1:
               if j == 0:
                   conec.extend([[i+1, i+2], [i+1, i+1+nb], [i+1, i+2+nb]])
               elif j == nh-1:
                   conec.extend([[i+1+j*nb, i+2+(j-1)*nb], [i+1+j*nb, i+2+j*nb]])
               else:
                   conec.extend([[i+1+j*nb, i+2+(j-1)*nb], [i+1+j*nb, i+2+j*nb], [i+1+j*nb, i+1+(j+1)*nb], [i+1+j*nb, i+2+(j+1)*nb]])
           else:
               if j != nh-1:
                   conec.append([i+1+j*nb, i+1+(j+1)*nb])
   # Ajeitando o array de conectividade por ordem crescente de nós "de referência".
elif c == 1:
   b = 1  # Base da unidade-triângulo
   bb = 4  # Base da treliça
   hh = 4  # Altura da treliça
   nb = int(bb//b) + 1  # Número de nós na base
   h = hh/(nb-1)  # Altura da unidade-triângulo
   # Gerando um array que contem as coordenadas X/Y, percorrendo cada linha
   aux0 = 0
   for j in range(nb):
       for i in range(nb-aux0):
           if j % 2 == 0:
               coord.append([i*b + (j/2)*b, j*h])
           else:
               coord.append([i*b + j*(b/2), j*h])
       aux0 += 1
   aux0 = 0
   aux1 = 1
   for j in range(nb-1):
       v = 1
       for i in range(nb-aux0):
           if i == 0:
               conec.extend([[aux1, aux1+1], [aux1, aux1+nb-aux0]])
           elif i == nb-aux0-1:
               conec.append([aux1+nb-aux0-1, aux1+2*nb-2*aux0-2])
           else:
               conec.extend([[aux1+v, aux1+1+v], [aux1+v, aux1+nb-aux0+v-1], [aux1+v, aux1+nb-aux0+v]])
               v += 1
       aux0 += 1
       aux1 = aux1 + 1 + nb - aux0

conec = np.loadtxt("C:\\Users\Arthur\PycharmProjects\pythonProject\ex\gerado\conec.txt")
nb = conec.shape[0]
conecaux = copy.deepcopy(conec)
coord = np.loadtxt("C:\\Users\Arthur\PycharmProjects\pythonProject\ex\gerado\coord.txt")
[nv, syscoord] = coord.shape
coordaux = copy.deepcopy(coord)
a = 1.e-5  # Área transversal
e = 1.e11  # Módulo de Young


if syscoord == 2:
   barrasdel = [1, 2, 3, 0]
   aux1 = 1
   passo = 10
   conecauxlist = []
   desloclist = []
   for z in barrasdel:
       fig = plt.figure()
       fig.add_subplot().set_aspect(1)
       conecauxlist.append(conecaux)
       nb = conecaux.shape[0]
       mod = e*a*np.ones((nb, 1))
       kg = rigidez2(nb, nv, conecaux, coord, mod)
       kr = np.copy(kg)
       iden = np.identity(2*nv)
       nr = restringe(conecaux, coordaux, 0, 0)
       for i in nr:
           kr[2*i-2, :] = iden[2*i-2, :]
           kr[2*i-1, :] = iden[2*i-1, :]
       f = np.zeros((2*nv, 1))
       f[30] = -50000  # Uma força horizontal em 15 (índice 14), por exemplo, deve usar de f[28], e a vertical, f[29]
       f[31] = -50000
       b = np.zeros((2*nv, 1))
       krs = sc.sparse.csr_matrix(kr)
       for k in range(passo):
           b = b + (f/passo)
           desloc = sc.sparse.linalg.spsolve(krs, b)
           forces = maxforce(conecaux, coord, desloc, e, a, nb)
           for index, forca in enumerate(forces):
               if forca == 0:
                   print(index)
           im = np.argmax(np.abs(forces))
           maximum = forces[im]
       deslocaux = np.zeros((nv, 2))
       for i in range(nv):
           deslocaux[i][0] = desloc[2*i]
           deslocaux[i][1] = desloc[2*i+1]
       n = coord + 4*deslocaux
       desloclist.append(n)
       print(desloc.T@krs@desloc)
       # plotb(conecaux, coord, "o", "r")
       # plotb(conecaux, n, "o", "b")
       # for i in range(nv):
       #     plt.text(coord[i, 0]+0.05, coord[i, 1]+0.01, i+1)
       # plt.show()
       # conecaux = np.delete(conecaux, z - aux1, axis=0)
       aux1 += 1