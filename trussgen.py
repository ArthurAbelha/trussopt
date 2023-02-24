import numpy as np
import numpy.linalg as npla
import random
import matplotlib.pyplot as plt
import os
import time


def delbarra(barras, conec):
    aux = 1
    for z in barras:
        conec = np.delete(conec, z - aux, axis=0)
        aux += 1


conec = []
coord = []
a = 1  # 0 vizinhança e 1 todos com todos
c = 0  # c 0 para treliça quadrada, 1 para triangular
b = 1  # A base da unidade
h = 1  # A altura da unidade
bb = 8  # Base da treliça
hh = 8  # Altura da treliça
timee = time.time()
if a == 0:
    if c == 0:
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
    conec = sorted(conec)
elif a == 1:
    if c == 0:
        nb = int(bb//b) + 1  # Número de nós na base
        nh = int(hh//h) + 1
        # Gerando um array que contem as coordenadas X/Y, percorrendo cada linha
        for j in range(nh):
            for i in range(nb):
                coord.append([i*b, j*h])
        for i in range(nb * nh):  # Todos os pontos
            for j in range(i + 1, nb * nh):  # Todos os pontos depois de i (e.g. i = 5, j = 6, 7, 8...
                if conec.count([i + 1, j + 1]) == 0 and conec.count([j + 1, i + 1]) == 0:
                    conec.append([i + 1, j + 1])
        for k in conec:
            if k[0] > k[1]:
                conec[conec.index(k)] = [k[1], k[0]]
        for i in conec:
            for j in conec:
                if i[0] == j[0] or i[1] == j[1]:
                    if np.cross(np.subtract(coord[i[0]-1], coord[i[1]-1]), np.subtract(coord[j[0]-1], coord[j[1]-1])) \
                            == 0:
                        if npla.norm([coord[i[0]-1], coord[i[1]-1]]) != npla.norm([coord[j[0]-1], coord[j[1]-1]]):
                            print("\n\n main: ", i, j)
                            if npla.norm(np.subtract(coord[i[0] - 1], coord[i[1] - 1])) > \
                                    npla.norm(np.subtract(coord[j[0] - 1], coord[j[1] - 1])):
                                if i in conec:
                                    print("i is bigger than j: norm i = ", npla.norm([coord[i[0] - 1], coord[i[1] - 1]])
                                          , "norm j = ", npla.norm([coord[j[0] - 1], coord[j[1] - 1]]))
                                    conec.pop(conec.index(i))
                            elif npla.norm(np.subtract(coord[i[0] - 1], coord[i[1] - 1])) < npla.norm(np.subtract(coord[j[0] - 1], coord[j[1] - 1])):
                                if j in conec:
                                    print("j is bigger than i.")
                                    conec.pop(conec.index(j))
    elif c == 1:
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
        for i in range(len(coord)):
            for j in range(i+1, len(coord)):
                conec.append([i+1, j+1])
    conec = sorted(conec)
    print("Time:", time.time()-timee)
    print("Number of links: ", np.array(conec).shape[0])
if os.path.exists("\ex\gerado") == 0:
    os.makedirs("\ex\gerado")
np.savetxt("\ex\gerado\coord.txt", coord)
np.savetxt("\ex\gerado\conec.txt", conec, fmt='%i')
print(np.size(conec))
