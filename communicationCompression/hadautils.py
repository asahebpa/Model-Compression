import numpy as np
from sympy import fwht, ifwht
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def hada_gen(n):
    m = np.int(np.log2(n))
    H = 1;
    for i in range(m):
        col1 = np.vstack((H,H))
        col2 = np.vstack((H,-H))
        H = np.hstack((col1,col2))

    return H/np.sqrt(n)

def Sort_Vecs_Vectors(M):
    n, m = M.shape
    Indsort = np.zeros_like(M)
    M_sorted = np.copy(M)
    print('dd',M.shape)
    kmeans = KMeans(n_clusters=10, random_state=0).fit(M_sorted.T)
    print(np.where(kmeans.labels_==1)[0].shape)
    for i in range(m):
        temp = M[:, i]
        M_sorted[:, i] = np.sort(temp)
        Indsort[:, i] = np.argsort(temp)
    return Indsort, M_sorted

def Sort_Vecs_Vectors2(M,NC):
    n, m = M.shape
    M_sorted2 = np.copy(M)
    num_clusters = NC
    Indsort = np.zeros_like(M)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(M.T)
    for i in range(num_clusters):
        Mtemp = np.copy(M_sorted2[:,np.where(kmeans.labels_==i)[0]])
        n2, m2 = Mtemp.shape
        vars = np.linalg.norm(Mtemp,1, axis=1)
        idxmax = np.argmax(vars)
        M_sorted = np.copy(Mtemp[idxmax, :])
        M_sorted = np.expand_dims(M_sorted, axis=0)
        remain_ind = [i for i in range(n2)]
        remain_ind = remain_ind[1:]
        Rearrange = [idxmax]
        for i2 in range(n2 - 1):
            temp = np.expand_dims(M_sorted[-1, :], axis=0)
            dist = Mtemp[remain_ind, :] - np.repeat(temp, len(remain_ind), axis=0)
            dist_normal = np.mean(np.power(dist,2), axis=1)
            idxmin = np.argmin(dist_normal)
            Rearrange.append(remain_ind[idxmin])
            M_sorted = np.vstack((M_sorted, Mtemp[remain_ind[idxmin], :]))
            remain_ind.pop(idxmin)
        Indsort[:,np.where(kmeans.labels_==i)[0]] = np.repeat(np.expand_dims(np.asarray(Rearrange).T,axis=1),np.where(kmeans.labels_==i)[0].shape[0],axis=1)
        M_sorted2[:,np.where(kmeans.labels_==i)[0]] = M_sorted

        # if M_sorted.shape[1]==2:
        #     print('ggg')
        #     print(M_sorted)
        #     input('f')
    return Indsort, M_sorted2

def Hadamard_trans(M):
    n,m = M.shape
    realn = n
    power2 = np.ceil(np.log2(n))
    hadDim = np.power(2,power2)
    res = np.power(2,power2)-n
    M_transformed = np.copy(M)
    M_transformed = np.vstack((M_transformed,np.repeat(np.expand_dims(M_transformed[-1,:],axis=0),res,axis = 0)))
    offset = np.min(np.min(M_transformed))
    M_transformed += offset
    H = hada_gen(hadDim)
    M_transformed = H@M_transformed
    return M_transformed, np.int(hadDim), realn, offset


def Compress(M_transformed, level, whole=False):
    if whole == False:
        n, m = M_transformed.shape
        energy_average = np.mean(np.power(M_transformed,2),axis=1)
        Energy_normal = energy_average/np.sum(energy_average)
        Energy_Sorted = np.flip(np.sort(Energy_normal))
        Energy_Sorted_idx = np.flip(np.argsort(Energy_normal))
        cumlative = np.cumsum(Energy_Sorted)
        idx_choose = np.argmax(cumlative > level)
        Chosen_Idx = Energy_Sorted_idx[:idx_choose+1]
        M_transformed_compressed = M_transformed[Chosen_Idx,:]
        return Chosen_Idx,M_transformed_compressed
    else:
        print('fffff')
        return [i for i in range(M_transformed.shape[0])],M_transformed


def Compress2(M_transformed, level, whole=False):
    if whole == False:
        n, m = M_transformed.shape
        Chosen_Idx = -1*np.ones_like(M_transformed)
        M_transformed_compressed = np.zeros_like(M_transformed)
        for i in range(m):
            energy_average = np.power(M_transformed[:,i],2)
            Energy_normal = energy_average/np.sum(energy_average)
            Energy_Sorted = np.flip(np.sort(Energy_normal))
            Energy_Sorted_idx = np.flip(np.argsort(Energy_normal))
            cumlative = np.cumsum(Energy_Sorted)
            idx_choose = np.argmax(cumlative > level)
            Chosen_Idx[:idx_choose+1,i] = Energy_Sorted_idx[:idx_choose+1]
            M_transformed_compressed[(Chosen_Idx[:idx_choose+1,i].astype(int)),i] = M_transformed[(Chosen_Idx[:idx_choose+1,i].astype(int)),i]
        return Chosen_Idx,M_transformed_compressed
    else:
        print('fffff')
        return [i for i in range(M_transformed.shape[0])],M_transformed

def Hadamard_invtrans2(n,Chosen_Idx,M_transformed_compressed,realn):
    # M_invtransformed = np.zeros((n,M_transformed_compressed.shape[1]))
    M_invtransformed = np.copy(M_transformed_compressed)
    H = hada_gen(n)
    M_invtransformed = (H.T)@M_invtransformed
    return M_invtransformed[:realn,:]
def Hadamard_invtrans(n,Chosen_Idx,M_transformed_compressed,realn):
    M_invtransformed = np.zeros((n,M_transformed_compressed.shape[1]))
    M_invtransformed[Chosen_Idx,:] = np.copy(M_transformed_compressed)
    H = hada_gen(n)
    M_invtransformed = (H.T)@M_invtransformed
    return M_invtransformed[:realn,:]
def deperm_vec(PermuteInd, M_invTransed):
    n, m = M_invTransed.shape
    M_depermed = np.copy(M_invTransed)
    for i in range(m):
        M_depermed[PermuteInd[:,i],i] = M_invTransed[:,i]
    return M_depermed