import numpy as np
from sympy import fwht, ifwht
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def hada_gen(n):
    m = np.int(np.log2(n))
    H = 1;
    for i in range(m):
        col1 = np.vstack((H,H))
        col2 = np.vstack((H,-H))
        H = np.hstack((col1,col2))

    return H/np.sqrt(n)

def Sort_Vecs_mat(M):
    n,m = M.shape
    vars = np.sqrt(np.diag(1./np.linalg.norm(M,axis=0)))
    M_sorted = np.copy(M[0,:])
    M_sorted = np.expand_dims(M_sorted,axis=0)
    remain_ind = [i for i in range(n)]
    remain_ind = remain_ind[1:]
    Rearrange = [0]
    Permute = np.zeros((n,n))
    Permute[0,0] = 1
    for i in range(n-1):
        temp = np.expand_dims(M_sorted[-1,:],axis=0)
        dist = M[remain_ind,:]-np.repeat(temp,len(remain_ind),axis=0)
        dist_normal = np.mean(np.power(dist@vars,2),axis=1)
        idxmin = np.argmin(dist_normal)
        Rearrange.append(remain_ind[idxmin])
        M_sorted = np.vstack((M_sorted, M[remain_ind[idxmin], :]))
        Permute[i+1, remain_ind[idxmin]] = 1
        remain_ind.pop(idxmin)
    return Permute, M_sorted

def Sort_Vecs_Vectors(M):
    n, m = M.shape
    Indsort = np.zeros_like(M)
    M_sorted = np.copy(M)
    for i in range(m):
        temp = M[:, i]
        M_sorted[:, i] = np.sort(temp)
        Indsort[:, i] = np.argsort(temp)
    return Indsort, M_sorted

def Hadamard_trans(M):
    n,m = M.shape
    realn = n
    power2 = np.ceil(np.log2(n))
    hadDim = np.power(2,power2)
    res = np.power(2,power2)-n
    M_transformed = np.copy(M)
    M_transformed = np.vstack((M_transformed,np.repeat(np.expand_dims(M_transformed[-1,:],axis=0),res,axis = 0)))
    H = hada_gen(hadDim)
    M_transformed = H@M_transformed
    return M_transformed, np.int(hadDim), realn

def Hadamard_trans_vecs(M):
    n,m = M.shape
    realn = n
    power2 = np.ceil(np.log2(n))
    hadDim = np.power(2,power2)
    res = np.power(2,power2)-n
    M_transformed = np.copy(M)
    M_transformed = np.vstack((M_transformed,np.repeat(np.expand_dims(M_transformed[-1,:],axis=0),res,axis = 0)))
    H = hada_gen(hadDim)
    M_transformed = H@M_transformed
    return M_transformed, np.int(hadDim), realn

def Compress(M_transformed, level, whole=False):
    if whole == False:
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

# def Compress_vec(M_transformed, level, whole=False):
#     if whole == False:
#         energy_average = np.mean(np.power(M_transformed,2),axis=1)
#         Energy_normal = energy_average/np.sum(energy_average)
#         Energy_Sorted = np.flip(np.sort(Energy_normal))
#         Energy_Sorted_idx = np.flip(np.argsort(Energy_normal))
#         cumlative = np.cumsum(Energy_Sorted)
#         idx_choose = np.argmax(cumlative > level)
#         Chosen_Idx = Energy_Sorted_idx[:idx_choose+1]
#         M_transformed_compressed = M_transformed[Chosen_Idx,:]
#         return Chosen_Idx,M_transformed_compressed
#     else:
#         print('fffff')
#         return [i for i in range(M_transformed.shape[0])],M_transformed

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
# def Hadamard_invtrans(M_transformed):
#     return M