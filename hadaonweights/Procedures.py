from utils import *
from hadautils import *

def MatriceformCompress(ptemp,compressdim,Level,whole_flag):
    if compressdim == 1:
        ptemp = ptemp.T
    paramorg = [ptemp.shape[0] * ptemp.shape[1]];
    Perm_mat, Newp = Sort_Vecs_mat(ptemp)
    Newp_transformed, hadDim, n_real = Hadamard_trans(Newp)
    Chosen_row, Newp_compressed = Compress(Newp_transformed, Level, whole_flag)
    paramcompress = [Newp_compressed.shape[0] * Newp_compressed.shape[1]]
    Newp_hat = Hadamard_invtrans(hadDim, Chosen_row, Newp_compressed, n_real)
    Decompressedp = (Perm_mat.T) @ Newp_hat
    if compressdim == 1:
        return Decompressedp.T,paramorg,paramcompress
    else:
        return Decompressedp, paramorg, paramcompress

def VectorizedCompress(ptemp,compressdim,Level,whole_flag):
    if compressdim == 1:
        ptemp = ptemp.T
    paramorg = [ptemp.shape[0] * ptemp.shape[1]];
    Perm_mat, Newp = Sort_Vecs_Vectors(ptemp)
    Newp_transformed, hadDim, n_real = Hadamard_trans(Newp)
    Chosen_row, Newp_compressed = Compress(Newp_transformed, Level, whole_flag)
    paramcompress = [Newp_compressed.shape[0] * Newp_compressed.shape[1]]
    Newp_hat = Hadamard_invtrans(hadDim, Chosen_row, Newp_compressed, n_real)
    print(Perm_mat)
    print(Perm_mat.shape)
    Decompressedp = deperm_vec(Perm_mat.astype(np.int32), Newp_hat)
    if compressdim == 1:
        return Decompressedp.T, paramorg, paramcompress
    else:
        return Decompressedp, paramorg, paramcompress