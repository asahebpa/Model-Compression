from utils import *
from hadautils import *

def Compression_procedure(ptemp,compressdim,Level,whole_flag,NC):
    if compressdim == 1:
        ptemp = ptemp.T
    paramorg = [ptemp.shape[0] * ptemp.shape[1]]
    Perm_mat, Newp = Sort_Vecs_Vectors2(ptemp,NC)
    Newp_transformed, hadDim, n_real, offset = Hadamard_trans(Newp)
    Chosen_row, Newp_compressed = Compress2(Newp_transformed, Level, whole_flag)
    paramcompress = [np.where(Chosen_row>-1)[0].shape[0]]
    # paramcompress = [Newp_compressed.shape[0]*Newp_compressed.shape[1]]
    Newp_hat = Hadamard_invtrans2(hadDim, Chosen_row, Newp_compressed, n_real)
    Newp_hat -= offset
    Decompressedp = deperm_vec(Perm_mat.astype(np.int32), Newp_hat)
    if compressdim == 1:
        return Decompressedp.T, paramorg, paramcompress
    else:
        return Decompressedp, paramorg, paramcompress

