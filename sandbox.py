
import numpy as np

import matplotlib.pyplot as plt

# a = np.load("data/data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48_2_ID2C00_streams.combined_streams_scaled.generated.npy")
# a = np.load("data/data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48_1_ID1E66900_streams.combined_streams.npy")
# a = np.load("data/dekaworld_alex_guille_neosdata2/data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48_2_ID2C00_streams.combined_streams.npy")
# a = np.load("data/kulzaworld_guille_neosdata_npy/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.proc_feats.npy")
# a = np.load("data/kulzaworld_guille_neosdata_npy_axis_angle/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.proc_feats.npy")
# a = np.load("data/kulzaworld_guille_neosdata_npy_axis_angle/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.rest_feats.npy")
# a = np.load("data/kulzaworld_guille_neosdata_npy_axis_angle/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.dat.person1.npy")
a = np.load("data/kulzaworld_guille_neosdata_npy_relative/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.dat.person1.rel.npy")
# a = np.load("data/kulzaworld_guille_neosdata_npy_axis_angle/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_1_ID2C00_streams.dat.person1.npy")
# a = np.load("data/transflower_expmap_cr4_bs5_og2_futureN_gauss5_single2_kulzaworld_neosraw/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.proc_feats.generated.npy")
# a = np.load("data/moglow_expmap1_tf3_single_kulzaworld_neosraw/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.proc_feats.generated.npy")
# b = np.load("data\example_numpy_frames.npy")
# b = np.load("data\example_numpy_frames.npy")

a.shape
a[0:2,0]
plt.matshow(a[:100,0,:])
# plt.matshow(a[200:300])
a[110:120,7]
a[110:120,6:9]
prev_rot=a[303,12:15]
rot=a[304,12:15]
norms = np.linalg.norm(rot,axis=-1, keepdims=True)
rot2 = -(2*np.pi-norms)*rot/norms
rot2
prev_rot
np.all(np.abs(prev_rot-rot2)<=1e-1)
np.any(np.abs(rot-prev_rot) >= 1e-1) and np.abs(np.linalg.norm(rot)-np.linalg.norm(prev_rot)) <= 1e-1
np.all(np.abs(prev_rot+rot)<=1e-1)
norm = np.linalg.norm(rot,axis=-1, keepdims=True)
norm2 = np.linalg.norm(prev_rot,axis=-1, keepdims=True)
# rot2 = -(2*np.pi-norm)*rot_stream[i]/norm
norm3 = (2*np.pi-norm2)
np.any(np.abs(rot-prev_rot)>=1e-1) and np.abs(norm - norm3) <=1e-1
np.linalg.norm(a[303,12:15])
np.linalg.norm(a[304,12:15])
a[303:305,12:15]
plt.matshow(a[290:305])
plt.matshow(a[300:305])
plt.matshow(a[:400])
plt.matshow(a[:1000])
plt.matshow(a[200:300,27:31])
a[10000:10100,:10]
a[10000:10200,:3]
np.unique(a[:,:3])
plt.matshow(b[:,:])
