
import torch
model = torch.jit.load('compiled_jit.pth')

import numpy as np

root_folder = "data/kulzaworld_guille_neosdata_npy_relative/"

import pickle
acts_scaler = pickle.load(open(root_folder+"rel_feats_scaled_scaler.pkl", "rb"))
conds_scaler = pickle.load(open(root_folder+"root_pos_scaled_scaler.pkl", "rb"))
rel_feats_scaled = np.load(root_folder+"data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_2_ID2C00_streams.rel_feats_scaled.npy")
acts_scaler.inverse_transform(rel_feats_scaled)[2000:2010,7:10]
acts_scaler.inverse_transform(rel_feats_scaled)[2000:2010,:3]
root_pos_scaled = np.load(root_folder+"data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_2_ID2C00_streams.root_pos_scaled.npy")

inputs = [torch.from_numpy(root_pos_scaled[2:122]).unsqueeze(1).float().cuda(), torch.from_numpy(rel_feats_scaled[2:122]).unsqueeze(1).float().cuda()]

rel_feats_scaled[2:122].max()

inputs[1].max()
rel_feats_scaled[121]
inputs[1].shape
out = model(inputs, temp=0.1)
out
out = model(inputs)[0][0][:1,0]
print(out.shape)
out
acts_scaler.transform(out.cpu().numpy())
out = model(inputs)
print(out)


################

#%%

import sys
root_dir = "/home/guillefix/code/multimodal-transflower"
sys.path.append(root_dir)

from inference.generate import load_model_from_logs_path

#load hparams file
default_save_path = "data/transflower_expmap_cr4_bs5_og2_futureN_gauss5_rel_single4_kulzaworld_neosraw_rel2/"
logs_path = default_save_path
model, opt = load_model_from_logs_path(logs_path)
