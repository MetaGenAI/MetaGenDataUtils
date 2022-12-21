import grpc
import logging
from concurrent import futures
import pose_interaction_pb2_grpc
from pose_interaction_pb2 import EmptyMessage, RefIDMessage, Heading, Frame
import pickle

from process_neos_data import NeosPoseData

import numpy as np
import torch

# root_folder = "data/kulzaworld_guille_neosdata_npy_relative/"
root_folder = "data/quantum_bar_neosdata1_npy_relative/"

import sys
root_dir = "/home/guillefix/code/multimodal-transflower"
sys.path.append(root_dir)


import websockets
import asyncio

class PoseInteractionServicer(pose_interaction_pb2_grpc.PoseInteractionServicer):
    def __init__(self, *args, **kwargs):
        super(*args, **kwargs)
        self.npd = NeosPoseData()
        self.npd.load_json("data/basic_config.json")
        self.use_axis_angle = True
        self.is_relative = True
        # self.use_axis_angle = False
        # self.is_relative = False
        self.prev_frames = None
        self.index = 0
        # self.index = 2
        #example numpy frames for single person. When running interactively with Transflower, this would be obtained via websockets from transflower
        # self.frames = np.load("data/data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48_2_ID2C00_streams.combined_streams_scaled.generated.npy")
        # self.frames = np.load("data/dekaworld_alex_guille_neosdata2/data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48_2_ID2C00_streams.combined_streams.npy")
        # self.frames = np.load("data/dekaworld_alex_guille_neosdata2/data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48_2_ID2C00_streams.combined_streams.npy")
        # self.frames = np.load("data/dekaworld_alex_guille_neosdata2/data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48_2_ID2C00_streams.combined_streams_scaled.npy")
        # self.frames = np.load("data/data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48_2_ID2C00_streams.combined_streams_scaled.generated.npy")
        # self.frames = np.load("data/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_2_ID2C00_streams.person1_scaled.generated.npy")
        # self.frames = np.load("data/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_2_ID2C00_streams.person1_scaled.generated.npy")
        # self.frames = np.load("data/data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48_2_ID2C00_streams.combined_streams_scaled.generated.npy")
        # self.frames = np.load("data/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_2_ID2C00_streams.proc_feats.npy")
        # self.frames = np.load("data/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.proc_feats.generated.npy")
        # self.frames = np.load("data/transflower_expmap_cr4_bs5_og2_futureN_gauss5_single2_kulzaworld_neosraw/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.proc_feats.generated.npy")

        # self.frames = np.load("data/transflower_expmap_cr4_bs5_og2_futureN_gauss5_rel_combined2_dekaworld_neosraw_rel/data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48_2_ID2C00_streams.rel_feats_scaled1.generated.npy")

        # self.frames = np.load("data/moglow_expmap1_tf2_single_kulzaworld_neosraw/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.person1_scaled.generated.npy")
        # self.frames = np.load("data/transflower_expmap_cr4_bs5_og2_futureN_gauss5_single3_kulzaworld_neosraw_fixed/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.proc_feats_scaled.generated.npy")
        # self.frames = np.load("data/transflower_expmap_cr4_bs5_og2_futureN_gauss5_rel_single2_kulzaworld_neosraw_rel/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.rel_feats_scaled.generated.npy")

        # self.frames = np.load("data/kulzaworld_guille_neosdata_npy_relative/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.dat.person1.npy")
        # self.using_model = True
        self.using_model = True
        # n = 120
        n = 60
        if self.using_model:
            from inference.generate import load_model_from_logs_path
            # frames = np.load("data/kulzaworld_guille_neosdata_npy_relative/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.dat.person1.npy")
            # self.framees = frames
            # self.npd.append_concat_frame(frames[:1], use_axis_angle=self.use_axis_angle, is_relative=self.is_relative)
            # self.npd.append_concat_frame(frames[1:2], use_axis_angle=self.use_axis_angle, is_relative=self.is_relative)

            # self.frames = np.load("data/transflower_expmap_cr4_bs5_og2_futureN_gauss5_rel_single2_kulzaworld_neosraw_rel2/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.rel_feats_scaled.generated.npy")
            # rel_feats_scaled = np.load(root_folder+"data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.rel_feats_scaled.npy")
            rel_feats_scaled = np.load(root_folder+"data_quantum_bar_neosdata1_5_IDE51D00_streams.rel_feats_scaled1.npy")
            # self.frames = rel_feats_scaled

            # self.acts_scaler = pickle.load(open(root_folder+"rel_feats_scaled_scaler.pkl", "rb"))
            # self.conds_scaler = pickle.load(open(root_folder+"root_pos_scaled_scaler.pkl", "rb"))
            self.acts_scaler = pickle.load(open(root_folder+"rel_feats_scaled1_scaler.pkl", "rb"))
            self.conds_scaler = pickle.load(open(root_folder+"envelope_scaled_scaler.pkl", "rb"))

            # self.prev_frames = self.acts_scaler.inverse_transform(rel_feats_scaled[:120])
            # self.prev_frames = self.acts_scaler.inverse_transform(rel_feats_scaled)
            # self.prev_frames = rel_feats_scaled
            self.prev_frames = rel_feats_scaled[:n]

            # root_pos_scaled = np.load(root_folder+"data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.root_pos_scaled.npy")
            root_pos_scaled = np.load(root_folder+"data_quantum_bar_neosdata1_5_IDE51D00_voice.ogg_envelope_scaled.npy")
            # self.conds = self.conds_scaler.inverse_transform(root_pos_scaled)
            self.conds = root_pos_scaled

            # self.model = torch.jit.load('compiled_jit.pth')
            # default_save_path = "data/transflower_expmap_cr4_bs5_og2_futureN_gauss5_rel_single4_kulzaworld_neosraw_rel2/"
            # default_save_path = "data/transflower_expmap_cr4_bs5_og2_futureN_gauss5_rel_single2_quantum_bar_rel/"
            # default_save_path = "data/transflower_expmap_cr4_bs5_og2_futureN_gauss5_rel_single2_quantum_bar_rel/"
            default_save_path = "data/transflower_expmap_cr4_bs5_og2_futureN_gauss5_rel_single3_nodp_smol_quantum_bar_rel_nodp/"
            # default_save_path = "data/transflower_expmap_cr4_bs5_og2_futureN_gauss5_rel_single3_quantum_bar_rel_nodp/"
            # default_save_path = "data/transflower_expmap_cr4_bs5_og2_futureN_gauss5_rel_single3_dp_quantum_bar_rel_dp/"
            # default_save_path = "data/transflower_expmap_cr4_bs5_og2_futureN_gauss5_rel_single2_quantum_bar_rel_nodp/"
            logs_path = default_save_path
            model, opt = load_model_from_logs_path(logs_path, version_index=-1)
            self.model = model
            self.temp = 0.05
            # self.temp = 0.13
            # self.temp = 1.0
            # self.temp = 0.2

            # inputs = self.make_inputs(self.conds[self.index:self.index+120], self.prev_frames)
            # inputs = self.make_inputs(self.conds[self.index:self.index+120], self.prev_frames[self.index:self.index+120])
            # # print(inputs)
            # frame = self.model(inputs)[0][0][:1,0].cpu().numpy()
            # frame = self.acts_scaler.inverse_transform(frame)
            # print(frame)
            # frame = self.model(inputs)[0][0][:1,0].cpu().numpy()
            # frame = self.acts_scaler.inverse_transform(frame)
            # print(frame)
            print("Loaded model")
        else:
            # self.frames = np.load("data/transflower_expmap_cr4_bs5_og2_futureN_gauss5_rel_single4_kulzaworld_neosraw_rel2/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.rel_feats_scaled.generated.npy")
            self.frames = np.load("data/transflower_expmap_cr4_bs5_og2_futureN_gauss5_rel_single4_kulzaworld_neosraw_rel2/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.rel_feats_scaled.generated.npy")
            if len(self.frames.shape) == 3: #for generated ones
                self.frames = self.frames[:,0]
            print(self.frames.shape)

        self.last_frame = None
        self.second_last_frame = None

        # self.frames = np.load("data/transflower_expmap_cr4_bs5_og2_futureN_gauss5_rel_single4_kulzaworld_neosraw_rel2/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.rel_feats_scaled.generated.npy")
        # self.frames = np.load("data/transflower_expmap_cr4_bs5_og2_futureN_gauss5_rel_single3_kulzaworld_neosraw_rel_smol/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.rel_feats_scaled.generated.npy")
        # self.frames = np.load("data/transflower_expmap_cr4_bs5_og2_futureN_gauss5_rel_single2_kulzaworld_neosraw_rel_nonshuff/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.rel_feats_scaled.generated.npy")
        # self.frames = np.load("data/discrete_model_kulzaworld_neosraw_rel2/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.rel_feats_scaled.generated.npy")
        # self.frames = np.load("data/discrete_model2_kulzaworld_neosraw_rel2/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.rel_feats_scaled.generated.npy")
        # self.frames = np.load("data/discrete_model_kulzaworld_neosraw_rel/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.rel_feats_scaled.generated.npy")

        # self.frames = np.load("data/moglow_expmap1_tf3_rel_single_kulzaworld_neosraw_rel/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.rel_feats_scaled.generated.npy")
        # self.frames = np.load("data/kulzaworld_guille_neosdata_npy/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.dat.person1.npy")
        # self.frames = np.load("data/kulzaworld_guille_neosdata_npy_axis_angle/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.dat.person1.npy")
        # self.npd.append_concat_frame(self.frames[1000:1001], use_axis_angle=True, is_relative=False)


        # self.frames = np.load("data/kulzaworld_guille_neosdata_npy_axis_angle/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.proc_feats.npy")
        # self.frames = np.load("data/kulzaworld_guille_neosdata_npy_axis_angle/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.dat.person1.npy")
        # self.frames = np.load("data/kulzaworld_guille_neosdata_npy_testing/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.dat.person1.rel.npy")
        # self.frames = np.load("data/kulzaworld_guille_neosdata_npy_relative/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.dat.person1.rel.npy")
        # self.frames = np.load("data/moglow_expmap1_tf3_single_kulzaworld_neosraw/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.proc_feats.generated.npy")

        # transform = pickle.load(open("data/dekaworld_alex_guille_neosdata2/combined_streams_scaled_scaler.pkl", "rb"))
        # self.frames = transform.inverse_transform(self.frames)
        # self.frames = np.load("data/dekaworld_alex_guille_neosdata2/data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48_1_ID1E66900_streams.combined_streams.npy")
        # self.frames = np.load("data/data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48_1_ID1E66900_streams.combined_streams.npy")
        # self.frames = np.load("data/numpys/data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48_1_ID1E66900_streams.dat.person1.npy")
        # self.frames = np.load("data/numpys/data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48_2_ID2C00_streams.dat.person1.npy")
        # self.frames = np.load("data/numpys/data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48_2_ID2C00_streams.dat.person1.npy")
        # self.frames = np.load("data/kulzaworld_guille_neosdata_npy/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_2_ID2C00_streams.dat.person1.npy")

        # self.frames[3:6] = np.cumsum(self.frames[3:6], axis=0)
        # self.frames = self.frames[:,0,:self.frames.shape[2]//2]
        # self.frames = self.frames[:,0,self.frames.shape[2]//2:]
        # self.frames = self.frames[:,:self.frames.shape[1]//2]
        # self.frames = self.frames[9000:,:self.frames.shape[1]//2]
        # self.frames = self.frames[9000:,]
        # self.frames = np.load("data/example_numpy_frames.npy")
    def make_inputs(self, conds, acts):
        # conds = self.conds_scaler.transform(conds)
        # acts = self.acts_scaler.transform(acts)
        inputs = [torch.from_numpy(conds).unsqueeze(1).float().cuda(), torch.from_numpy(acts).unsqueeze(1).float().cuda()]
        return inputs
    def SendHeadingBytes(self, request, context):
        return EmptyMessage()
    def GetHeadingBytes(self, request, context):
        print("GetHeadingBytes")
        print(request.ref_id)
        self.index = 0
        self.npd.load_json("data/basic_config.json")
        bs = self.npd.get_heading_bytes()
        # print(bs)
        return Heading(ref_id=request.ref_id, data=bs)
        # return Heading(ref_id="IDC00", data=b'')
    def SendFrameBytes(self, request, context):
        return EmptyMessage()
    def GetFrameBytes(self, request, context):
        # print("GetFrameBytes")
        # print(request.ref_id)
        
        # framee = self.framees[self.index:self.index+1]
        # print(framee)
        if self.using_model:
            # print("HO")
            # if self.index == 0:
            if True: #self.index % 2 == 0:
                with torch.no_grad():
                    # inputs = self.make_inputs(self.conds[self.index:self.index+60].copy(), self.prev_frames.copy())
                    inputs = self.make_inputs(self.conds[self.index:self.index+60], self.prev_frames)
                    # inputs = self.make_inputs(self.conds_scaler.transform(0.05*np.sin(np.array(range(self.index-1,self.index)))), self.prev_frames.copy())
                    # inputs = self.make_inputs(np.expand_dims(np.sin(np.array(range(self.index-60,self.index))),1), self.prev_frames.copy())
                    # inputs = self.make_inputs(self.prev_conds.copy(), self.prev_frames.copy())
                    # print(inputs)
                    # inputs = self.make_inputs(self.conds[self.index:self.index+120], self.prev_frames[self.index:self.index+120])
                    # frame = self.model(inputs)[0][0][:1,0].cpu().numpy()
                    frame_scaled = self.model.forward(inputs, temp=self.temp)[0][0][:1,0].cpu().numpy()
                # print(frame_scaled)
                frame = self.acts_scaler.inverse_transform(frame_scaled)
                # print(frame.shape)
                # frame = self.prev_frames[self.index:self.index+1]
                # frame[:,:3] = np.array([[0.85,0.85,0.85]])
                # frame[:,6:10] = np.array([[0,0,0,0]])
                frame[:,16:17] = np.array([[0]])
                frame[:,11:13] = np.array([[0,0]])
                frame[:,9:10] = np.array([[0]])
                frame[:,4:6] = np.array([[0,0]])
                # frame[:,6:10] = 0
                # frame[:,7:10] = np.array([[-3.14,0,0]])
                # frame[:,10:13] = np.array([[0,0,0]])
                # print(frame)
            else:
                frame = self.last_frame
                if self.second_last_frame is not None:
                    frame = self.last_frame + (self.last_frame - self.second_last_frame)
                frame_scaled = self.acts_scaler.transform(frame)
            self.prev_frames = np.concatenate([self.prev_frames[1:], frame_scaled])
        else:
            frame = self.frames[self.index:self.index+1]
        self.npd.append_concat_frame(frame, use_axis_angle=self.use_axis_angle, is_relative=self.is_relative)
        self.last_frame = frame
        self.second_last_frame = self.last_frame
        # if self.use_axis_angle:
        #     self.npd.convert_axis_angles_to_quaternions(only_last_frame=True)
        bs = self.npd.get_frame_bytes(self.index)
        self.index += 1
        # print(len(bs))
        return Frame(ref_id=request.ref_id, data=bs)

class PoseInteractionServicerTesting(pose_interaction_pb2_grpc.PoseInteractionServicer):
    def __init__(self, *args, **kwargs):
        super(*args, **kwargs)
        self.npd = NeosPoseData("data/example/1/ID2C00_streams.dat")
    def SendHeadingBytes(self, request, context):
        return EmptyMessage()
    def GetHeadingBytes(self, request, context):
        print("GetHeadingBytes")
        print(request.ref_id)
        self.npd = NeosPoseData("data/example/1/ID2C00_streams.dat")
        bs,_ = self.npd.process_heading_bytes()
        # print(bs)
        return Heading(ref_id=request.ref_id, data=bs)
        # return Heading(ref_id="IDC00", data=b'')
    def SendFrameBytes(self, request, context):
        return EmptyMessage()
    def GetFrameBytes(self, request, context):
        print("GetFrameBytes")
        print(request.ref_id)
        bs,_ = self.npd.process_frame_bytes()
        # print(len(bs))
        return Frame(ref_id=request.ref_id, data=bs)

from grpc import aio

async def serve():
    server = aio.server(futures.ThreadPoolExecutor(max_workers=10))
    pose_interaction_pb2_grpc.add_PoseInteractionServicer_to_server(
        PoseInteractionServicer(), server)
    server.add_insecure_port('[::]:40052')
    await server.start()
    await server.wait_for_termination()

if __name__ == '__main__':
    # logging.basicConfig()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    print("HII")
    async def loop1(websocket,path):
        while 1:
            print("hi")
            response = await websocket.recv()
            print(response)
    start_server = websockets.serve(loop1, "localhost", "8766")
    async def loop2():
        await serve()
    loop.run_until_complete(asyncio.gather(start_server, loop2()))
    loop.run_forever()
