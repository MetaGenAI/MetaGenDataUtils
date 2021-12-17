import grpc
import logging
from concurrent import futures
import pose_interaction_pb2_grpc
from pose_interaction_pb2 import EmptyMessage, RefIDMessage, Heading, Frame
import pickle

from process_neos_data import NeosPoseData

import numpy as np

class PoseInteractionServicer(pose_interaction_pb2_grpc.PoseInteractionServicer):
    def __init__(self, *args, **kwargs):
        super(*args, **kwargs)
        self.npd = NeosPoseData()
        self.npd.load_json("data/basic_config.json")
        self.use_axis_angle = True
        # self.use_axis_angle = False
        # self.is_relative = False
        self.is_relative = True
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

        # self.frames = np.load("data/transflower_expmap_cr4_bs5_og2_futureN_gauss5_single2_kulzaworld_neosraw/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.proc_feats.generated.npy")
        # self.frames = np.load("data/kulzaworld_guille_neosdata_npy/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.proc_feats.npy")
        # self.frames = np.load("data/kulzaworld_guille_neosdata_npy_axis_angle/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.proc_feats.npy")
        # self.frames = np.load("data/kulzaworld_guille_neosdata_npy_axis_angle/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.dat.person1.npy")
        # self.frames = np.load("data/kulzaworld_guille_neosdata_npy_testing/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.dat.person1.npy")
        self.frames = np.load("data/kulzaworld_guille_neosdata_npy_relative/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.dat.person1.rel.npy")
        # self.frames = np.load("data/moglow_expmap1_tf3_single_kulzaworld_neosraw/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_3_ID2C00_streams.proc_feats.generated.npy")

        # transform = pickle.load(open("data/dekaworld_alex_guille_neosdata2/combined_streams_scaled_scaler.pkl", "rb"))
        # self.frames = transform.inverse_transform(self.frames)
        # self.frames = np.load("data/dekaworld_alex_guille_neosdata2/data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48_1_ID1E66900_streams.combined_streams.npy")
        # self.frames = np.load("data/data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48_1_ID1E66900_streams.combined_streams.npy")
        # self.frames = np.load("data/numpys/data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48_1_ID1E66900_streams.dat.person1.npy")
        # self.frames = np.load("data/numpys/data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48_2_ID2C00_streams.dat.person1.npy")
        # self.frames = np.load("data/numpys/data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48_2_ID2C00_streams.dat.person1.npy")
        # self.frames = np.load("data/kulzaworld_guille_neosdata_npy/data_kulzaworld_guille_neosdata_U_Kulza_R_57ea6247_a178_45c5_a3bb_a95af490bfb0_S-898a7978-79fa-4fd0-8f4d-e7cfb8a1e397_a06ffd39-1343-4854-8d2f-225156c7cf5d_2_ID2C00_streams.dat.person1.npy")

        if len(self.frames.shape) == 3: #for generated ones
            self.frames = self.frames[:,0]
        # self.frames[3:6] = np.cumsum(self.frames[3:6], axis=0)
        # self.frames = self.frames[:,0,:self.frames.shape[2]//2]
        # self.frames = self.frames[:,0,self.frames.shape[2]//2:]
        # self.frames = self.frames[:,:self.frames.shape[1]//2]
        # self.frames = self.frames[9000:,:self.frames.shape[1]//2]
        # self.frames = self.frames[9000:,]
        # self.frames = np.load("data/example_numpy_frames.npy")
        print(self.frames.shape)
        self.index = 0
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
        print("GetFrameBytes")
        print(request.ref_id)
        self.npd.append_concat_frame(self.frames[self.index:self.index+1], use_axis_angle=self.use_axis_angle, is_relative=self.is_relative)
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

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pose_interaction_pb2_grpc.add_PoseInteractionServicer_to_server(
        PoseInteractionServicer(), server)
    server.add_insecure_port('[::]:40052')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()
