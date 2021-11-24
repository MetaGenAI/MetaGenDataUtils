import grpc
import logging
from concurrent import futures
import pose_interaction_pb2_grpc
from pose_interaction_pb2 import EmptyMessage, RefIDMessage, Heading, Frame

from process_neos_data import NeosPoseData

import numpy as np

class PoseInteractionServicer(pose_interaction_pb2_grpc.PoseInteractionServicer):
    def __init__(self, *args, **kwargs):
        super(*args, **kwargs)
        self.npd = NeosPoseData()
        self.npd.load_json("data/basic_config.json")
        #example numpy frames for single person. When running interactively with Transflower, this would be obtained via websockets from transflower
        self.frames = np.load("data/data_U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40_S_d03a6c7b.npy")
        self.frames = self.frames[:,0,:self.frames.shape[2]//2]
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
        # print("GetFrameBytes")
        # print(request.ref_id)
        self.npd.append_concat_frame(self.frames[self.index:self.index+1])
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
