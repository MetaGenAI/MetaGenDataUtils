import grpc
import logging
from concurrent import futures
import pose_interaction_pb2_grpc
from pose_interaction_pb2 import EmptyMessage, RefIDMessage, Heading, Frame

from process_neos_data import get_heading_bytes, get_frame_bytes

class PoseInteractionServicer(pose_interaction_pb2_grpc.PoseInteractionServicer):
    def SendHeadingBytes(self, request, context):
        pass
    def GetHeadingBytes(self, request, context):
        print(request.ref_id)
        bs = get_heading_bytes()
        return Frame(ref_id=request.ref_id, data=bs)
    def SendFrameBytes(self, request, context):
        pass
    def GetFrameBytes(self, request, context):
        print(request.ref_id)
        bs = get_frame_bytes()
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
