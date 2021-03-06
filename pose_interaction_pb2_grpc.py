# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import pose_interaction_pb2 as pose__interaction__pb2


class PoseInteractionStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.SendHeadingBytes = channel.unary_unary(
                '/PoseInteraction/SendHeadingBytes',
                request_serializer=pose__interaction__pb2.Heading.SerializeToString,
                response_deserializer=pose__interaction__pb2.EmptyMessage.FromString,
                )
        self.GetHeadingBytes = channel.unary_unary(
                '/PoseInteraction/GetHeadingBytes',
                request_serializer=pose__interaction__pb2.RefIDMessage.SerializeToString,
                response_deserializer=pose__interaction__pb2.Heading.FromString,
                )
        self.SendFrameBytes = channel.unary_unary(
                '/PoseInteraction/SendFrameBytes',
                request_serializer=pose__interaction__pb2.Frame.SerializeToString,
                response_deserializer=pose__interaction__pb2.EmptyMessage.FromString,
                )
        self.GetFrameBytes = channel.unary_unary(
                '/PoseInteraction/GetFrameBytes',
                request_serializer=pose__interaction__pb2.RefIDMessage.SerializeToString,
                response_deserializer=pose__interaction__pb2.Frame.FromString,
                )


class PoseInteractionServicer(object):
    """Missing associated documentation comment in .proto file."""

    def SendHeadingBytes(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetHeadingBytes(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SendFrameBytes(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetFrameBytes(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_PoseInteractionServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'SendHeadingBytes': grpc.unary_unary_rpc_method_handler(
                    servicer.SendHeadingBytes,
                    request_deserializer=pose__interaction__pb2.Heading.FromString,
                    response_serializer=pose__interaction__pb2.EmptyMessage.SerializeToString,
            ),
            'GetHeadingBytes': grpc.unary_unary_rpc_method_handler(
                    servicer.GetHeadingBytes,
                    request_deserializer=pose__interaction__pb2.RefIDMessage.FromString,
                    response_serializer=pose__interaction__pb2.Heading.SerializeToString,
            ),
            'SendFrameBytes': grpc.unary_unary_rpc_method_handler(
                    servicer.SendFrameBytes,
                    request_deserializer=pose__interaction__pb2.Frame.FromString,
                    response_serializer=pose__interaction__pb2.EmptyMessage.SerializeToString,
            ),
            'GetFrameBytes': grpc.unary_unary_rpc_method_handler(
                    servicer.GetFrameBytes,
                    request_deserializer=pose__interaction__pb2.RefIDMessage.FromString,
                    response_serializer=pose__interaction__pb2.Frame.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'PoseInteraction', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class PoseInteraction(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def SendHeadingBytes(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/PoseInteraction/SendHeadingBytes',
            pose__interaction__pb2.Heading.SerializeToString,
            pose__interaction__pb2.EmptyMessage.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetHeadingBytes(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/PoseInteraction/GetHeadingBytes',
            pose__interaction__pb2.RefIDMessage.SerializeToString,
            pose__interaction__pb2.Heading.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SendFrameBytes(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/PoseInteraction/SendFrameBytes',
            pose__interaction__pb2.Frame.SerializeToString,
            pose__interaction__pb2.EmptyMessage.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetFrameBytes(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/PoseInteraction/GetFrameBytes',
            pose__interaction__pb2.RefIDMessage.SerializeToString,
            pose__interaction__pb2.Frame.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
