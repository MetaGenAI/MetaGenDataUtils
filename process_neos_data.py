import numpy as np
import os
import json
from scipy.spatial.transform import Rotation as R

# fileName = "data/example/1/ID2C00_streams.dat"
# file = open(fileName, mode='rb')
# fileContent = file.read()

oldToNewNodeInts = {
1:1,
9:10,
16:17,
45:46,
}

#%%
import struct
import numpy as np

def convert_quaternions_to_axis_angles(quats):
    Rs = R.from_quat(quats)
    return Rs.as_rotvec()

def convert_axis_angles_to_quaternions(axangs):
    Rs = R.from_rotvec(axangs)
    return Rs.as_quat()

def read_float(data_iter):
    bs = [next(data_iter,None) for _ in range(4)]
    if np.any(np.array(bs) == None):
        return None,None
    bs = bytes(bs)
    # bytes = b''.join(bytes)
    x, = struct.unpack('f',bs)
    return x,bs

def float_to_bytes(f):
    bs = bytearray(struct.pack("f", f))
    return bs

def read_int(data_iter):
    bs = [next(data_iter,None) for _ in range(4)]
    if np.any(np.array(bs) == None):
        return None,None
    bs = bytes(bs)
    # bytes = b''.join(bytes)
    x, = struct.unpack('i',bs)
    return x,bs

def int_to_bytes(i):
    bs = bytearray(struct.pack("i", i))
    return bs

def read_bool(data_iter):
    bs = [next(data_iter,None) for _ in range(1)]
    if np.any(np.array(bs) == None):
        return None,None
    bs = bytes(bs)
    # bytes = b''.join(bytes)
    x, = struct.unpack('?',bs)
    return x,bs

def bool_to_bytes(b):
    bs = bytearray(struct.pack("?", b))
    return bs
#%%

######HEADING
class NeosPoseData(): #Single-person pose data
    def __init__(self, fileName = None, data_iter = None):
    # fileName = "data/example/1/ID2C00_streams.dat"
        self.data = {}
        self.num_frames = 0
        self.data["is_relative"] = False
        self.data["rot_param"] = "quat"
        if data_iter is not None:
            self.data_iter = data_iter
            self.load_data()
        elif fileName is not None:
            self.file = open(fileName, mode='rb')
            self.fileContent = self.file.read()
            self.data_iter = iter(self.fileContent)
            self.load_data()

    def process_heading_bytes(self):
        body_node_data = {}
        self.data = {}
        body_node_list = []
        data_iter = self.data_iter
        # READ absolute time
        bs = bytes()
        absolute_time,b = read_float(data_iter)
        self.data["absolute_time"] = absolute_time
        print(absolute_time)
        bs+=b

        # READ version number
        version_number,b = read_int(data_iter)
        self.data["version_number"] = version_number
        print(version_number)
        bs+=b

        num_body_nodes = version_number
        if version_number >= 1000:
            #READ relative avatar scale
            rel_avatar_scale_x,b = read_float(data_iter)
            self.data["rel_avatar_scale_x"] = rel_avatar_scale_x
            print(rel_avatar_scale_x)
            bs+=b
            rel_avatar_scale_y,b = read_float(data_iter)
            self.data["rel_avatar_scale_y"] = rel_avatar_scale_y
            print(rel_avatar_scale_y)
            bs+=b
            rel_avatar_scale_z,b = read_float(data_iter)
            self.data["rel_avatar_scale_z"] = rel_avatar_scale_z
            print(rel_avatar_scale_z)
            bs+=b
            #READ number of body nodes
            num_body_nodes,b = read_int(data_iter)
            print(num_body_nodes)
            bs+=b
        self.data["num_body_nodes"] = num_body_nodes

        for i in range(num_body_nodes):
            #READ body node type
            nodeInt,b = read_int(data_iter)
            bs+=b
            if version_number < 1000:
                nodeInt = oldToNewNodeInts[nodeInt]
            nodeInt = str(nodeInt)
            body_node_data[nodeInt] = {}
            body_node_list.append(nodeInt)
            print(nodeInt)
            #READ if scale stream exists
            scale_exists,b = read_bool(data_iter)
            print(scale_exists)
            bs+=b
            body_node_data[nodeInt]["scale_exists"] = scale_exists
            if scale_exists:
                body_node_data[nodeInt]["scale_stream"] = []
            #READ if position stream exists
            pos_exists,b = read_bool(data_iter)
            print(pos_exists)
            bs+=b
            body_node_data[nodeInt]["pos_exists"] = pos_exists
            if pos_exists:
                body_node_data[nodeInt]["pos_stream"] = []
            #READ if rotation stream exists
            rot_exists,b = read_bool(data_iter)
            print(rot_exists)
            bs+=b
            body_node_data[nodeInt]["rot_exists"] = rot_exists
            if rot_exists:
                body_node_data[nodeInt]["rot_stream"] = []

        self.data["body_node_data"] = body_node_data
        self.data["body_node_list"] = body_node_list
        #READ whether hands are tracked
        hands_are_tracked,b = read_bool(data_iter)
        self.data["hands_are_tracked"] = hands_are_tracked
        print(hands_are_tracked)
        bs+=b
        #READ whether metacarpals are tracked
        metacarpals_are_tracked,b = read_bool(data_iter)
        self.data["metacarpals_are_tracked"] = metacarpals_are_tracked
        print(metacarpals_are_tracked)
        bs+=b
        self.data["deltaTs"] = []
        self.data["left_hand_rots"] = []
        self.data["right_hand_rots"] = []
        return bs, self.data

    #######STREAM

    # while True:
    def process_frame_bytes(self, include_was_succesful=False):
        data_iter = self.data_iter
        body_node_list = self.data["body_node_list"]
        body_node_data = self.data["body_node_data"]
        #READ deltaT
        bs=bytes()
        deltaT,b = read_float(data_iter)
        self.data["deltaTs"].append(deltaT)
        # print(deltaT)
        if deltaT is None:
            # break
            return None
        bs+=b
        #READ bodyNode transforms
        for key in body_node_list:
            bodyNode = body_node_data[key]
            # print(bodyNode)
            if bodyNode["scale_exists"]:
                scalex,b = read_float(data_iter)
                bs+=b
                scaley,b = read_float(data_iter)
                bs+=b
                scalez,b = read_float(data_iter)
                bs+=b
                scale = [scalex,scaley,scalez]
                bodyNode["scale_stream"] += [scale]
            if bodyNode["pos_exists"]:
                posx,b = read_float(data_iter)
                bs+=b
                posy,b = read_float(data_iter)
                bs+=b
                posz,b = read_float(data_iter)
                bs+=b
                pos = [posx,posy,posz]
                # print(key, pos)
                bodyNode["pos_stream"] += [pos]
            if bodyNode["rot_exists"]:
                rotx,b = read_float(data_iter)
                bs+=b
                roty,b = read_float(data_iter)
                bs+=b
                rotz,b = read_float(data_iter)
                bs+=b
                rotw,b = read_float(data_iter)
                bs+=b
                rot = [rotx,roty,rotz,rotw]
                bodyNode["rot_stream"] += [rot]
        #READ finger poses
        if self.data["hands_are_tracked"]:
            # print("HI")
            #Left hand
            left_hand_rots = []
            for i in range(23):
                was_succesful,b = read_bool(data_iter)
                if include_was_succesful:
                    left_hand_rots.append(was_succesful)
                bs+=b
                x,b = read_float(data_iter)
                left_hand_rots.append(x)
                bs+=b
                y,b = read_float(data_iter)
                left_hand_rots.append(y)
                bs+=b
                z,b = read_float(data_iter)
                left_hand_rots.append(z)
                bs+=b
                w,b = read_float(data_iter)
                left_hand_rots.append(w)
                bs+=b

            self.data["left_hand_rots"].append(left_hand_rots)
            #Right hand
            right_hand_rots = []
            for i in range(23):
                was_succesful,b = read_bool(data_iter)
                if include_was_succesful:
                    right_hand_rots.append(was_succesful)
                bs+=b
                x,b = read_float(data_iter)
                right_hand_rots.append(x)
                bs+=b
                y,b = read_float(data_iter)
                right_hand_rots.append(y)
                bs+=b
                z,b = read_float(data_iter)
                right_hand_rots.append(z)
                bs+=b
                w,b = read_float(data_iter)
                right_hand_rots.append(w)
                bs+=b
            self.data["right_hand_rots"].append(right_hand_rots)
        return bs, self.data

    def _get_all_frames(self, include_was_succesful=False):
        self.num_frames = 0
        while self.process_frame_bytes(include_was_succesful=include_was_succesful) is not None:
            self.num_frames += 1
        self.data["num_frames"] = self.num_frames
        return self.data

    # def load_data(self, include_was_succesful=False):
    #     self.process_heading_bytes()
    #     data = self.get_all_frames(include_was_succesful=include_was_succesful)
    #     return data

    def load_data(self, include_was_succesful=False):
        self.process_heading_bytes()
        data = self._get_all_frames(include_was_succesful=include_was_succesful)
        for node in data["body_node_list"]:
            d = data["body_node_data"][node]
            if d["scale_exists"]:
                d["scale_stream"] = np.array(d["scale_stream"])
            if d["pos_exists"]:
                d["pos_stream"] = np.array(d["pos_stream"])
            if d["rot_exists"]:
                d["rot_stream"] = np.array(d["rot_stream"])
        if data["hands_are_tracked"]:
            data["left_hand_rots"] = np.array(data["left_hand_rots"])
            data["right_hand_rots"] = np.array(data["right_hand_rots"])
        self.data = data
        return data

    def clear_trajectories(self):
        data = self.data
        for node in data["body_node_list"]:
            d = data["body_node_data"][node]
            if "scale_exists" in d:
                d["scale_stream"] = []
            if "pos_exists" in d:
                d["pos_stream"] = []
            if "rot_exists" in d:
                d["rot_stream"] = []
        data["left_hand_rots"] = []
        data["right_hand_rots"] = []
        data["deltaTs"] = []
        self.data = data

    def save_json(self, fileName):
        # print(self.data)
        json.dump(self.data, open(fileName, "w"))

    def load_json(self, fileName):
        self.data = json.load(open(fileName, "r"))
        if "num_frames" in self.data:
            self.num_frames = self.data["num_frames"]

    def load_heading_json(self, fileName):
        data = self.data
        new_data = json.load(open(fileName, "r"))
        if data != {}:
            for node in new_data["body_node_list"]:
                print(node)
                assert str(node) in data["body_node_data"]
        del new_data["body_node_data"]
        # del new_data["hands_are_tracked"]
        del new_data["left_hand_rots"]
        del new_data["right_hand_rots"]
        del new_data["num_frames"]
        data.update(new_data)
        self.data = data

    def load_frames_json(self, fileName):
        data = self.data
        new_data = json.load(open(fileName, "r"))
        for node in data["body_node_list"]:
            assert str(node) in new_data["body_node_data"]
        data["body_node_data"] = new_data["body_node_data"]
        data["hands_are_tracked"] = new_data["hands_are_tracked"]
        data["left_hand_rots"] = new_data["left_hand_rots"]
        data["right_hand_rots"] = new_data["right_hand_rots"]
        data["num_frames"] = new_data["num_frames"]
        self.data = data
        if "num_frames" in self.data:
            self.num_frames = self.data["num_frames"]

    def concat_numpy(self):
        arrays = []
        data = self.data
        relative = data["is_relative"]
        for node in data["body_node_list"]:
            d = data["body_node_data"][node]
            if d["scale_exists"]:
                arrays.append(d["scale_stream"])
            if d["pos_exists"]:
                arrays.append(d["pos_stream"])
                if node in ["1","10"] and relative:
                    print(d["deltas_rel"].shape)
                    arrays.append(d["deltas_rel"])
            if d["rot_exists"]:
                arrays.append(d["rot_stream"])
                if node in ["1","10"] and relative:
                    print(d["thetas_diff"].shape)
                    arrays.append(d["thetas_diff"])
        if data["hands_are_tracked"]:
            arrays.append(data["left_hand_rots"])
            arrays.append(data["right_hand_rots"])

        arrays = np.concatenate(arrays, axis=1)
        return arrays

    def append_concat_frame(self, frame, deltaT=33.33, use_axis_angle=False, is_relative=False):
        assert self.data["rot_param"] == "quat"
        assert self.data["is_relative"] == False
        if frame.shape[0]>0:
            i = 0
            data = self.data
            for node in data["body_node_list"]:
                node = str(node)
                d = data["body_node_data"][node]
                if d["scale_exists"]:
                    if len(d["scale_stream"]) == 0 or d["scale_stream"].shape[0] == 0:
                        d["scale_stream"] = frame[:,i:i+3]
                    else:
                        d["scale_stream"] = np.concatenate([d["scale_stream"],frame[:,i:i+3]])
                    i+=3
                if d["pos_exists"]:
                    if node in ["1", "10"] and is_relative:
                        pos_y = frame[:,i:i+1]
                        pos = np.zeros((frame.shape[0],3))
                        i+=1
                    else:
                        pos = frame[:,i:i+3]
                        i+=3
                    if len(d["pos_stream"]) == 0 or d["pos_stream"].shape[0] == 0:
                        d["pos_stream"] = pos
                    else:
                        d["pos_stream"] = np.concatenate([d["pos_stream"],pos])
                    if node in ["1", "10"] and is_relative:
                        d["pos_stream_y"] = pos_y
                        d["deltas_rel"] = frame[:,i:i+2]
                        i+=2
                if d["rot_exists"]:
                    if use_axis_angle:
                        rot =convert_axis_angles_to_quaternions(frame[:,i:i+3])
                        i+=3
                    else:
                        rot = frame[:,i:i+4]
                        i+=4
                    if node in ["1", "10"] and is_relative:
                        d["thetas_diff"] = frame[:,i:i+1]
                        i+=1
                    if len(d["rot_stream"]) == 0 or d["rot_stream"].shape[0] == 0:
                        d["rot_stream"] = rot
                    else:
                        d["rot_stream"] = np.concatenate([d["rot_stream"],rot])
            if data["hands_are_tracked"]:
                if len(data["left_hand_rots"]) == 0 or data["left_hand_rots"].shape[0] == 0:
                    if use_axis_angle:
                        data["left_hand_rots"] = frame[:,i:i+23*3]
                        num_frames = frame.shape[0]
                        data["left_hand_rots"] = data["left_hand_rots"].reshape((num_frames*23,-1))
                        data["left_hand_rots"] = convert_axis_angles_to_quaternions(data["left_hand_rots"])
                        data["left_hand_rots"] = data["left_hand_rots"].reshape((num_frames,-1))
                    else:
                        data["left_hand_rots"] = frame[:,i:i+23*4]
                else:
                    if use_axis_angle:
                        frame_chunk = frame[:,i:i+23*3]
                        num_frames = frame_chunk.shape[0]
                        frame_chunk = frame_chunk.reshape((num_frames*23,-1))
                        frame_chunk = convert_axis_angles_to_quaternions(frame_chunk)
                        frame_chunk = frame_chunk.reshape((num_frames,-1))
                        frame_chunk = np.concatenate([frame_chunk,frame_chunk])
                    else:
                        data["left_hand_rots"] = np.concatenate([data["left_hand_rots"],frame[:,i:i+23*4]])
                if use_axis_angle:
                    i+=23*3
                else:
                    i+=23*4
                if len(data["right_hand_rots"]) == 0 or data["right_hand_rots"].shape[0] == 0:
                    if use_axis_angle:
                        frame_chunk = frame[:,i:i+23*3]
                        num_frames = frame_chunk.shape[0]
                        frame_chunk = frame_chunk.reshape((num_frames*23,-1))
                        frame_chunk = convert_axis_angles_to_quaternions(frame_chunk)
                        frame_chunk = frame_chunk.reshape((num_frames,-1))
                        data["right_hand_rots"] = frame_chunk
                    else:
                        data["right_hand_rots"] = frame[:,i:i+23*4]
                else:
                    if use_axis_angle:
                        frame_chunk = frame[:,i:i+23*3]
                        num_frames = frame_chunk.shape[0]
                        frame_chunk = frame_chunk.reshape((num_frames*23,-1))
                        frame_chunk = convert_axis_angles_to_quaternions(frame_chunk)
                        frame_chunk = frame_chunk.reshape((num_frames,-1))
                        data["right_hand_rots"] = np.concatenate([data["right_hand_rots"],frame_chunk])
                    else:
                        data["right_hand_rots"] = np.concatenate([data["right_hand_rots"],frame[:,i:i+23*4]])
                if use_axis_angle:
                    i+=23*3
                else:
                    i+=23*4

            if is_relative:
                self.make_absolute(only_last_frame=True)
            self.data = data
            self.data["deltaTs"].append(deltaT)
            self.num_frames += 1
        else:
            self.data["deltaTs"].append(None)

    def get_frame_bytes(self, frame_index):
        #READ deltaT
        bs=bytes()
        deltaT = self.data["deltaTs"][frame_index]
        # print(deltaT)
        if deltaT is None:
            # break
            return None
        b = float_to_bytes(deltaT)
        bs+=b
        #READ bodyNode transforms
        body_node_list=self.data["body_node_list"]
        body_node_data=self.data["body_node_data"]
        # print(body_node_data)
        for key in body_node_list:
            bodyNode = body_node_data[str(key)]
            # print(bodyNode)
            if bodyNode["scale_exists"]:
                scale = bodyNode["scale_stream"][frame_index]
                b = float_to_bytes(scale[0])
                bs+=b
                b = float_to_bytes(scale[1])
                bs+=b
                b = float_to_bytes(scale[2])
                bs+=b
            if bodyNode["pos_exists"]:
                pos = bodyNode["pos_stream"][frame_index]
                b = float_to_bytes(pos[0])
                bs+=b
                b = float_to_bytes(pos[1])
                bs+=b
                b = float_to_bytes(pos[2])
                bs+=b
            if bodyNode["rot_exists"]:
                rot = bodyNode["rot_stream"][frame_index]
                b = float_to_bytes(rot[0])
                bs+=b
                b = float_to_bytes(rot[1])
                bs+=b
                b = float_to_bytes(rot[2])
                bs+=b
                b = float_to_bytes(rot[3])
                bs+=b
        #READ finger poses
        if self.data["hands_are_tracked"]:
            # print("HI")
            #Left hand
            left_hand_rots = bodyNode["left_hand_rots"][frame_index]
            for i in range(23):
                b = bool_to_bytes(True)
                bs+=b
                b = float_to_bytes(left_hand_rots[i*4+0])
                bs+=b
                b = float_to_bytes(left_hand_rots[i*4+1])
                bs+=b
                b = float_to_bytes(left_hand_rots[i*4+2])
                bs+=b
                b = float_to_bytes(left_hand_rots[i*4+3])
                bs+=b

            #Right hand
            right_hand_rots = bodyNode["right_hand_rots"][frame_index]
            for i in range(23):
                b = bool_to_bytes(True)
                bs+=b
                b = float_to_bytes(right_hand_rots[i*4+0])
                bs+=b
                b = float_to_bytes(right_hand_rots[i*4+1])
                bs+=b
                b = float_to_bytes(right_hand_rots[i*4+2])
                bs+=b
                b = float_to_bytes(right_hand_rots[i*4+3])
                bs+=b
        return bs

    def get_heading_bytes(self):
        # READ absolute time
        bs = bytes()
        b = float_to_bytes(self.data["absolute_time"])
        bs+=b

        # READ version number
        version_number = self.data["version_number"]
        b = int_to_bytes(version_number)
        print(version_number)
        bs+=b

        num_body_nodes = version_number
        if version_number >= 1000:
            #READ relative avatar scale
            b = float_to_bytes(self.data["rel_avatar_scale_x"])
            bs+=b
            b = float_to_bytes(self.data["rel_avatar_scale_y"])
            bs+=b
            b = float_to_bytes(self.data["rel_avatar_scale_z"])
            bs+=b
            #READ number of body nodes
            num_body_nodes = self.data["num_body_nodes"]
            b = int_to_bytes(num_body_nodes)
            bs+=b

        for i in range(num_body_nodes):
            #READ body node type
            nodeInt = self.data["body_node_list"][i]
            # print(self.data["body_node_data"])
            print(nodeInt)
            b = int_to_bytes(int(nodeInt))
            bs+=b
            nodeInt = str(nodeInt)
            #READ if scale stream exists
            scale_exists = self.data["body_node_data"][nodeInt]["scale_exists"]
            b = bool_to_bytes(scale_exists)
            bs+=b
            #READ if position stream exists
            pos_exists = self.data["body_node_data"][nodeInt]["pos_exists"]
            b = bool_to_bytes(pos_exists)
            bs+=b
            #READ if rotation stream exists
            rot_exists = self.data["body_node_data"][nodeInt]["rot_exists"]
            b = bool_to_bytes(rot_exists)
            bs+=b
        #READ whether hands are tracked
        b = bool_to_bytes(self.data["hands_are_tracked"])
        bs+=b
        #READ whether metacarpals are tracked
        b = bool_to_bytes(self.data["metacarpals_are_tracked"])
        bs+=b
        return bs

    def fix_root_big_rescalings(self):
        assert self.data["body_node_data"]["1"]["scale_exists"]
        prev_scale = None
        prev_pos = None
        prev_rot = None
        root_scale_stream = self.data["body_node_data"]["1"]["scale_stream"]
        root_pos_stream = self.data["body_node_data"]["1"]["pos_stream"]
        root_rot_stream = self.data["body_node_data"]["1"]["rot_stream"]
        for i, scale in enumerate(root_scale_stream):
            pos = root_pos_stream[i]
            rot = root_rot_stream[i]
            if prev_scale is None:
                # prev_scale = scale
                prev_scale = np.array([1.0,1.0,1.0])
            if prev_pos is None:
                prev_pos = pos
            if prev_rot is None:
                prev_rot = rot
            if np.any(np.abs(scale-prev_scale) >= prev_scale*0.25) or np.any(np.abs(pos-prev_pos) >= 1) or np.any(np.abs(rot-prev_rot) >= 1):
                print(np.abs(scale-prev_scale), prev_scale*0.25)
                R1 = R.from_quat(prev_rot)
                R2 = R.from_quat(rot)
                deltaR = R2 * (R1.inv())
                ratio_scale = scale/prev_scale
                delta_pos = pos-deltaR.apply(prev_pos)*ratio_scale # x2 = ratio_scale*deltaR*x1 + delta_pos
                root_pos_stream[i:] -= delta_pos
                root_pos_stream[i:] *= 1.0/ratio_scale
                root_pos_stream[i:] = deltaR.inv().apply(root_pos_stream[i:])
                root_rot_stream[i:] = (deltaR.inv() * R.from_quat(root_rot_stream[i:])).as_quat()
                root_scale_stream[i:] *= 1.0/ratio_scale

            prev_scale = scale
            prev_pos = pos
            prev_rot = rot

    def fix_rot_discontinuities(self):
        body_node_list=self.data["body_node_list"]
        body_node_data=self.data["body_node_data"]
        assert self.data["rot_param"] in ["quat", "axis_angle"]
        use_axis_angle = self.data["rot_param"] == "axis_angle"
        # assert not self.data["is_relative"]
        # print(body_node_data)
        for key in body_node_list:
            bodyNode = body_node_data[str(key)]
            if bodyNode["rot_exists"]:
                rot_stream = bodyNode["rot_stream"]
                prev_rot = None
                for i, rot in enumerate(rot_stream):
                    if prev_rot is None:
                        prev_rot = rot
                    # if np.any(np.abs(rot-prev_rot) >= 1e-1) and np.abs(np.linalg.norm(rot)-np.linalg.norm(prev_rot)) <= 1e-1:
                    # if np.any(np.abs(rot-prev_rot) >= 1e-1):
                    if use_axis_angle:
                        # if np.any(np.abs(rot-prev_rot) >= 1e-1) and np.abs(np.linalg.norm(rot)-np.linalg.norm(prev_rot)) <= 1e-1:
                        norm = np.linalg.norm(rot,axis=-1, keepdims=True)
                        # norm2 = np.linalg.norm(prev_rot,axis=-1, keepdims=True)
                        rot2 = -(2*np.pi-norm)*rot/norm
                        # norm3 = (2*np.pi-norm2)
                        # if np.any(np.abs(rot-prev_rot)>=3e-1) and np.abs(norm - norm3) <=3e-1:
                        if np.linalg.norm(prev_rot-rot)>=np.linalg.norm(prev_rot-rot2):
                            norms = np.linalg.norm(rot_stream[i:],axis=-1, keepdims=True)
                            rot_stream[i:] = -(2*np.pi-norms)*rot_stream[i:]/norms
                    else:
                        if np.all(np.abs(prev_rot+rot)<=1e-1):
                            rot_stream[i:] = -rot_stream[i:]
                    prev_rot = rot_stream[i]

        if self.data["hands_are_tracked"]:
            rot_stream = self.data["left_hand_rots"]
            num_frames = rot_stream.shape[0]
            rot_stream = np.reshape(rot_stream, (num_frames*23,-1))
            prev_rot = None
            for i, rot in enumerate(rot_stream):
                if prev_rot is None:
                    prev_rot = rot
                if np.any(np.abs(rot-prev_rot) >= 1e-1):
                    if use_axis_angle:
                        norms = np.linalg.norm(rot_stream[i:],axis=-1, keepdims=True)
                        rot_stream[i:] = -(2*np.pi-norms)*rot_stream[i:]/norms
                    else:
                        rot_stream[i:] = -rot_stream[i:]
                prev_rot = rot
            self.data["left_hand_rots"] = rot_stream.reshape(num_frames,-1)

            rot_stream = self.data["right_hand_rots"]
            num_frames = rot_stream.shape[0]
            rot_stream = np.reshape(rot_stream, (num_frames*23,-1))
            prev_rot = None
            for i, rot in enumerate(rot_stream):
                if prev_rot is None:
                    prev_rot = rot
                if np.any(np.abs(rot-prev_rot) >= 1e-1):
                    if use_axis_angle:
                        norms = np.linalg.norm(rot_stream[i:],axis=-1, keepdims=True)
                        rot_stream[i:] = -(2*np.pi-norms)*rot_stream[i:]/norms
                    else:
                        rot_stream[i:] = -rot_stream[i:]
                prev_rot = rot
            self.data["right_hand_rots"] = rot_stream.reshape(num_frames,-1)

    def convert_quaternions_to_axis_angles(self, only_last_frame=False):
        body_node_list=self.data["body_node_list"]
        body_node_data=self.data["body_node_data"]
        # assert not self.data["is_relative"]
        assert self.data["rot_param"] == "quat"
        self.data["rot_param"] = "axis_angle"
        # print(body_node_data)
        for key in body_node_list:
            bodyNode = body_node_data[str(key)]
            if bodyNode["rot_exists"]:
                rot_stream = bodyNode["rot_stream"]
                if only_last_frame:
                    if rot_stream.shape[0] == 1:
                        bodyNode["rot_stream"] = convert_quaternions_to_axis_angles(rot_stream)
                    else:
                        bodyNode["rot_stream"][-1] = convert_quaternions_to_axis_angles(rot_stream[-1])
                else:
                    bodyNode["rot_stream"] = convert_quaternions_to_axis_angles(rot_stream)
        if self.data["hands_are_tracked"]:
            rot_stream = self.data["left_hand_rots"]
            num_frames = rot_stream.shape[0]
            rot_stream = np.reshape(rot_stream, (num_frames*23,-1))
            if only_last_frame:
                if num_frames == 1:
                    rot_stream = convert_quaternions_to_axis_angles(rot_stream)
                else:
                    rot_stream[-23] = convert_quaternions_to_axis_angles(rot_stream[-23])
            else:
                rot_stream = convert_quaternions_to_axis_angles(rot_stream)
            self.data["left_hand_rots"] = rot_stream.reshape(num_frames,-1)

            rot_stream = self.data["right_hand_rots"]
            num_frames = rot_stream.shape[0]
            rot_stream = np.reshape(rot_stream, (num_frames*23,-1))
            if only_last_frame:
                if num_frames == 1:
                    rot_stream = convert_quaternions_to_axis_angles(rot_stream)
                else:
                    rot_stream[-23] = convert_quaternions_to_axis_angles(rot_stream[-23])
            else:
                rot_stream = convert_quaternions_to_axis_angles(rot_stream)
            self.data["right_hand_rots"] = rot_stream.reshape(num_frames,-1)

    def convert_axis_angles_to_quaternions(self, only_last_frame=False):
        body_node_list=self.data["body_node_list"]
        body_node_data=self.data["body_node_data"]
        assert self.data["rot_param"] == "axis_angle"
        self.data["rot_param"] = "quat"
        # print(body_node_data)
        for key in body_node_list:
            bodyNode = body_node_data[str(key)]
            if bodyNode["rot_exists"]:
                rot_stream = bodyNode["rot_stream"]
                if only_last_frame:
                    if rot_stream.shape[0] == 1:
                        bodyNode["rot_stream"] = convert_axis_angles_to_quaternions(rot_stream)
                    else:
                        bodyNode["rot_stream"][-1] = convert_axis_angles_to_quaternions(rot_stream[-1])
                else:
                    bodyNode["rot_stream"] = convert_axis_angles_to_quaternions(rot_stream)

        if self.data["hands_are_tracked"]:
            rot_stream = self.data["left_hand_rots"]
            num_frames = rot_stream.shape[0]
            rot_stream = np.reshape(rot_stream, (num_frames*23,-1))
            if only_last_frame:
                if num_frames == 1:
                    rot_stream = convert_axis_angles_to_quaternions(rot_stream)
                else:
                    rot_stream[-23] = convert_axis_angles_to_quaternions(rot_stream[-23])
            else:
                rot_stream = convert_axis_angles_to_quaternions(rot_stream)
            self.data["left_hand_rots"] = rot_stream.reshape(num_frames,-1)

            rot_stream = self.data["right_hand_rots"]
            num_frames = rot_stream.shape[0]
            rot_stream = np.reshape(rot_stream, (num_frames*23,-1))
            if only_last_frame:
                if num_frames == 1:
                    rot_stream = convert_axis_angles_to_quaternions(rot_stream)
                else:
                    rot_stream[-23] = convert_axis_angles_to_quaternions(rot_stream[-23])
            else:
                rot_stream = convert_axis_angles_to_quaternions(rot_stream)
            self.data["right_hand_rots"] = rot_stream.reshape(num_frames,-1)

    def make_relative(self):
        body_node_list=self.data["body_node_list"].copy()
        body_node_data=self.data["body_node_data"]
        assert not self.data["is_relative"]
        self.data["is_relative"] = True
        assert "10" in body_node_list
        idx = body_node_list.index("10")
        if idx != 0:
            other = body_node_list[0]
            body_node_list[0] = "10"
            body_node_list[idx] = other
        # print(body_node_data)
        rots_y_head_inv = None
        head_forwards = None
        head_rights = None
        rots_y_root_inv = None
        root_forwards = None
        root_rights = None
        pos_head = None
        for key in body_node_list:
            bodyNode = body_node_data[str(key)]
            if bodyNode["rot_exists"]:
                rot_stream = bodyNode["rot_stream"]
                if self.data["rot_param"] == "quat":
                    Rs = R.from_quat(rot_stream)
                elif self.data["rot_param"] == "axis_angle":
                    Rs = R.from_rotvec(rot_stream)
                else:
                    raise NotSupportedExcpetion()
                if str(key) == "1": # root
                    # Rs = Rs * Rs[0].inv()
                    # Rms = Rs.as_matrix()
                    vs = Rs.apply(np.array([0,0,1]).T)
                    vs[:,1] = 0
                    root_forwards = vs.copy()
                    root_rights = vs.copy()
                    root_rights[:,0] = vs[:,1]
                    root_rights[:,1] = -vs[:,0]
                    # print(vs.shape)
                    thetas = np.arctan(vs[:,0]/vs[:,2])
                    thetas += np.pi*(np.sign(vs[:,2])+1)/2
                    print(thetas)
                    Rys = R.from_euler(seq="y", angles=thetas)
                    thetas = np.concatenate([np.array([0]), thetas])
                    thetas_diff = np.diff(thetas)
                    # Rys_diff = R.from_euler(seq="y", angles=np.diff(thetas))
                    rots_y_root_inv = Rys.inv()
                    Rs = rots_y_root_inv * Rs
                    # Rs = R.from_matrix(Rms)
                    bodyNode["thetas_diff"] = np.expand_dims(thetas_diff,1)
                elif str(key) == "10": # head
                    # Rs = Rs * Rs[0].inv()
                    # Rms = Rs.as_matrix()
                    vs = Rs.apply(np.array([0,0,1]).T)
                    vs[:,1] = 0
                    head_forwards = vs.copy()
                    head_rights = vs.copy()
                    head_rights[:,0] = vs[:,1]
                    head_rights[:,1] = -vs[:,0]
                    # print(vs.shape)
                    thetas = np.arctan(vs[:,0]/vs[:,2])
                    thetas += np.pi*(np.sign(vs[:,2])+1)/2
                    print(thetas)
                    Rys = R.from_euler(seq="y", angles=thetas)
                    thetas = np.concatenate([np.array([0]), thetas])
                    thetas_diff = np.diff(thetas)
                    for diff in thetas_diff:
                        if np.abs(diff+2*np.pi) < np.abs(diff):
                            diff = diff+2*np.pi
                        elif np.abs(diff-2*np.pi) < np.abs(diff):
                            diff = diff-2*np.pi
                    # Rys_diff = R.from_euler(seq="y", angles=np.diff(thetas))
                    rots_y_head_inv = Rys.inv()
                    # Rs = R.from_matrix(Rms)
                    Rs = rots_y_head_inv * Rs
                    bodyNode["thetas_diff"] = np.expand_dims(thetas_diff,1)
                    # continue
                else:
                    Rs = rots_y_head_inv * Rs
                    # continue
                if self.data["rot_param"] == "quat":
                    rot_stream = Rs.as_quat()
                elif self.data["rot_param"] == "axis_angle":
                    rot_stream = Rs.as_rotvec()
                else:
                    raise NotSupportedExcpetion()
                bodyNode["rot_stream"] = rot_stream
        for key in body_node_list:
            bodyNode = body_node_data[str(key)]
            if bodyNode["pos_exists"]:
                pos_stream = bodyNode["pos_stream"]
                if key == "1":
                    pos_root = pos_stream.copy()
                    pos_xz = np.stack([pos_root[:,0],pos_root[:,2]],axis=1)
                    pos_xz = np.concatenate([np.zeros((1,2)), pos_xz])
                    deltas = np.diff(pos_xz,axis=0)
                    deltas_rel = np.einsum("ijk,ik->ij",np.stack([root_forwards[:,[0,2]], root_rights[:,[0,2]]],axis=1),deltas)
                    bodyNode["pos_stream_y"] = pos_stream[:,1:2]
                    # pos_stream = np.zeros_like(pos_stream)
                    # bodyNode["pos_stream"] = pos_stream
                    bodyNode["deltas_rel"] = deltas_rel
                elif key == "10":
                    pos_head = pos_stream.copy()
                    pos_xz = np.stack([pos_head[:,0],pos_head[:,2]],axis=1)
                    pos_xz = np.concatenate([np.zeros((1,2)), pos_xz])
                    deltas = np.diff(pos_xz,axis=0)
                    # print(np.stack([head_forwards[:,[0,2]], head_rights[:,[0,2]]],axis=1).shape)
                    # print(deltas.shape)
                    deltas_rel = np.einsum("ijk,ik->ij",np.stack([head_forwards[:,[0,2]], head_rights[:,[0,2]]],axis=1),deltas)
                    bodyNode["pos_stream_y"] = pos_stream[:,1:2]
                    # pos_stream[:,0] -= pos_head[:,0]
                    # pos_stream[:,2] -= pos_head[:,2]
                    # bodyNode["pos_stream"] = pos_stream
                    bodyNode["deltas_rel"] = deltas_rel
                    continue
                else:
                    pos_stream[:,0] -= pos_head[:,0]
                    pos_stream[:,2] -= pos_head[:,2]
                    pos_stream = rots_y_head_inv.apply(pos_stream)
                    # what is happening?
                    bodyNode["pos_stream"] = pos_stream
                    continue

    def make_absolute(self, only_last_frame=False):
        self.data["is_relative_to_head"] = True
        body_node_list=self.data["body_node_list"].copy()
        body_node_data=self.data["body_node_data"]
        if not only_last_frame:
            assert self.data["is_relative"]
        self.data["is_relative"] = False
        assert "10" in body_node_list
        idx = body_node_list.index("10")
        if idx != 0:
            other = body_node_list[0]
            body_node_list[0] = "10"
            body_node_list[idx] = other

        if only_last_frame:
            indices=[-1]
        else:
            indices=range(body_node_data["1"]["pos_stream"].shape[0])
        # print(body_node_data)
        rots_y_head = None
        head_forwards = None
        head_rights = None
        rots_y_root = None
        root_forwards = None
        root_rights = None
        pos_head = None
        for key in body_node_list:
            bodyNode = body_node_data[str(key)]
            if bodyNode["rot_exists"]:
                rot_stream = bodyNode["rot_stream"][indices,:]
                if self.data["rot_param"] == "quat":
                    Rs = R.from_quat(rot_stream)
                elif self.data["rot_param"] == "axis_angle":
                    Rs = R.from_rotvec(rot_stream)
                else:
                    raise NotSupportedExcpetion()
                if str(key) == "1": #root
                    thetas_diff = bodyNode["thetas_diff"][indices,:]
                    if only_last_frame:
                        if "thetas" in bodyNode:
                            thetas = bodyNode["thetas"] + thetas_diff
                        else:
                            thetas = thetas_diff
                        bodyNode["thetas"] = thetas
                    else:
                        thetas = np.cumsum(thetas_diff)
                    rots_y_root = R.from_euler(seq="y", angles=thetas)
                    # Rys_diff = R.from_euler(seq="y", angles=np.diff(thetas))
                    # Rs = R.from_matrix(Rms)
                    Rs = rots_y_root * Rs
                    vs = Rs.apply(np.array([0,0,1]).T)
                    vs[:,1] = 0
                    root_forwards = vs.copy()
                    root_rights = vs.copy()
                    root_rights[:,0] = vs[:,1]
                    root_rights[:,1] = -vs[:,0]
                    bodyNode["thetas_diff"] = None
                elif str(key) == "10": # head
                    thetas_diff = bodyNode["thetas_diff"][indices,:]
                    if only_last_frame:
                        if "thetas" in bodyNode:
                            thetas = bodyNode["thetas"] + thetas_diff
                        else:
                            thetas = thetas_diff
                        bodyNode["thetas"] = thetas
                    else:
                        thetas = np.cumsum(thetas_diff)
                    rots_y_head = R.from_euler(seq="y", angles=thetas)
                    # Rys_diff = R.from_euler(seq="y", angles=np.diff(thetas))
                    # Rs = R.from_matrix(Rms)
                    Rs = rots_y_head * Rs
                    vs = Rs.apply(np.array([0,0,1]).T)
                    vs[:,1] = 0
                    head_forwards = vs.copy()
                    head_rights = vs.copy()
                    head_rights[:,0] = vs[:,1]
                    head_rights[:,1] = -vs[:,0]
                    bodyNode["thetas_diff"] = None
                    # continue
                else:
                    Rs = rots_y_head * Rs
                    # continue
                if self.data["rot_param"] == "quat":
                    rot_stream = Rs.as_quat()
                elif self.data["rot_param"] == "axis_angle":
                    rot_stream = Rs.as_rotvec()
                else:
                    raise NotSupportedExcpetion()
                bodyNode["rot_stream"][indices,:] = rot_stream
        for key in ["10","1","17","46"]:
            bodyNode = body_node_data[str(key)]
            if bodyNode["pos_exists"]:
                if key == "10":
                    pos_stream = bodyNode["pos_stream_y"][indices,:]
                    deltas_rel = bodyNode["deltas_rel"][indices,:]
                    # print(np.stack([head_forwards[:,[0,2]], head_rights[:,[0,2]]],axis=1).transpose(1,2).shape)
                    # print(deltas_rel.shape)
                    deltas = np.einsum("ijk,ik->ij",np.stack([head_forwards[:,[0,2]], head_rights[:,[0,2]]],axis=1).transpose(0,2,1),deltas_rel)
                    if only_last_frame:
                        if "pos_xz" in bodyNode:
                            pos_xz = bodyNode["pos_xz"] + deltas
                        else:
                            pos_xz = deltas
                        bodyNode["pos_xz"] = pos_xz
                    else:
                        pos_xz = np.cumsum(deltas, axis=0)
                    pos_head = np.stack([pos_xz[:,0],pos_stream[:,0],pos_xz[:,1]],axis=1)
                    bodyNode["pos_stream"][indices,:] = pos_head
                    bodyNode["pos_stream_y"] = None
                elif key == "1":
                    pos_stream = bodyNode["pos_stream_y"][indices,:]
                    deltas_rel = bodyNode["deltas_rel"][indices,:]
                    deltas = np.einsum("ijk,ik->ij",np.stack([root_forwards[:,[0,2]], root_rights[:,[0,2]]],axis=1).transpose(0,2,1),deltas_rel)
                    if only_last_frame:
                        if "pos_xz" in bodyNode:
                            pos_xz = bodyNode["pos_xz"] + deltas
                        else:
                            pos_xz = deltas
                        bodyNode["pos_xz"] = pos_xz
                    else:
                        pos_xz = np.cumsum(deltas, axis=0)
                    pos_root = np.stack([pos_xz[:,0],pos_stream[:,0],pos_xz[:,1]],axis=1)
                    bodyNode["pos_stream"][indices,:] = pos_root
                    bodyNode["pos_stream_y"] = None
                else:
                    pos_stream = bodyNode["pos_stream"][indices,:]
                    pos_stream = rots_y_head.apply(pos_stream)
                    pos_stream[:,0] += pos_head[:,0]
                    pos_stream[:,2] += pos_head[:,2]
                    # what is happening?
                    bodyNode["pos_stream"][indices,:] = pos_stream
                    continue


def get_features(fileName, save_folder, suffixes=""):
    seq_id = "_".join(fileName.split("/"))
    npd = NeosPoseData(fileName)
    npd.load_heading_json("data/basic_config.json")
    print("Fixing root big rescaling")
    npd.fix_root_big_rescalings()
    npd.make_relative()
    # npd.make_absolute()
    # npd.fix_rot_discontinuities()
    npd.convert_quaternions_to_axis_angles()
    # print("Fixing rot discontinuities")
    npd.fix_rot_discontinuities()
    a = npd.concat_numpy()
    if type(suffixes) == type(""): suffixes = [suffixes]
    for suffix in suffixes:
        np.save(save_folder+"/"+seq_id+suffix, a)

if __name__ == '__main__':

    root_folder = "data/kulzaworld_guille_neosdata"
    # root_folder = "data/kulzaworld_guille_neosdata_smol"
    # save_folder = "data/kulzaworld_guille_neosdata_npy"
    # save_folder = "data/kulzaworld_guille_neosdata_npy_relative"
    save_folder = "data/kulzaworld_guille_neosdata_npy_testing"
    # save_folder = "data/kulzaworld_guille_neosdata_smol_npy_axis_angle"
    # root_folder = "data/U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40/S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48"
    # save_folder = "data/dekaworld_alex_guille_neosdata3"
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    for dirpath, dirs, files in os.walk(root_folder):
        for filename in files:
            fname = os.path.join(dirpath,filename)
            if fname.endswith('.dat') and not fname.endswith('mouth_streams.dat'):
                print(fname)
                # get_features(fname, save_folder=save_folder, suffixes=[".person1", ".person2"])
                get_features(fname, save_folder=save_folder, suffixes=[".person1.rel", ".person2.rel"])
    # npd = NeosPoseData("data/example/1/ID2C00_streams.dat")

    # npd.load_data()
    #
    # npd.data["hands_are_tracked"] = False
    # a = npd.concat_numpy()
    # np.save("data/example_numpy_frames", a)
    #
    # npd.clear_trajectories()
    # npd.save_json("data/basic_config.json")
