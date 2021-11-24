import numpy as np
import json

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
        if data_iter is not None:
            self.data_iter = data_iter
            self.load_data()
        elif fileName is not None:
            self.file = open(fileName, mode='rb')
            self.fileContent = self.file.read()
            self.data_iter = iter(self.fileContent)
            self.load_data()
        self.data = {}
        self.num_frames = 0

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
        print(self.data)
        json.dump(self.data, open(fileName, "w"))

    def load_json(self, fileName):
        self.data = json.load(open(fileName, "r"))
        if "num_frames" in self.data:
            self.num_frames = self.data["num_frames"]

    def load_heading_json(self, fileName):
        data = self.data
        new_data = json.load(open(fileName, "r"))
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
        for node in data["body_node_list"]:
            d = data["body_node_data"][node]
            if d["scale_exists"]:
                arrays.append(d["scale_stream"])
            if d["pos_exists"]:
                arrays.append(d["pos_stream"])
            if d["rot_exists"]:
                arrays.append(d["rot_stream"])
        if data["hands_are_tracked"]:
            arrays.append(data["left_hand_rots"])
            arrays.append(data["right_hand_rots"])

        arrays = np.concatenate(arrays, axis=1)
        return arrays

    def append_concat_frame(self, frame, deltaT=33.33):
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
                    if len(d["pos_stream"]) == 0 or d["pos_stream"].shape[0] == 0:
                        d["pos_stream"] = frame[:,i:i+3]
                    else:
                        d["pos_stream"] = np.concatenate([d["pos_stream"],frame[:,i:i+3]])
                    i+=3
                if d["rot_exists"]:
                    if len(d["rot_stream"]) == 0 or d["rot_stream"].shape[0] == 0:
                        d["rot_stream"] = frame[:,i:i+4]
                    else:
                        d["rot_stream"] = np.concatenate([d["rot_stream"],frame[:,i:i+4]])
                    i+=4
            if data["hands_are_tracked"]:
                if len(data["left_hand_rots"]) == 0 or data["left_hand_rots"].shape[0] == 0:
                    data["left_hand_rots"] = frame[:,i:i+23*4]
                else:
                    data["left_hand_rots"] = np.concatenate([data["left_hand_rots"],frame[:,i:i+23*4]])
                i+=23*4
                if len(data["right_hand_rots"]) == 0 or data["right_hand_rots"].shape[0] == 0:
                    data["right_hand_rots"] = frame[:,i:i+23*4]
                else:
                    data["right_hand_rots"] = np.concatenate([data["right_hand_rots"],frame[:,i:i+23*4]])
                i+=23*4

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
        print(body_node_data)
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

def get_features(fileName, suffix=""):
    seq_id = "_".join(fileName.split("/"))
    npd = NeosPoseData(fileName)
    npd.load_heading_json("data/basic_config.json")
    a = npd.concat_numpy()
    np.save("data/numpys/"+seq_id+suffix, a)

if __name__ == '__main__':

    session_folder = "data/U_dekatron_R_00ee7d25_447d_4a2e_9d72_07c055ac4d40/S-d03a6c7b-1767-4582-8ffc-9277d5f5d4b5_4f45c65b-8524-4c2e-849d-e3c2cf17bd48"
    for i in range(2):
        get_features(session_folder+"/"+str(i+1)+"/"+"ID1E66900_streams.dat", suffix=".person1")
        get_features(session_folder+"/"+str(i+1)+"/"+"ID2C00_streams.dat", suffix=".person2")
        get_features(session_folder+"/"+str(i+1)+"/"+"ID1E66900_streams.dat", suffix=".person2")
        get_features(session_folder+"/"+str(i+1)+"/"+"ID2C00_streams.dat", suffix=".person1")
    # npd = NeosPoseData("data/example/1/ID2C00_streams.dat")

    # npd.load_data()
    #
    # npd.data["hands_are_tracked"] = False
    # a = npd.concat_numpy()
    # np.save("data/example_numpy_frames", a)
    #
    # npd.clear_trajectories()
    # npd.save_json("data/basic_config.json")
