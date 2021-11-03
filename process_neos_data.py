

fileName = "data/example/1/ID2B00_streams.dat"
file = open(fileName, mode='rb')
fileContent = file.read()

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

def read_int(data_iter):
    bs = [next(data_iter,None) for _ in range(4)]
    if np.any(np.array(bs) == None):
        return None,None
    bs = bytes(bs)
    # bytes = b''.join(bytes)
    x, = struct.unpack('i',bs)
    return x,bs

def read_bool(data_iter):
    bs = [next(data_iter,None) for _ in range(1)]
    if np.any(np.array(bs) == None):
        return None,None
    bs = bytes(bs)
    # bytes = b''.join(bytes)
    x, = struct.unpack('?',bs)
    return x,bs

#%%
data_iter = iter(fileContent)

######HEADING
def get_heading_bytes():
    # READ absolute time
    bs = bytes()
    absolute_time,b = read_float(data_iter)
    bs+=b

    # READ version number
    version_number,b = read_int(data_iter)
    bs+=b
    print("VERSION NUMBER", version_number)

    num_body_nodes = version_number
    if version_number >= 1000:
        #READ relative avatar scale
        rel_avatar_scale_x,b = read_float(data_iter)
        bs+=b
        rel_avatar_scale_y,b = read_float(data_iter)
        bs+=b
        rel_avatar_scale_z,b = read_float(data_iter)
        bs+=b
        #READ number of body nodes
        num_body_nodes,b = read_int(data_iter)


    body_node_data = {}
    for i in range(num_body_nodes):
        #READ body node type
        nodeInt,bs = read_int(data_iter)
        bs+=b
        if version_number == 1000:
            nodeInt = oldToNewNodeInts[nodeInt]
        body_node_data[nodeInt] = {}
        print(nodeInt)
        #READ if scale stream exists
        scale_exists,b = read_bool(data_iter)
        bs+=b
        body_node_data[nodeInt]["scale_exists"] = scale_exists
        if scale_exists:
            body_node_data[nodeInt]["scale_stream"] = []
        #READ if position stream exists
        pos_exists,b = read_bool(data_iter)
        bs+=b
        body_node_data[nodeInt]["pos_exists"] = pos_exists
        if pos_exists:
            body_node_data[nodeInt]["pos_stream"] = []
        #READ if rotation stream exists
        rot_exists,b = read_bool(data_iter)
        bs+=b
        body_node_data[nodeInt]["rot_exists"] = rot_exists
        if rot_exists:
            body_node_data[nodeInt]["rot_stream"] = []

    #READ whether hands are tracked
    hands_are_tracked,b = read_bool(data_iter)
    bs+=b
    #READ whether metacarpals are tracked
    metacarpals_are_tracked,b = read_bool(data_iter)
    bs+=b
    return bs

#######STREAM

# while True:
def get_frame_bytes():
    #READ deltaT
    bs=bytes()
    deltaT,b = read_float(data_iter)
    if deltaT is None:
        # break
        return None
    bs+=b
    #READ bodyNode transforms
    for key, bodyNode in body_node_data.items():
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
    if hands_are_tracked:
        #Left hand
        for i in range(23):
            was_succesful,b = read_bool(data_iter)
            bs+=b
            x,b = read_float(data_iter)
            bs+=b
            y,b = read_float(data_iter)
            bs+=b
            z,b = read_float(data_iter)
            bs+=b
            w,b = read_float(data_iter)
            bs+=b
        #Right hand
        for i in range(23):
            was_succesful,b = read_bool(data_iter)
            bs+=b
            x,b = read_float(data_iter)
            bs+=b
            y,b = read_float(data_iter)
            bs+=b
            z,b = read_float(data_iter)
            bs+=b
            w,b = read_float(data_iter)
            bs+=b
            
    return bs
