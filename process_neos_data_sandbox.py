

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
        return None
    bs = bytes(bs)
    # bytes = b''.join(bytes)
    x, = struct.unpack('f',bs)
    return x

def read_int(data_iter):
    bs = [next(data_iter,None) for _ in range(4)]
    if np.any(np.array(bs) == None):
        return None
    bs = bytes(bs)
    # bytes = b''.join(bytes)
    x, = struct.unpack('i',bs)
    return x

def read_bool(data_iter):
    bs = [next(data_iter,None) for _ in range(1)]
    if np.any(np.array(bs) == None):
        return None
    bs = bytes(bs)
    # bytes = b''.join(bytes)
    x, = struct.unpack('?',bs)
    return x

#%%
data_iter = iter(fileContent)

######HEADING
# READ absolute time
absolute_time = read_float(data_iter)

# READ version number
version_number = read_int(data_iter)
print("VERSION NUMBER", version_number)

num_body_nodes = version_number
if version_number >= 1000:
    #READ relative avatar scale
    rel_avatar_scale_x = read_float(data_iter)
    rel_avatar_scale_y = read_float(data_iter)
    rel_avatar_scale_z = read_float(data_iter)
    #READ number of body nodes
    num_body_nodes = read_int(data_iter)


body_node_data = {}
for i in range(num_body_nodes):
    #READ body node type
    nodeInt = read_int(data_iter)
    if version_number == 1000:
        nodeInt = oldToNewNodeInts[nodeInt]
    body_node_data[nodeInt] = {}
    print(nodeInt)
    #READ if scale stream exists
    scale_exists = read_bool(data_iter)
    body_node_data[nodeInt]["scale_exists"] = scale_exists
    if scale_exists:
        body_node_data[nodeInt]["scale_stream"] = []
    #READ if position stream exists
    pos_exists = read_bool(data_iter)
    body_node_data[nodeInt]["pos_exists"] = pos_exists
    if pos_exists:
        body_node_data[nodeInt]["pos_stream"] = []
    #READ if rotation stream exists
    rot_exists = read_bool(data_iter)
    body_node_data[nodeInt]["rot_exists"] = rot_exists
    if rot_exists:
        body_node_data[nodeInt]["rot_stream"] = []

#READ whether hands are tracked
hands_are_tracked = read_bool(data_iter)
#READ whether metacarpals are tracked
metacarpals_are_tracked = read_bool(data_iter)

#######STREAM

while True:
    #READ deltaT
    deltaT = read_float(data_iter)
    if deltaT is None:
        break
    #READ bodyNode transforms
    for key, bodyNode in body_node_data.items():
        if bodyNode["scale_exists"]:
            scalex = read_float(data_iter)
            scaley = read_float(data_iter)
            scalez = read_float(data_iter)
            scale = [scalex,scaley,scalez]
            bodyNode["scale_stream"] += [scale]
        if bodyNode["pos_exists"]:
            posx = read_float(data_iter)
            posy = read_float(data_iter)
            posz = read_float(data_iter)
            pos = [posx,posy,posz]
            bodyNode["pos_stream"] += [pos]
        if bodyNode["rot_exists"]:
            rotx = read_float(data_iter)
            roty = read_float(data_iter)
            rotz = read_float(data_iter)
            rotw = read_float(data_iter)
            rot = [rotx,roty,rotz,rotw]
            bodyNode["rot_stream"] += [rot]
    #READ finger poses
    if hands_are_tracked:
        #Left hand
        for i in range(23):
            was_succesful = read_bool(data_iter)
            x = read_float(data_iter)
            y = read_float(data_iter)
            z = read_float(data_iter)
            w = read_float(data_iter)
        #Right hand
        for i in range(23):
            was_succesful = read_bool(data_iter)
            x = read_float(data_iter)
            y = read_float(data_iter)
            z = read_float(data_iter)
            w = read_float(data_iter)
