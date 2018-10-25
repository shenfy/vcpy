import numpy as np
import struct

def load_ply(filename):
    with open(filename, 'rb') as f:
        is_binary = True
        endianness = '<'

        while True:
            line = f.readline()
            if line.startswith(b'element vertex'):
                num_vertices = int(line[14:])
            elif line.startswith(b'element face'):
                num_faces = int(line[12:])
            elif line.startswith(b'format'):
                is_binary = not line.startswith(b'format ascii')
                if is_binary:
                    endianness = '<' if b'little_endian' in line else '>'
            elif line.startswith(b'end_header'):
                break

        if is_binary:
            vertices = np.fromfile(f, np.float32, 3 * num_vertices)
            vertices = vertices.reshape(-1, 3)
        else:
            vertices = np.zeros((num_vertices, 3), dtype=np.float32)
            for i in range(num_vertices):
                vertices[i, :] = [np.float32(num) for num in f.readline().split()[:3]]

        if is_binary:
            basic_format = 'B3i'
            s = struct.Struct(endianness + basic_format * num_faces)
            triangles = np.array(s.unpack(f.read(s.size)), dtype=np.int32)
            triangles = triangles.reshape(-1, 4)
            assert np.all(triangles[:, 0] == 3)
            triangles = triangles[:, 1:]
        else:
            triangles = np.zeros((num_faces, 3), dtype=np.int32)
            for i in range(num_faces):
                nums = [np.int32(num) for num in f.readline().split()]
                assert nums[0] == 3
                triangles[i, :] = nums[1:4]
    return vertices, triangles

def write_ply(pts, simplices, filename, write_binary=True):
    outfile = open(filename, 'wb')
    outfile.write(b'ply\n')
    if write_binary:
        outfile.write(b'format binary_little_endian 1.0\n')
    else:
        outfile.write(b'format ascii 1.0\n')
    outfile.write(b'element vertex %d\n' % (pts.shape[0]))
    outfile.write(b'property float x\n')
    outfile.write(b'property float y\n')
    outfile.write(b'property float z\n')
    if simplices is not None:
        outfile.write(b'element face %d\n' % (simplices.shape[0]))
        outfile.write(b'property list uchar int vertex_indices\n')
    outfile.write(b'end_header\n')

    if write_binary:
        pts.astype(np.float32).tofile(outfile)
    else:
        for vertex in pts:
            outfile.write(b'%f %f %f\n' % (vertex[0], vertex[1], vertex[2]))

    if simplices is not None:
        if write_binary:
            basic_format = 'B3i'
            s = struct.Struct('<' + basic_format * simplices.shape[0])
            simplices = np.hstack([3 * np.ones((simplices.shape[0], 1), dtype=np.int32), simplices])
            outfile.write(s.pack(*simplices.flatten()))
        else:
            for face in simplices:
                outfile.write(b'3 %d %d %d\n' % (face[0], face[1], face[2]))

    outfile.close()
