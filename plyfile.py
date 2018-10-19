import numpy as np

def load_ply(filename):
    with open(filename, 'rb') as f:
        binary = True

        while True:
            line = f.readline()
            if line.startswith(b'element vertex'):
                num_vertices = int(line[14:])
            elif line.startswith(b'element face'):
                num_faces = int(line[12:])
            elif line.startswith(b'format'):
                binary = not line.startswith(b'format ascii')
            elif line.startswith(b'end_header'):
                break

        vertices = np.zeros((num_vertices, 3), dtype=np.float32)
        triangles = np.zeros((num_faces, 3), dtype=np.int32)
        for i in range(num_vertices):
            if binary:
                vertices[i, :] = np.fromfile(f, np.float32, 3)
            else:
                vertices[i, :] = [np.float32(num) for num in f.readline().split()[:3]]
        for i in range(num_faces):
            if binary:
                num = np.fromfile(f, np.uint8, 1)
                assert num[0] == 3
                triangles[i, :] = np.fromfile(f, np.int32, 3)
            else:
                nums = [np.int32(num) for num in f.readline().split()]
                assert nums[0] == 3
                triangles[i, :] = nums[1:4]
    return vertices, triangles

def write_ply(pts, simplices, filename):
    outfile = open(filename, 'w')
    outfile.write('ply\nformat ascii 1.0\n')
    outfile.write('element vertex %d\n' % (pts.shape[0]))
    outfile.write('property float x\n')
    outfile.write('property float y\n')
    outfile.write('property float z\n')
    if simplices is not None:
        outfile.write('element face %d\n' % (simplices.shape[0]))
        outfile.write('property list uchar int vertex_indices\n')
    outfile.write('end_header\n')

    for vertex in pts:
        outfile.write('%f %f %f\n' % (vertex[0], vertex[1], vertex[2]))

    if simplices is not None:
        for face in simplices:
            outfile.write('3 %d %d %d\n' % (face[0], face[1], face[2]))

    outfile.close()
