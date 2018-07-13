import numpy as np

def load_ply(filename):
    with open(filename) as f:
        for line in f:
            if line.startswith('element vertex'):
                num_vertices = int(line[14:])
            elif line.startswith('element face'):
                num_faces = int(line[12:])
            elif line.startswith('end_header'):
                vertices = np.zeros((num_vertices, 3), dtype=float)
                triangles = np.zeros((num_faces, 3), dtype=int)
                break
        for i in range(num_vertices):
            vertices[i, :] = [float(num) for num in next(f).split()][:3]
        for i in range(num_faces):
            nums = [int(num) for num in next(f).split()]
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
    outfile.write('element face %d\n' % (simplices.shape[0]))
    outfile.write('property list uchar int vertex_indices\n')
    outfile.write('end_header\n')

    for vertex in pts:
        outfile.write('%f %f %f\n' % (vertex[0], vertex[1], vertex[2]))

    for face in simplices:
        outfile.write('3 %d %d %d\n' % (face[0], face[1], face[2]))

    outfile.close()
