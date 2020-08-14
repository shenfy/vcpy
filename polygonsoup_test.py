import unittest, io
import numpy as np
from polygonsoup import VertexAttribute, Vertex, PolygonSoup, write_ply, load_ply

class TestVertex(unittest.TestCase):
  def test_constructor(self):
    # attribute list
    vertex = Vertex(VertexAttribute.POSITION, VertexAttribute.NORMAL)
    self.assertTrue(hasattr(vertex, VertexAttribute.POSITION.value))
    self.assertTrue(hasattr(vertex, VertexAttribute.NORMAL.value))
    self.assertFalse(hasattr(vertex, VertexAttribute.TANGENT.value))
    self.assertFalse(hasattr(vertex, VertexAttribute.TEX_COORD.value))
    self.assertFalse(hasattr(vertex, VertexAttribute.COLOR.value))
    # attribute init value
    vertex = Vertex(position=[1, 2, 3], color=[255, 0, 0])
    vertex_expected = Vertex(VertexAttribute.POSITION, VertexAttribute.COLOR)
    vertex_expected.position = np.array([1, 2, 3], dtype=np.float32)
    vertex_expected.color = np.array([255, 0, 0], dtype=np.uint8)
    self.assertEqual(vertex, vertex_expected)
    # positional arguments and keyword arguments used together
    vertex = Vertex(VertexAttribute.POSITION, normal=[0, 0, 1])
    vertex_expected = Vertex(VertexAttribute.POSITION, VertexAttribute.NORMAL)
    vertex_expected.normal = np.array([0, 0, 1], dtype=np.float32)
    self.assertEqual(vertex, vertex_expected)

  def test_eq(self):
    # different attributes
    v1 = Vertex(VertexAttribute.POSITION)
    v2 = Vertex(VertexAttribute.COLOR)
    self.assertNotEqual(v1, v2)
    # different shape
    v1 = Vertex(VertexAttribute.POSITION)
    v2 = Vertex(VertexAttribute.POSITION)
    v1.position = np.array(1, dtype=np.float32)
    v2.position = np.array([1, 1, 1], dtype=np.float32)
    self.assertNotEqual(v1, v2)
    # different dtype
    v1 = Vertex(VertexAttribute.POSITION)
    v2 = Vertex(VertexAttribute.POSITION)
    v1.position = np.array([1, 2, 3], dtype=np.float32)
    v2.position = np.array([1, 2, 3], dtype=np.float64)
    self.assertNotEqual(v1, v2)
    # different value
    v1 = Vertex(VertexAttribute.POSITION)
    v2 = Vertex(VertexAttribute.POSITION)
    v1.position = np.array([1, 2, 3], dtype=np.float32)
    v2.position = np.array([0, 2, 3], dtype=np.float32)
    self.assertNotEqual(v1, v2)
    # all same
    v1 = Vertex(VertexAttribute.POSITION)
    v2 = Vertex(VertexAttribute.POSITION)
    v1.position = np.array([1, 2, 3], dtype=np.float32)
    v2.position = np.array([1, 2, 3], dtype=np.float32)
    self.assertEqual(v1, v2)

  def test_get_attrib(self):
    # legal access
    vertex = Vertex(VertexAttribute.POSITION)
    position = np.array([1, 2, 3], dtype=np.float32)
    vertex.position = position
    self.assertIs(position, vertex.get_attrib(VertexAttribute.POSITION))
    # bad dtype
    vertex.position = np.array([1, 2, 3], dtype=np.int32)
    with self.assertRaises(RuntimeError):
      vertex.get_attrib(VertexAttribute.POSITION)
    # bad shape
    vertex.position = np.array([[1, 2, 3]], dtype=np.float32)
    with self.assertRaises(RuntimeError):
      vertex.get_attrib(VertexAttribute.POSITION)

  def test_set_attrib(self):
    # legal set
    vertex = Vertex(VertexAttribute.POSITION)
    vertex.set_attrib(VertexAttribute.POSITION, [1, 2, 3])
    vertex_expected = Vertex(VertexAttribute.POSITION)
    vertex_expected.position = np.array([1, 2, 3], dtype=np.float32)
    self.assertEqual(vertex, vertex_expected)
    # bad attribute
    with self.assertRaises(RuntimeError):
      vertex.set_attrib(VertexAttribute.COLOR, [255, 0, 0])
    # bad shape
    with self.assertRaises(RuntimeError):
      vertex.set_attrib(VertexAttribute.POSITION, [1, 2])

class TestPolygonSoup(unittest.TestCase):
  def test_add_vertex(self):
    soup = PolygonSoup([VertexAttribute.POSITION, VertexAttribute.COLOR])
    # test default value
    vertex = soup.add_vertex()
    self.assertEqual(vertex, Vertex(*soup.vertex_attributes))
    # test positional arguments
    vertex = soup.add_vertex([1, 2, 3], [255, 0, 0])
    vertex_expected = Vertex(position=[1, 2, 3], color=[255, 0, 0])
    self.assertEqual(vertex, vertex_expected)
    # test keyword arguments
    vertex = soup.add_vertex(position=[1, 2, 3])
    vertex_expected = Vertex(VertexAttribute.COLOR, position=[1, 2, 3])
    self.assertEqual(vertex, vertex_expected)
    # test positional arguments + keyword argument
    with self.assertRaises(RuntimeError):
      vertex = soup.add_vertex([1, 2, 3], [255, 0, 0], position=[1, 2, 3])
    # test incorrect number of positional arguments
    with self.assertRaises(RuntimeError):
      vertex = soup.add_vertex([1, 2, 3])
    # test invalid keyword argument
    with self.assertRaises(RuntimeError):
      vertex = soup.add_vertex(normal=[0, 0, 1])

  def _test_ply_serialization(self, soup):
    # ascii ply
    file = io.BytesIO()
    write_ply(file, soup, False)
    file.seek(0, io.SEEK_SET)
    soup_ascii = load_ply(file)
    self.assertFalse(file.read(1))
    self.assertEqual(soup, soup_ascii)
    # binary ply
    file = io.BytesIO()
    write_ply(file, soup, True)
    file.seek(0, io.SEEK_SET)
    soup_binary = load_ply(file)
    self.assertFalse(file.read(1))
    self.assertEqual(soup, soup_binary)

  def test_ply_serialization(self):
    # an example soup
    soup = PolygonSoup([VertexAttribute.POSITION, VertexAttribute.NORMAL,
      VertexAttribute.TEX_COORD, VertexAttribute.COLOR])
    soup.add_vertex([-1.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0], [255, 0, 0])
    soup.add_vertex([1.0, -1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0], [0, 255, 0])
    soup.add_vertex([1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0], [0, 0, 255])
    soup.add_vertex([-1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0], [255, 255, 255])
    soup.faces = [[0, 1, 2], [0, 2, 3]]
    self._test_ply_serialization(soup)
    # empty soup
    soup = PolygonSoup([])
    self._test_ply_serialization(soup)

if __name__ == '__main__':
  unittest.main()
