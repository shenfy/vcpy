import unittest, io
import numpy as np
from polygonsoup import VertexAttribute, PolygonSoup, write_ply, load_ply

class TestPolygonSoup(unittest.TestCase):
  def test_add_vertex(self):
    # test default value
    soup = PolygonSoup(0, (VertexAttribute.POSITION, VertexAttribute.COLOR))
    self.assertEqual(soup.num_verts(), 0)
    soup.add_vertex()
    self.assertEqual(soup.num_verts(), 1)
    self.assertTrue((soup.position == np.array([[0, 0, 0]], dtype=np.float32)).all())
    self.assertTrue((soup.color == np.array([[0, 0, 0,]], dtype=np.uint8)).all())
    # test positional arguments
    soup = PolygonSoup(0, (VertexAttribute.POSITION, VertexAttribute.COLOR))
    self.assertEqual(soup.num_verts(), 0)
    soup.add_vertex([1, 2, 3], [255, 0, 0])
    self.assertEqual(soup.num_verts(), 1)
    self.assertTrue((soup.position == np.array([[1, 2, 3]], dtype=np.float32)).all())
    self.assertTrue((soup.color == np.array([[255, 0, 0,]], dtype=np.uint8)).all())
    # test keyword arguments
    soup = PolygonSoup(0, (VertexAttribute.POSITION, VertexAttribute.COLOR))
    self.assertEqual(soup.num_verts(), 0)
    soup.add_vertex(position=[1, 2, 3])
    self.assertEqual(soup.num_verts(), 1)
    self.assertTrue((soup.position == np.array([[1, 2, 3]], dtype=np.float32)).all())
    self.assertTrue((soup.color == np.array([[0, 0, 0,]], dtype=np.uint8)).all())
    # test empty soup
    soup = PolygonSoup(0, ())
    self.assertEqual(soup.num_verts(), 0)
    soup.add_vertex()
    self.assertEqual(soup.num_verts(), 0)
    # test positional arguments + keyword argument
    with self.assertRaises(RuntimeError):
      soup.add_vertex([1, 2, 3], [255, 0, 0], position=[1, 2, 3])
    # test incorrect number of positional arguments
    with self.assertRaises(RuntimeError):
      soup.add_vertex([1, 2, 3])
    # test invalid keyword argument
    with self.assertRaises(RuntimeError):
      soup.add_vertex(normal=[0, 0, 1])

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
    soup = PolygonSoup(0, (VertexAttribute.POSITION, VertexAttribute.NORMAL,
      VertexAttribute.TEX_COORD, VertexAttribute.COLOR))
    soup.add_vertex([-1.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0], [255, 0, 0])
    soup.add_vertex([1.0, -1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0], [0, 255, 0])
    soup.add_vertex([1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0], [0, 0, 255])
    soup.add_vertex([-1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0], [255, 255, 255])
    soup.faces = [[0, 1, 2], [0, 2, 3]]
    self._test_ply_serialization(soup)
    # empty soup
    soup = PolygonSoup(0, ())
    self._test_ply_serialization(soup)
    # soup with wrong column
    soup = PolygonSoup(1, (VertexAttribute.POSITION,))
    soup.position = np.array([[1, 2]], dtype=np.float32)
    with self.assertRaises(RuntimeError):
      file = io.BytesIO()
      write_ply(file, soup, False)
    # soup with mismatched row
    soup = PolygonSoup(1, (VertexAttribute.POSITION, VertexAttribute.NORMAL))
    soup.position = np.array([[1, 2, 3]], dtype=np.float32)
    soup.normal = np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float32)
    with self.assertRaises(RuntimeError):
      file = io.BytesIO()
      write_ply(file, soup, False)
    # soup with wrong dtype
    soup = PolygonSoup(1, (VertexAttribute.POSITION,))
    soup.position = np.array([[1, 2, 3]], dtype=np.float64)
    with self.assertRaises(RuntimeError):
      file = io.BytesIO()
      write_ply(file, soup, False)

if __name__ == '__main__':
  unittest.main()
