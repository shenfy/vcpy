import enum, io, itertools, struct
import numpy as np

@enum.unique
class VertexAttribute(enum.Enum):
  POSITION = 'position'
  NORMAL = 'normal'
  TANGENT = 'tangent'
  TEX_COORD = 'tex_coord'
  COLOR = 'color'

_VERTEX_ATTRIB_DEFAULT_TYPE = {
  VertexAttribute.POSITION: (3, np.float32),
  VertexAttribute.NORMAL: (3, np.float32),
  VertexAttribute.TANGENT: (3, np.float32),
  VertexAttribute.TEX_COORD: (2, np.float32),
  VertexAttribute.COLOR: (3, np.uint8)
}

class PolygonSoup:
  def __init__(self, num_verts, vertex_attributes):
    self.vertex_attributes = vertex_attributes
    for attrib in self.vertex_attributes:
      count, dtype = _VERTEX_ATTRIB_DEFAULT_TYPE[attrib]
      setattr(self, attrib.value, np.zeros(shape=(num_verts, count), dtype=dtype))
    self.faces = []

  def __eq__(self, other):
    if self.vertex_attributes != other.vertex_attributes:
      return False
    for attrib in self.vertex_attributes:
      self_value = getattr(self, attrib.value)
      other_value = getattr(other, attrib.value)
      if self_value.size != 0 and other_value.size != 0:
        if (self_value != other_value).any():
          return False
    return self.faces == other.faces

  def num_verts(self):
    if len(self.vertex_attributes) == 0:
      return 0
    return getattr(self, self.vertex_attributes[0].value).shape[0]

  def add_vertex(self, *args, **kwargs):
    if args and kwargs:
      raise RuntimeError('Positional arguments and keyword arguments cannot be used together')

    if args:
      if len(args) != len(self.vertex_attributes):
        raise RuntimeError('Expected {} vertex attributes, got {}'.format(
          len(self.vertex_attributes), len(args)))

      for arg, attrib in zip(args, self.vertex_attributes):
        value = getattr(self, attrib.value)
        _, dtype = _VERTEX_ATTRIB_DEFAULT_TYPE[attrib]
        setattr(self, attrib.value, np.vstack((value, np.array(arg, dtype=dtype))))
    else:
      for key, value in kwargs.items():
        if not hasattr(self, key):
          raise RuntimeError('Attribute {} not present'.format(VertexAttribute(key)))

      for attrib in self.vertex_attributes:
        count, dtype = _VERTEX_ATTRIB_DEFAULT_TYPE[attrib]
        value = getattr(self, attrib.value)
        setattr(self, attrib.value, np.vstack((value, np.zeros(shape=(1, count), dtype=dtype))))

      for key, value in kwargs.items():
        getattr(self, key)[-1, :] = value

def _parse_ply_header(file):
  line = file.readline()
  if not line:
    raise RuntimeError('Unexpected EOF')
  if line != b'ply\n':
    raise RuntimeError('Unexpected line {}'.format(line))

  result = {
    'format': '',
    'version': '',
    'elements': []
  }
  current_element = None
  while True:
    # read line
    line = file.readline()
    if not line:
      raise RuntimeError('Unexpected EOF')
    line = line.strip()
    if not line:
      continue

    # get words
    words = line.split(b' ')

    # format line
    if words[0] == b'format':
      if len(words) != 3:
        raise RuntimeError('Unrecognized format {}'.format(line))
      result['format'] = words[1].decode('utf8')
      result['version'] = words[2].decode('utf8')
      continue

    # comment line
    if words[0] == b'comment':
      continue

    # end header line, also ends element declaration
    if words[0] == b'end_header':
      if current_element is not None:
        result['elements'].append(current_element)
      break

    # element line, starts a new element and ends last element
    if words[0] == b'element':
      if len(words) != 3:
        raise RuntimeError('Unexpected element {}'.format(line))

      if current_element is not None:
        result['elements'].append(current_element)

      current_element = {
        'name': words[1].decode('utf8'),
        'size': int(words[2]),
        'properties': []
      }
      continue

    # property line
    if words[0] == b'property':
      if current_element is None:
        raise RuntimeError('Property out of any element')

      if len(words) < 3:
        raise RuntimeError('Unexpected property {}'.format(line))

      if words[1] == b'list':
        # list property
        if len(words) < 5:
          raise RuntimeError('Unexpected list property {}'.format(line))
        current_element['properties'].append(
          (words[2].decode('utf8'), words[3].decode('utf8'), words[4].decode('utf8'))
        )
      else:
        # scalar property
        current_element['properties'].append(
          (words[1].decode('utf8'), words[2].decode('utf8'))
        )
      continue

    raise RuntimeError('Unexpected line {}'.format(line))

  return result

_ATTRIB_TO_PROPERTY = {
  VertexAttribute.POSITION: (('float', 'x'), ('float', 'y'), ('float', 'z')),
  VertexAttribute.NORMAL: (('float', 'nx'), ('float', 'ny'), ('float', 'nz')),
  VertexAttribute.TANGENT: (('float', 'tx'), ('float', 'ty'), ('float', 'tz')),
  VertexAttribute.TEX_COORD: (('float', 'u'), ('float', 'v')),
  VertexAttribute.COLOR: (('uint8', 'red'), ('uint8', 'green'), ('uint8', 'blue'))
}

_PROPERTY_TO_ATTRIB = {
  ('float' ,'x'): (
    VertexAttribute.POSITION,
    (('float', 'x'), ('float', 'y'), ('float', 'z'))
  ),
  ('float32', 'x'): (
    VertexAttribute.POSITION,
    (('float32', 'x'), ('float32', 'y'), ('float32', 'z'))
  ),
  ('float', 'nx'): (
    VertexAttribute.NORMAL,
    (('float', 'nx'), ('float', 'ny'), ('float', 'nz'))
  ),
  ('float32', 'nx'): (
    VertexAttribute.NORMAL,
    (('float32', 'nx'), ('float32', 'ny'), ('float32', 'nz'))
  ),
  ('float', 'tx'): (
    VertexAttribute.TANGENT,
    (('float', 'tx'), ('float', 'ty'), ('float', 'tz'))
  ),
  ('float32', 'tx'): (
    VertexAttribute.TANGENT,
    (('float32', 'tx'), ('float32', 'ty'), ('float32', 'tz'))
  ),
  ('float', 'u'): (
    VertexAttribute.TEX_COORD,
    (('float', 'u'), ('float', 'v'))
  ),
  ('float32', 'u'): (
    VertexAttribute.TEX_COORD,
    (('float32', 'u'), ('float32', 'v'))
  ),
  ('uint8', 'red'): (
    VertexAttribute.COLOR,
    (('uint8', 'red'), ('uint8', 'green'), ('uint8', 'blue'))
  )
}

def _collect_vertex_attributes(element):
  result = []
  index = 0
  while index < len(element['properties']):
    prop = element['properties'][index]
    if not prop in _PROPERTY_TO_ATTRIB:
      raise RuntimeError('Unsupported vertex property {}'.format(prop))

    attrib, needed_props = _PROPERTY_TO_ATTRIB[prop]
    result.append(attrib)
    if index + len(needed_props) > len(element['properties']):
      raise RuntimeError('Missing property of attribute {}'.format(attrib))

    for (prop, needed_prop) in zip(element['properties'][index:], needed_props):
      if prop != needed_prop:
        raise RuntimeError('Expected property {}, got {}'.format(needed_prop, prop))

    index = index + len(needed_props)

  return tuple(result)

def _validate_face_element(element):
  if len(element['properties']) != 1:
    raise RuntimeError('Unsupported number of face property: {}'.format(len(element['properties'])))

  prop = element['properties'][0]
  if len(prop) != 3:
    raise RuntimeError('Unsupported face property {}'.format(prop))

  size_type, data_type, name = prop
  if ((size_type != 'uchar' and size_type != 'uint8')
    or (data_type != 'int' and data_type != 'uint32')
    or name != 'vertex_indices'):
    raise RuntimeError('Unsupported face list property {}'.format(prop))

def _read_at_least(file, size):
  result = file.read(size)
  if len(result) != size:
    raise RuntimeError('Unexpected EOF')
  return result

def write_ply(file, soup, write_binary):
  # file format header
  file.write(b'ply\n')
  if write_binary:
    file.write(b'format binary_little_endian 1.0\n')
  else:
    file.write(b'format ascii 1.0\n')
  file.write(b'comment vcpy.polygonsoup generated\n')

  # vertex header
  num_verts = soup.num_verts()
  file.write('element vertex {}\n'.format(num_verts).encode('utf8'))
  for attrib in soup.vertex_attributes:
    for p in _ATTRIB_TO_PROPERTY[attrib]:
      file.write('property {} {}\n'.format(*p).encode('utf8'))

  # face header
  file.write('element face {}\n'.format(len(soup.faces)).encode('utf8'))
  file.write(b'property list uint8 uint32 vertex_indices\n')

  # end header
  file.write(b'end_header\n')

  # vertex data
  values = [getattr(soup, attrib.value) for attrib in soup.vertex_attributes]
  # ensure data shapes and types are correct
  for attrib, value in zip(soup.vertex_attributes, values):
    count, dtype = _VERTEX_ATTRIB_DEFAULT_TYPE[attrib]
    if value.shape != (soup.num_verts(), count) or value.dtype != dtype:
      raise RuntimeError('Invalid shape {}/dtype {} for vertex attribute {}'.format(
        value.shape, value.dtype, attrib))

  if write_binary:
    for i in range(num_verts):
      for attrib, value in zip(soup.vertex_attributes, values):
        file.write(value[i, :].tobytes())
  else:
    for i in range(num_verts):
      for attrib, value in zip(soup.vertex_attributes, values):
        value = value[i, :]
        file.write(('{} ' * value.size).format(*value).encode('utf8'))
      file.write(b'\n')

  # face data
  if write_binary:
    for face in soup.faces:
      file.write(np.array(len(face), dtype=np.uint8).tobytes())
      file.write(np.array(face, dtype=np.uint32).tobytes())
  else:
    for face in soup.faces:
      file.write('{} '.format(len(face)).encode('utf8'))
      for v in face:
        file.write('{} '.format(v).encode('utf8'))
      file.write(b'\n')

def load_ply(file):
  header = _parse_ply_header(file)
  if header['format'] == 'binary_little_endian':
    is_binary = True
  elif header['format'] == 'ascii':
    is_binary = False
  else:
    raise RuntimeError('Unsupported PLY format {}'.format(header['format']))

  # decide meta data
  num_verts = 0
  num_faces = 0
  vertex_attributes = ()
  for element in header['elements']:
    if element['name'] == 'vertex':
      num_verts = element['size']
      vertex_attributes = _collect_vertex_attributes(element)
    elif element['name'] == 'face':
      num_faces = element['size']
      _validate_face_element(element)
    else:
      raise RuntimeError('Unsupported element {}'.format(element['name']))

  # read data
  result = PolygonSoup(num_verts, vertex_attributes)
  values = [getattr(result, attrib.value) for attrib in vertex_attributes]
  for element in header['elements']:
    if element['name'] == 'vertex':
      if is_binary:
        for i in range(num_verts):
          for attrib, value in zip(vertex_attributes, values):
            count, dtype = _VERTEX_ATTRIB_DEFAULT_TYPE[attrib]
            buf = _read_at_least(file, count * np.dtype(dtype).itemsize)
            value[i, :] = np.frombuffer(buf, dtype=dtype, count=count)
      else:
        # parse line by line
        for i in range(num_verts):
          # mask sure we consume the whole line
          line = file.readline()
          words = line.strip().split(b' ')
          index = 0
          for attrib, value in zip(vertex_attributes, values):
            count, dtype = _VERTEX_ATTRIB_DEFAULT_TYPE[attrib]
            if len(words) < index + count:
              raise RuntimeError('Invalid vertex {}'.format(line))
            value[i, :] = np.array([dtype(v) for v in words[index:index+count]], dtype=dtype)
            index += count
          if index != len(words):
            raise RuntimeError('Invalid vertex {}'.format(line))
    elif element['name'] == 'face':
      if is_binary:
        for _ in range(num_faces):
          num_vertex = _read_at_least(file, np.dtype(np.uint8).itemsize)
          num_vertex, = struct.unpack('B', num_vertex)

          face = _read_at_least(file, num_vertex * np.dtype(np.uint32).itemsize)
          face = [int(v) for v in np.frombuffer(face, dtype=np.uint32, count=num_vertex)]
          result.faces.append(face)
      else:
        for iface in range(num_faces):
          # make sure we consume the whole line
          line = file.readline()
          words = [int(v) for v in line.strip().split(b' ')]
          if len(words) != words[0] + 1:
            raise RuntimeError('Invalid face {}'.format(line))
          result.faces.append(words[1:])

  return result
