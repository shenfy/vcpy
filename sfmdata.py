import json
import numpy as np
from vcpy.m3d import gl_frustum

class View:
  def __init__(self):
    self.filename = ''
    self.width = 0
    self.height = 0
    self.id = -1
    self.intrinsics = None
    self.pose = None
    self.observations = {}

  def project(self, p, distort=True):
    if self.intrinsics is None or self.pose is None:
      return None
    view_point = self.pose.world2view(p)
    return self.intrinsics.project(view_point, distort)

# only support pinhole_radial_k3
class Intrinsics:
  def __init__(self):
    self.width = 0
    self.height = 0
    self.f = 0
    self.cx = 0
    self.cy = 0
    self.distortions = [0, 0, 0]

  @property
  def K(self):
    result = np.identity(3, dtype=float)
    result[0, 0] = result[1, 1] = self.f
    result[0, 2] = self.cx
    result[1, 2] = self.cy
    return result

  def projection(self, z_near, z_far):
    left = -self.cx / self.f
    right = (self.width - self.cx) / self.f
    bottom = -(self.height - self.cy) / self.f
    top = self.cy / self.f
    result = gl_frustum(left * z_near, right * z_near, bottom * z_near, top * z_near, z_near, z_far)
    return result

  def project(self, p, distort):
    p = (p / p[-1])[:-1]
    if distort:
      p = self.add_disto(p)
    p = self.cam2ima(p)
    return p

  def add_disto(self, p):
    r2 =  np.dot(p, p)
    r4 = r2 * r2
    r6 = r4 * r2
    coeff = (1 + self.distortions[0] * r2 + self.distortions[1] * r4 + self.distortions[2] * r6)
    return p * coeff

  def cam2ima(self, p):
    return self.f * p + np.array([self.cx, self.cy], dtype=p.dtype)

class Extrinsic:
  def __init__(self):
    self.camera_frame = np.identity(4, dtype=float)

  def world2view(self, p):
    return self.camera_frame[:3, :3].T @ (p - self.camera_frame[:3, 3])

class Landmark:
  def __init__(self):
    self.X = np.zeros(3, dtype=float)
    self.observations = {}

class Observation:
  def __init__(self):
    self.id_feat = -1
    self.x = np.zeros(2, dtype=float)

class SfMData:
  def __init__(self):
    self.root_path = ''
    self.views = {}
    self.intrinsics = {}
    self.extrinsics = {}
    self.structure = {}

  def dump(self, f):
    raise NotImplementedError('current implementation is wrong')
    intrinsics = {value: key for key, value in self.intrinsics.items()}
    extrinsics = {value: key for key, value in self.extrinsics.items()}
    content = {
      'sfm_data_version': '0.3',
      'root_path': self.root_path,
      'views':[
        {
          'key': key,
          'value': {
            'polymorphic_id': 1073741824,
            'ptr_wrapper': {
              'id': 2147483649,
              'data': {
                'local_path': '',
                'filename': view.filename,
                'width': view.width,
                'height': view.height,
                'id_view': view.id,
                'id_intrinsic': intrinsics[view.intrinsics] if view.intrinsics else -1,
                'id_pose': extrinsics[view.pose] if view.pose else -1
              }
            }
          }
        } for key, view in self.views.items()
      ],
      'intrinsics': [
        {
          'key': key,
          'value': {
            'polymorphic_id': 2147483649,
            'polymorphic_name': 'pinhole_radial_k3',
            'ptr_wrapper': {
              'id': 2147483793,
              'data': {
                'width': intrinsics.width,
                'height': intrinsics.height,
                'focal_length': intrinsics.f,
                'principal_point': [intrinsics.cx, intrinsics.cy],
                'disto_k3': intrinsics.distortions
              }
            }
          }
        } for key, intrinsics in self.intrinsics.items()
      ],
      'extrinsics': [
        {
          'key': key,
          'value': {
            'rotation': [list(row) for row in extrinsic.camera_frame[:3, :3].T],
            'center': list(extrinsic.camera_frame[:3, 3])
          }
        } for key, extrinsic in self.extrinsics.items()
      ],
      'structure': [
        {
          'key': key,
          'value': {
            'X': list(landmark.X),
            'observations': [
              {
                'key': view.id,
                'value': {
                  'id_feat': observation.id_feat,
                  'x': list(observation.x)
                }
              } for view, observation in landmark.observations.items()
            ]
          }
        } for key, landmark in self.structure.items()
      ],
      'control_points': []
    }
    json.dump(content, f, indent=4)

def _parse_intrinsics(intrinsics):
  result = {}
  for intrinsics_ in intrinsics:
    key = intrinsics_['key']
    value = intrinsics_['value']
    intrin = Intrinsics()
    data = value['ptr_wrapper']['data']
    intrin.width = data['width']
    intrin.height = data['height']
    intrin.f = data['focal_length']
    principal_point = data['principal_point']
    intrin.cx = principal_point[0]
    intrin.cy = principal_point[1]
    intrin.distortions = data['disto_k3']
    result[key] = intrin
  return result

def _parse_extrinsics(extrinsics):
  result = {}
  for extrinsic in extrinsics:
    key = extrinsic['key']
    value = extrinsic['value']
    extrin = Extrinsic()
    extrin.camera_frame[:3, :3] = np.array(value['rotation']).T
    extrin.camera_frame[:3, 3] = np.array(value['center'])
    result[key] = extrin
  return result

def _parse_views(views, intrinsics, extrinsics):
  result = {}
  for view in views:
    key = view['key']
    value = view['value']
    v = View()
    data = value['ptr_wrapper']['data']
    v.filename = data['filename']
    v.width = data['width']
    v.height = data['height']
    v.id = data['id_view']
    assert v.id == key
    id_intrinsic = data['id_intrinsic']
    if id_intrinsic in intrinsics:
      v.intrinsics = intrinsics[id_intrinsic]
    id_pose = data['id_pose']
    if id_pose in extrinsics:
      v.pose = extrinsics[id_pose]
    result[key] = v
  return result

def _parse_structure(structure, views):
  result = {}
  for s in structure:
    key = s['key']
    value = s['value']
    landmark = Landmark()
    landmark.X = np.array(value['X'])
    for observation in value['observations']:
      view_key = observation['key']
      value = observation['value']
      ob = Observation()
      ob.id_feat = value['id_feat']
      ob.x = np.array(value['x'])
      landmark.observations[views[view_key]] = ob
    result[key] = landmark
  return result

def load_sfm_data(file):
  content = json.load(file)
  result = SfMData()
  result.root_path = content['root_path']
  result.intrinsics = _parse_intrinsics(content['intrinsics'])
  result.extrinsics = _parse_extrinsics(content['extrinsics'])
  result.views = _parse_views(content['views'], result.intrinsics, result.extrinsics)
  result.structure = _parse_structure(content['structure'], result.views)

  for landmark in result.structure.values():
    for view in landmark.observations:
      view.observations[landmark] = landmark.observations[view]

  return result

def sfm_to_gl_camera_frame(camera_frame):
  result = camera_frame.copy()
  result[:, 1] = -result[:, 1]
  result[:, 2] = -result[:, 2]
  return result
