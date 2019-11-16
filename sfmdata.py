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
    view_point = self.pose.world_2_view(p)
    return self.intrinsics.project(view_point, distort)

# support pinhole_radial_k3, pinhole_radial_k1, partly support pinhole_radial_k1_pba
class Intrinsics:
  NoDistortion = 0
  DistortionRadial3 = 1
  DistortionRadial1 = 2
  DistortionRadial1_PBA = 6

  def __init__(self):
    self.width = 0
    self.height = 0
    self.fx = 0
    self.fy = 0
    self.cx = 0
    self.cy = 0
    self.distortion_type = Intrinsics.NoDistortion
    self.distortions = []

  @property
  def K(self):
    result = np.identity(3, dtype=float)
    result[0, 0] = self.fx
    result[1, 1] = self.fy
    result[0, 2] = self.cx
    result[1, 2] = self.cy
    return result

  def projection(self, z_near, z_far):
    cx = self.cx + 0.5
    cy = self.cy + 0.5
    left = -cx / self.fx
    right = (self.width - cx) / self.fx
    bottom = -(self.height - cy) / self.fy
    top = cy / self.fy
    result = gl_frustum(left * z_near, right * z_near, bottom * z_near, top * z_near, z_near, z_far)
    return result

  # pts: array of [3, n]
  def project(self, pts, distort):
    pts_2d = pts[:2, :] / pts[2, :]
    if distort:
      pts_2d = self.add_disto(pts_2d)
    pts_2d = self.cam_2_img(pts_2d)
    return pts_2d

  # pts: array of [2, n]
  def add_disto(self, pts):
    if self.distortion_type == Intrinsics.NoDistortion:
      return pts
    elif self.distortion_type == Intrinsics.DistortionRadial3:
      r2 = np.sum(pts * pts, 0)
      r4 = r2 * r2
      r6 = r4 * r2
      coeff = r2 * self.distortions[0] + r4 * self.distortions[1] + r6 * self.distortions[2] + 1.0
      return pts * coeff
    elif self.distortion_type == Intrinsics.DistortionRadial1:
      r2 = np.sum(pts * pts, 0)
      coeff = r2 * self.distortions[0] + 1.0
      return pts * coeff
    else:
      raise RuntimeError('Distortion type {} unsupported'.format(self.distortion_type))

  # pts: array of [2, n]
  def cam_2_img(self, p):
    return ([self.fx, self.fy] * p.T + np.array([self.cx, self.cy], dtype=p.dtype)).T

class Extrinsic:
  def __init__(self):
    self.camera_frame = np.identity(4, dtype=float)

  # pts: array of size [3, n]
  def world_2_view(self, pts):
    return self.camera_frame[:3, :3].T @ (pts.T - self.camera_frame[:3, 3]).T

# Landmark stores feature points' info
# X: 3D coords in world frame
# observations: dict{View: Observation} for each view it is observed in
class Landmark:
  def __init__(self):
    self.X = np.zeros(3, dtype=float)
    self.observations = {}

# 2D feature info
# id_feat: local feature index for a view
# x: sub-pixel 2D image coords
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
                'focal_length': (intrinsics.fx + intrinsics.fy) / 2,
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

def __parse_intrinsics(intrinsics):
  result = {}
  for intrinsics_ in intrinsics:
    key = intrinsics_['key']
    value = intrinsics_['value']
    intrin = Intrinsics()
    data = value['ptr_wrapper']['data']
    intrin.width = data['width']
    intrin.height = data['height']
    focal_length = data['focal_length']
    if isinstance(focal_length, list):
      intrin.fx = focal_length[0]
      intrin.fy = focal_length[1]
    else:
      intrin.fx = intrin.fy = focal_length
    principal_point = data['principal_point']
    intrin.cx = principal_point[0]
    intrin.cy = principal_point[1]
    if 'disto_k3' in data:
      intrin.distortion_type = Intrinsics.DistortionRadial3
      intrin.distortions = data['disto_k3']
    elif 'disto_k1' in data:
      intrin.distortion_type = Intrinsics.DistortionRadial1
      intrin.distortions = data['disto_k1']
    elif 'disto_k1_pba' in data:
      intrin.distortion_type = Intrinsics.DistortionRadial1_PBA
      intrin.distortions = data['disto_k1_pba']
    elif 'disto_t2_2' in value['ptr_wrapper']['data']:
      intrin.distortions = value['ptr_wrapper']['data']['disto_t2_2']
    result[key] = intrin
  return result

def __parse_extrinsics(extrinsics):
  result = {}
  for extrinsic in extrinsics:
    key = extrinsic['key']
    value = extrinsic['value']
    extrin = Extrinsic()
    extrin.camera_frame[:3, :3] = np.array(value['rotation']).T
    extrin.camera_frame[:3, 3] = np.array(value['center'])
    result[key] = extrin
  return result

def __parse_views(views, intrinsics, extrinsics):
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

def __parse_structure(structure, views):
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
  result.intrinsics = __parse_intrinsics(content['intrinsics'])
  result.extrinsics = __parse_extrinsics(content['extrinsics'])
  result.views = __parse_views(content['views'], result.intrinsics, result.extrinsics)
  result.structure = __parse_structure(content['structure'], result.views)

  for landmark in result.structure.values():
    for view in landmark.observations:
      view.observations[landmark] = landmark.observations[view]

  return result

def sfm_to_gl_camera_frame(camera_frame):
  result = camera_frame.copy()
  result[:, 1] = -result[:, 1]
  result[:, 2] = -result[:, 2]
  return result
