import json
from pathlib import Path
import numpy as np
import bson
from vcpy.m3d import gl_frustum

class View:
  def __init__(self):
    self.filename = ''
    self.camera_name = ''
    self.width = 0
    self.height = 0
    self.id = -1
    self.intrinsics = None
    self.pose = None
    self.observations = {}
    self.valid_frame = False

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
  DistortionRadial3Brown2 = 11

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

  def frustum_vec(self, z_near, z_far):
    cx = self.cx + 0.5
    cy = self.cy + 0.5
    left = -cx / self.fx
    right = (self.width - cx) / self.fx
    bottom = -(self.height - cy) / self.fy
    top = cy / self.fy
    return np.array([left * z_near, right * z_near, bottom * z_near, top * z_near, z_near, z_far])

  def projection(self, z_near, z_far):
    result = gl_frustum(*self.frustum_vec(z_near, z_far))
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
    elif self.distortion_type == Intrinsics.DistortionRadial3Brown2:
      x_u = pts[0, :]
      y_u = pts[1, :]
      r2 = x_u ** 2 + y_u ** 2
      r4 = r2 * r2
      r6 = r4 * r2
      k1 = self.distortions[0]
      k2 = self.distortions[1]
      k3 = self.distortions[2]
      t1 = self.distortions[3]
      t2 = self.distortions[4]
      r_coeff = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
      t_x = t2 * (r2 + 2.0 * x_u * x_u) + 2.0 * t1 * x_u * y_u
      t_y = t1 * (r2 + 2.0 * y_u * y_u) + 2.0 * t2 * x_u * y_u
      return np.array([self.cx + (x_u * r_coeff + t_x) * self.fx,
        self.cy + (y_u * r_coeff + t_y) * self.fy])
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
    self.id = -1
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

  def dump_to_tag(self, f):
    content = {
      'intrinsics': [
        {
          'name': key,
          'width': intrin.width,
          'height': intrin.height,
          'fx': intrin.fx,
          'fy': intrin.fy,
          'px': intrin.cx,
          'py': intrin.cy,
          'distortion_params': distortion_from_openmvg_to_cv(intrin.distortions)
        } for key, intrin in self.intrinsics.items()
      ],
      'views': [
        {
          'id': view.id,
          'name': Path(view.filename).stem,
          'filename': view.filename,
          'camera_name': view.camera_name,
          'R': view.pose.camera_frame[:3, :3].T.tolist(),
          't': (-view.pose.camera_frame[:3, :3].T @ view.pose.camera_frame[:3, 3]).tolist(),
          'tags': [
            {
              'id': landmark.id,
              'x': obs.x[0],
              'y': obs.x[1]
            } for landmark, obs in view.observations.items()
          ],
          'valid_frame': view.valid_frame
        } for _, view in self.views.items()
      ],
      'structure': {
        'tracks': [
          {
            'tag_id': point_id,
            'type': 'TagCenterTrack',
            'world_pt': landmark.X,
            'obs': [
              {
                'image_pt': obs.x.tolist(),
                'view_id': obs.id_feat
              } for view, obs in landmark.observations.items()
            ]
          } for point_id, landmark in self.structure.items()
        ]
      }
    }

    f.write(bson.dumps(content))

  def dump_to_openmvg(self, f):
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

def distortion_from_cv_to_openmvg(distortions):
  if len(distortions) != 5:
    raise RuntimeError('Unexpected distortion_type')

  return distortions[:2] + [distortions[-1]] + distortions[2:4]

def distortion_from_openmvg_to_cv(distortions):
  if len(distortions) != 5:
    raise RuntimeError('Unexpected distortion_type')

  return distortions[:2] + distortions[3:5] + [distortions[2]]

def __from_filename_get_camera_name(filename):
  return filename[:filename.find('_')]

def __parse_tag_intrinsics(intrinsics):
  result = {}
  for intrinsics_ in intrinsics:
    camera_name = intrinsics_['name']
    intrin = Intrinsics()
    intrin.width = intrinsics_['width']
    intrin.height = intrinsics_['height']
    intrin.fx = intrinsics_['fx']
    intrin.fy = intrinsics_['fy']
    intrin.cx = intrinsics_['px']
    intrin.cy = intrinsics_['py']
    intrin.distortion_type = Intrinsics.DistortionRadial3Brown2
    intrin.distortions = distortion_from_cv_to_openmvg(intrinsics_['distortion_params'])
    result[camera_name] = intrin

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

def __parse_tag_extrinsics(views):
  result = {}
  for view in views:
    view_id = view['id']
    extrin = Extrinsic()
    extrin.camera_frame[:3, :3] = np.array(view['R']).T
    extrin.camera_frame[:3, 3] = -np.array(view['R']).T @ view['t']
    result[view_id] = extrin

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
    v.camera_name = str(id_intrinsic)
    if id_intrinsic in intrinsics:
      v.intrinsics = intrinsics[id_intrinsic]
    id_pose = data['id_pose']
    if id_pose in extrinsics:
      v.pose = extrinsics[id_pose]
    result[key] = v
  return result

def __parse_tag_views(views, intrinsics, extrinsics):
  result = {}
  for view in views:
    key = view['id']
    v = View()
    if 'filename' in view:
      v.filename = view['filename']
    else:
      # in commit 828fb8071ae9ed057834926d2d012f36213c542b,
      # we hardcoded the filename as view['name'] + '.JPG'
      v.filename = view['name'] + '.JPG'
    if 'camera_name' in view:
      v.camera_name = view['camera_name']
    else:
      # in commit 828fb8071ae9ed057834926d2d012f36213c542b,
      # we set up the rule that view['name'] == camera_name + '_' + filename.stem
      v.camera_name = __from_filename_get_camera_name(view['name'])
    v.intrinsics = intrinsics[v.camera_name]
    v.width = v.intrinsics.width
    v.height = v.intrinsics.height
    v.id = view['id']
    v.pose = extrinsics[v.id]
    v.valid_frame = view['valid_frame']
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

def __parse_tag_structure(structure, views):
  result = {}
  for track in structure['tracks']:
    key = track['tag_id']
    landmark = Landmark()
    landmark.id = track['tag_id']
    landmark.X = track['world_pt']
    for observation in track['obs']:
      view_key = observation['view_id']
      ob = Observation()
      ob.id_feat = observation['view_id']
      ob.x = np.array(observation['image_pt'])
      landmark.observations[views[view_key]] = ob
    result[key] = landmark

  return result

def load_openmvg_sfm_data(file):
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

def load_tag_sfm_data(file):
  content = bson.loads(file.read())
  result = SfMData()
  result.root_path = None
  result.intrinsics = __parse_tag_intrinsics(content['intrinsics'])
  result.extrinsics = __parse_tag_extrinsics(content['views'])
  result.views = __parse_tag_views(content['views'], result.intrinsics, result.extrinsics)
  result.structure = __parse_tag_structure(content['structure'], result.views)

  for landmark in result.structure.values():
    for view in landmark.observations:
      view.observations[landmark] = landmark.observations[view]

  return result

def load_sfm_data(path):
  path = Path(path)
  if path.suffix == '.json':
    with path.open('r') as f:
      return load_openmvg_sfm_data(f)
  elif path.suffix == '.bson':
    with path.open('rb') as f:
      return load_tag_sfm_data(f)
  else:
    raise RuntimeError('Unrecognized SfM file {}'.format(path))

def sfm_to_gl_camera_frame(camera_frame):
  result = camera_frame.copy()
  result[:, 1] = -result[:, 1]
  result[:, 2] = -result[:, 2]
  return result
