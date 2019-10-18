import json
import numpy as np

class Intrinsics:
  def __init__(self):
    self.camera_type = ''
    self.width = 0
    self.height = 0
    self.focal = 0
    self.cx = 0
    self.cy = 0
    self.distortions = []

  @property
  def K(self):
    result = np.identity(3, dtype=np.float64)
    result[0, 0] = result[1, 1] = self.focal
    result[0, 2] = self.cx
    result[1, 2] = self.cy
    return result

def projection_from_intrinsics(intrinsics, z_near, z_far):
  cx = intrinsics.cx + 0.5
  cy = intrinsics.cy + 0.5

  left = -cx / intrinsics.focal
  right = (intrinsics.width - cx) / intrinsics.focal
  bottom = -(intrinsics.height - cy) / intrinsics.focal
  top = cy / intrinsics.focal
  result = (left * z_near, right * z_near, bottom * z_near, top * z_near, z_near, z_far)
  return result

class SfMCamera:
  def __init__(self):
    self.intrinsics = None
    self.camera_frame = np.identity(4, dtype=np.double)
    self.filename = ''

def _parse_views(doc):
  views = doc['views']
  result = []
  for view in views:
    data = view['value']['ptr_wrapper']['data']
    intrinsic_id = data['id_intrinsic']
    extrinsic_id = data['id_pose']
    filename = data['filename']
    result.append((intrinsic_id, extrinsic_id, filename))
  return result

def _parse_intrinsics(doc):
  result = {}
  for intrinsic_data in doc['intrinsics']:
    key = intrinsic_data['key']
    value = intrinsic_data['value']
    if value['polymorphic_name'] == 'pinhole_radial_k3':
      intrinsics = Intrinsics()
      intrinsics.camera_type = 'pinhole_radial_k3'
      data = value['ptr_wrapper']['data']
      intrinsics.width = data['width']
      intrinsics.height = data['height']
      intrinsics.focal = data['focal_length']
      intrinsics.cx = data['principal_point'][0]
      intrinsics.cy = data['principal_point'][1]
      intrinsics.distortions = data['disto_k3']
      result[key] = intrinsics
  return result

def _parse_extrinsics(doc):
  result = {}
  for extrinsic in doc['extrinsics']:
    key = extrinsic['key']
    value = extrinsic['value']
    camera_frame = np.identity(4, dtype=np.double)
    camera_frame[:3, :3] = np.array(value['rotation']).T
    camera_frame[:3, 3] = np.array(value['center'])
    result[key] = camera_frame
  return result

def load_sfm_cameras(filename):
  with open(filename) as f:
    doc = json.load(f)

  views = _parse_views(doc)
  intrinsics = _parse_intrinsics(doc)
  extrinsics = _parse_extrinsics(doc)

  result = []
  for view in views:
    if view[0] in intrinsics and view[1] in extrinsics:
      sfm_camera = SfMCamera()
      sfm_camera.filename = view[2]
      sfm_camera.intrinsics = intrinsics[view[0]]
      sfm_camera.camera_frame = extrinsics[view[1]]
      result.append(sfm_camera)
    else:
      print('{} has no camera parameter'.format(view[2]))
  return result

def sfm_to_gl_camera_frame(camera_frame):
  result = camera_frame.copy()
  result[:, 1] = -result[:, 1]
  result[:, 2] = -result[:, 2]
  return result
