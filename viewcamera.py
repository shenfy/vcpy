import bson
import numpy as np
from vcpy.m3d import gl_frustum

class ViewCamera:
  def __init__(self):
    self.name = None
    self.view_mat = np.eye(4)
    self.left = 0
    self.right = 0
    self.bottom = 0
    self.top = 0
    self.znear = None
    self.zfar = None
    self.width = None
    self.height = None

  def projection_mat(self, dmin=None, dmax=None):
    left = self.left
    right = self.right
    bottom = self.bottom
    top = self.top

    if (dmin != None) & (dmax != None):
      if (self.znear != None) & (self.znear != 1):
        left /= self.znear
        right /= self.znear
        bottom /= self.znear
        top /= self.znear
      return gl_frustum(left * dmin, right * dmin, bottom * dmin, top * dmin, dmin, dmax)
    elif (self.znear != None) & (self.zfar != None):
      return gl_frustum(left, right, bottom, top, self.znear, self.zfar)
    else:
      raise RuntimeError('can not calculate projection matrix')

class ViewCameras:
  def __init__(self):
    self.cameras = []

  def dump(self, f):
    views_data = {}
    views_data['views'] = []
    for view in self.cameras:
      view_data = {}
      if view.name != None:
        view_data['name'] = view.name
      view_data['view_mat'] = view.view_mat.tolist()
      view_data['left'] = view.left
      view_data['right'] = view.right
      view_data['bottom'] = view.bottom
      view_data['top'] = view.top
      if view.znear != None:
        view_data['znear'] = view.znear
      if view.zfar != None:
        view_data['zfar'] = view.zfar
      if view.width != None:
        view_data['width'] = view.width
      if view.height != None:
        view_data['height'] = view.height
      views_data['views'].append(view_data)

    f.write(bson.dumps(views_data))

def load_view_camera_data(file_fn):
  with open(file_fn, 'rb') as f:
    content = bson.loads(f.read())
    result = ViewCameras()
    for view_data in content['views']:
      view_camera = ViewCamera()
      if 'name' in view_data:
        view_camera.name = view_data['name']
      view_camera.view_mat = np.array(view_data['view_mat'])
      view_camera.left = view_data['left']
      view_camera.right = view_data['right']
      view_camera.bottom = view_data['bottom']
      view_camera.top = view_data['top']
      if 'znear' in view_data:
        view_camera.znear = view_data['znear']
      if 'zfar' in view_data:
        view_camera.zfar = view_data['zfar']
      if 'width' in view_data:
        view_camera.width = view_data['width']
      if 'height' in view_data:
        view_camera.height = view_data['height']
      result.cameras.append(view_camera)

  return result
