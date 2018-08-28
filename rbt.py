from pylab import *
from .quat import *
from . import m3d

class Rbt():
  def __init__(self, t, r):
    if t is not None:
      self.t = t
    else:
      self.t = zeros(3)

    if r is not None:
      self.r = r
    else:
      self.r = Quat()

  def set_t(self, t):
    self.t = t

  def set_r(self, r):
    self.r = r

  def t_rbt(self):
    return Rbt(t=self.t)

  def r_rbt(self):
    return Rbt(r=self.r)

  def __mul__(self, other):
    return Rbt(self.t + self.r.apply(m3d.v3v4(other.t, 0))[:3],
      self.r * other.r)

  def apply(self, pos):
    return m3d.v3v4(self.t, 0) * pos[3] + self.r.apply(pos)

  @classmethod
  def inv(cls, rbt):
    r_inv = Quat.inv(rbt.r)
    return Rbt(r_inv.apply(m3d.v3v4(-rbt.t, 1.0))[:3], r_inv)

  def to_mat(self):
    m = self.r.to_mat()
    m[:3, 3] = self.t
    return m

  @classmethod
  def do_m_2_o_wrt_a(cls, m, o, a):
    return a * m * Rbt.inv(a) * o

  @classmethod
  def look_at(cls, eye, target, up):
    m = m3d.gl_lookat(eye, target, up)
    r = Quat.from_mat(m)
    return Rbt(eye, r)


if __name__ == '__main__':
  pass