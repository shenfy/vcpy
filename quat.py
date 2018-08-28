from numpy import *

class Quat:
  def __init__(self, vals=array([1.0, 0, 0, 0])):
    self.v = vals[:4].copy()

  def set(self, w=1.0, x=0.0, y=0.0, z=0.0):
    self.v = array([w, x, y, z])

  def set_w(self, w):
    self.v[0] = w

  def set_axis(self, axis):
    self.v[1:] = axis

  @classmethod
  def x_rotation(cls, angle):
    return Quat(array([cos(angle / 2), sin(angle / 2), 0, 0]))

  @classmethod
  def y_rotation(cls, angle):
    return Quat(array([cos(angle / 2), 0, sin(angle / 2), 0]))

  @classmethod
  def z_rotation(cls, angle):
    return Quat(array([cos(angle / 2), 0, 0, sin(angle / 2)]))

  @classmethod
  def scaled(cls, q, s):
    return Quat(q.v * s)

  @classmethod
  def inv(cls, q):
    n = dot(q.v, q.v)
    result = Quat()
    result.set(q.v[0], -q.v[1], -q.v[2], -q.v[3])
    result.scale(1.0 / n)
    return result

  @classmethod
  def dot(cls, q0, q1):
    return dot(q0.v, q1.v)

  @classmethod
  def normalize(cls, q):
    return Quat(q.v / sqrt(dot(q.v, q.v)))

  def __str__(self):
    return self.v.__str__()

  def __add__(self, other):
    return Quat(self.v + other.v)

  def __sub__(self, other):
    return Quat(self.v - other.v)

  def __mul__(self, other):
    u = self.v[1:]
    v = other.v[1:]
    result = Quat()
    result.set_w(self.v[0] * other.v[0] - dot(u, v))
    result.set_axis(v * self.v[0] + u * other.v[0] + cross(u, v))
    return result

  def scale(self, s):
    self.v *= s

  def norm2(self):
    return dot(self.v, self.v)

  def norm(self):
    return sqrt(dot(self.v, self.v))

  def apply(self, pos):
    r = self * (Quat(array([0, pos[0], pos[1], pos[2]])) * Quat.inv(self))
    return array([r.v[1], r.v[2], r.v[3], pos[3]])

  def to_mat(q):
    r = identity(4)
    n = dot(q.v, q.v)
    two_over_n = 2.0 / n
    
    r[0, 0] -= (q.v[2] * q.v[2] + q.v[3] * q.v[3]) * two_over_n;
    r[0, 1] += (q.v[1] * q.v[2] - q.v[0] * q.v[3]) * two_over_n;
    r[0, 2] += (q.v[1] * q.v[3] + q.v[2] * q.v[0]) * two_over_n;
    r[1, 0] += (q.v[1] * q.v[2] + q.v[0] * q.v[3]) * two_over_n;
    r[1, 1] -= (q.v[1] * q.v[1] + q.v[3] * q.v[3]) * two_over_n;
    r[1, 2] += (q.v[2] * q.v[3] - q.v[1] * q.v[0]) * two_over_n;
    r[2, 0] += (q.v[1] * q.v[3] - q.v[2] * q.v[0]) * two_over_n;
    r[2, 1] += (q.v[2] * q.v[3] + q.v[1] * q.v[0]) * two_over_n;
    r[2, 2] -= (q.v[1] * q.v[1] + q.v[2] * q.v[2]) * two_over_n;
    return r

  @classmethod
  def from_mat(cls, m):
    four_x2_minus_1 = m[0, 0] - m[1, 1] - m[2, 2];
    four_y2_minus_1 = m[1, 1] - m[0, 0] - m[2, 2];
    four_z2_minus_1 = m[2, 2] - m[0, 0] - m[1, 1];
    four_w2_minus_1 = m[0, 0] + m[1, 1] + m[2, 2];

    biggest_idx = 0
    four_big2_minus_1 = four_w2_minus_1
    if (four_x2_minus_1 > four_big2_minus_1):
      four_big2_minus_1 = four_x2_minus_1
      biggest_idx = 1
    if (four_y2_minus_1 > four_big2_minus_1):
      four_big2_minus_1 = four_y2_minus_1
      biggest_idx = 2
    if (four_z2_minus_1 > four_big2_minus_1):
      four_big2_minus_1 = four_z2_minus_1
      biggest_idx = 3

    big_val = sqrt(four_big2_minus_1 + 1) * 0.5
    mult = 0.25 / big_val

    result = zeros(4)
    if (biggest_idx == 0):
      result[0] = big_val
      result[1] = (m[2, 1] - m[1, 2]) * mult
      result[2] = (m[0, 2] - m[2, 0]) * mult
      result[3] = (m[1, 0] - m[0, 1]) * mult
    elif (biggest_idx == 1):
      result[0] = (m[2, 1] - m[1, 2]) * mult;
      result[1] = big_val;
      result[2] = (m[1, 0] + m[0, 1]) * mult;
      result[3] = (m[0, 2] + m[2, 0]) * mult;
    elif (biggest_idx == 2):
      result[0] = (m[0, 2] - m[2, 0]) * mult;
      result[1] = (m[1, 0] + m[0, 1]) * mult;
      result[2] = big_val;
      result[3] = (m[2, 1] + m[1, 2]) * mult;
    elif (biggest_idx == 3):
      result[0] = (m[1, 0] - m[0, 1]) * mult;
      result[1] = (m[0, 2] + m[2, 0]) * mult;
      result[2] = (m[2, 1] + m[1, 2]) * mult;
      result[3] = big_val;
    return Quat(result)

if __name__ == '__main__':
  q0 = Quat(array([0.483, 0.837, -0.224, 0.129]))
  q1 = Quat(array([0.853, 0.492, 0.150, 0.087]))
  q0 = Quat.normalize(q0)
  q1 = Quat.normalize(q1)
  print(q0)
  print(q1)
  print(q0 + q1)  # 1.336,1.329,-0.074,0.216
  print(Quat.dot(q0, q1))  # 0.801426
  print(q0 * q1)  # 0.02257,0.91276,-0.12797,0.38782
  print(Quat.normalize(q0))
  print(Quat.inv(q0))
  print(q0.to_mat())
  # [0.8664562, -0.4992530, -0.0004377,
  # -0.2501931, -0.4334524, -0.8657496,
  #  0.4320384,  0.7502436, -0.5004772]
  pt = array([3.0, 2, 5, 1])
  print(q0.apply(pt))  # [ 1.5986739, -5.94623229, 0.2942164, 1.]
  print(dot(q0.to_mat(), pt))  # [ 1.5986739, -5.94623229, 0.2942164, 1.]

  m = q0.to_mat()
  print(q0)
  print(Quat.from_mat(m))
