import numpy as np

def remove_points(points, triangles, mask):
  indices_map = np.cumsum(mask) - 1
  points = points[mask, :]
  triangle_mask = mask[triangles]
  triangle_mask = np.all(triangle_mask, axis=1)
  triangles = triangles[triangle_mask, :]
  triangles = indices_map[triangles]
  return points, triangles

def connected_components(points, triangles):
  n_points = points.shape[0]
  rep = np.arange(n_points)
  def connect(i1, i2):
      if rep[i1] != rep[i2]:
          rep[rep == rep[i2]] = rep[i1]

  for tri in triangles:
      connect(tri[0], tri[1])
      connect(tri[1], tri[2])

  return [rep == c for c in set(rep)]

def select_largest_component(points, triangles):
  return max(connected_components(points, triangles), key=lambda mask: np.sum(mask))

def _edge_id(edge):
  return (edge[0], edge[1]) if edge[0] < edge[1] else (edge[1], edge[0])

def _collect_edges(triangles, mask):
  triangle_mask = np.all(mask[triangles], axis=1)
  triangles = triangles[triangle_mask, :]
  edges = np.row_stack([triangles[:, :2], triangles[:, 1:3],
    np.column_stack([triangles[:, 2], triangles[:, 0]])])
  edges = np.array(list(frozenset(_edge_id(e) for e in edges)))
  return edges

def _generate_new_triangles(triangles, mask, flag, cut_point_index):
  triangle_mask = np.all(mask[triangles], axis=1)
  triangles = triangles[triangle_mask, :]

  triangle_flag = flag[triangles]
  cross_mask = np.logical_and(np.logical_not(np.all(triangle_flag, axis=1)),
    np.any(triangle_flag, axis=1))
  result = []

  for indices, flag in zip(triangles[cross_mask, :], triangle_flag[cross_mask, :]):
    if np.sum(flag) == 1:
      # one point outside
      outside_index = np.where(flag)[0][0]
      outside_point = indices[outside_index]
      pre_point = indices[(outside_index + 2) % 3]
      new_point1 = cut_point_index[_edge_id((pre_point, outside_point))]
      post_point = indices[(outside_index + 1) % 3]
      new_point2 = cut_point_index[_edge_id((outside_point, post_point))]

      result.append([pre_point, new_point1, new_point2])
      result.append([pre_point, new_point2, post_point])
    elif np.sum(flag) == 2:
      # two points outside
      inside_index = np.where(np.logical_not(flag))[0][0]
      inside_point = indices[inside_index]
      new_point1 = cut_point_index[_edge_id((indices[(inside_index + 2) % 3], inside_point))]
      new_point2 = cut_point_index[_edge_id((inside_point, indices[(inside_index + 1) % 3]))]

      result.append([new_point1, inside_point, new_point2])
    else:
      assert False
  return result

def cut_mesh(points, triangles, mask, flag_generator, cut_point_generator):
  # flag indicates which points are goint to be cut out
  flag = flag_generator(points)
  # edges are all unique (i, j) pairs such that i, j are neighbours and i < j
  edges = _collect_edges(triangles, mask)
  # cut_edges are edges that has one vertex to be kept and one vertex to be cut
  cut_edges = edges[flag[edges[:, 0]] != flag[edges[:, 1]], :]
  # cut points are the intersetion of the cut edges and the cut line
  cut_points = cut_point_generator(points[cut_edges[:, 0], :], points[cut_edges[:, 1], :])
  # cut_points_index maps a cut edge to the index of the corresponding cut point
  cut_points_index = {tuple(e): i + points.shape[0] for i, e in enumerate(cut_edges)}
  points = np.row_stack([points, cut_points])
  # generate new triangles
  new_triangles = _generate_new_triangles(triangles, mask, flag, cut_points_index)
  if new_triangles:
    triangles = np.row_stack([triangles, new_triangles])
  # exlude old points from mask and add new points to mask
  mask[flag] = False
  mask = np.concatenate([mask, np.ones(cut_points.shape[0], dtype=mask.dtype)])
  return points, triangles, mask
