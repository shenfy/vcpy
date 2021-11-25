import numpy as np
import random

# Raise this exception to tell RANSAC it has selected a degenerate model
# and should try another sample.
class DegenerateModelError(RuntimeError):
  pass

def ransac_fit(samples, min_samples, max_iterations, inlier_threshold,
  fitter, cost_func, thresh_func, early_term_func):

  # samples: array of [k0, n], where n is the number of samples
  num_samples = samples.shape[-1]

  iteration = 0
  best_err = 1e32

  result = None

  while iteration < max_iterations:

    # select min sample number of random samples
    sample_ids = random.sample(range(num_samples), min_samples)
    maybe_inliers = samples[:, sample_ids]

    # fit model to samples
    try:
      maybe_model = fitter(maybe_inliers)
    except DegenerateModelError:
      continue

    # calc inlier percentage
    errs = cost_func(maybe_model, samples)
    threshold = thresh_func(maybe_model)
    inlier_mask = (errs < threshold).flatten()
    num_inliers = np.count_nonzero(inlier_mask)
    inlier_percent = num_inliers / num_samples

    # fit model
    if inlier_percent > inlier_threshold:
      # refit model with all inliers
      inliers = samples[:, inlier_mask]
      maybe_model = fitter(inliers)
      errs = cost_func(maybe_model, inliers)
      ss_err = np.sum(errs ** 2) / num_inliers

      if (ss_err < best_err):
        best_err = ss_err
        result = maybe_model
        if early_term_func(result, best_err):
          break

    iteration += 1

  return result
