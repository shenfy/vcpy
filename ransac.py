import numpy as np
import random

def ransac_fit(samples, min_samples, max_iterations, inlier_threshold,
  fitter, cost_func, thresh_func, early_term_func):
  
  # samples: array of [k0, n], where n is the number of samples
  num_samples = samples.shape[-1]
  found_model = False

  iteration = 0
  best_err = 1e32

  maybe_model = None

  while iteration < max_iterations:
    
    # select min sample number of random samples
    sample_ids = random.sample(range(num_samples), min_samples)
    maybe_inliers = samples[:, sample_ids]

    # fit model to samples
    maybe_model = fitter(maybe_inliers)

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
        found_model = True

    # early termination
    if found_model and early_term_func(maybe_model, best_err):
      break

    iteration += 1

  return maybe_model

if __name__ == '__main__':
  pass