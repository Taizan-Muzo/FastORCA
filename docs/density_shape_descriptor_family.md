# Density Shape Descriptor Family (FastORCA Enhancement)

This page documents FastORCA's density-shape substitute descriptors.  
These are **open-source enhancement descriptors**, not qcMol exact internal formulas.

## Scope

- Canonical single-scale view:
  - `realspace_features.density_shape_descriptor_family_v1`
  - default mass cutoff: `0.95`
- Multiscale companion:
  - `realspace_features.density_shape_multiscale_family_v1`
  - scales: `0.50 / 0.90 / 0.95`

## Core Algorithm

For each mass cutoff scale:

1. Flatten grid density values from realspace cube.
2. Keep finite positive density points.
3. Sort by density descending.
4. Select top points until cumulative density reaches target mass fraction.
5. Compute density-weighted center.
6. Compute density-weighted covariance tensor (`shape_tensor`).
7. Eigen decomposition (`lambda1 >= lambda2 >= lambda3 >= 0`).
8. Emit raw and normalized eigenvalues.
9. Compute descriptors.

## Formulas

- `lambda_i_norm = lambda_i / (lambda1 + lambda2 + lambda3 + eps)`
- `sphericity = 3*lambda3 / (lambda1 + lambda2 + lambda3 + eps)`
- `asphericity = (lambda1 - 0.5*(lambda2 + lambda3)) / (lambda1 + lambda2 + lambda3 + eps)`
- `anisotropy = ((lambda1-lambda2)^2 + (lambda2-lambda3)^2 + (lambda3-lambda1)^2) / (2*(lambda1+lambda2+lambda3)^2 + eps)`
- `relative_anisotropy_kappa2 = 1 - 3*(lambda1n*lambda2n + lambda2n*lambda3n + lambda3n*lambda1n)`
- `elongation = (lambda1 - lambda2) / (lambda1 + eps)`
- `planarity = (lambda2 - lambda3) / (lambda1 + eps)`

## Semantics

- Raw eigenvalues are **size-sensitive** (depend on absolute cloud extent).
- Normalized eigenvalues and `kappa2` are **shape-relative** (better for cross-size comparison).
- Multiscale meaning:
  - `0.50`: dense core shape
  - `0.90`: intermediate shell
  - `0.95`: near-complete cloud shape

## Stability & Limits

- `eps = 1e-12` for denominator stability.
- If density is all-zero, status is `unavailable` with `density_all_zero`.
- Values are clipped to maintain non-negative eigenvalues and bounded shape metrics.
- Outputs depend on density source and grid settings:
  - `density_source_*`
  - `cube_grid_shape`
  - `cube_spacing_angstrom`
  - `margin_angstrom`
