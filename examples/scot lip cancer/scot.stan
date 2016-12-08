data {
  int<lower = 1> n; # The number of observations
  int<lower = 1> p; # The number of coefficients
  matrix[n, p] X;  # The design matrix
  int<lower = 0> y[n]; # The IVs
  vector[n] log_offset;
  matrix<lower = 0, upper = 1>[n, n] W;
}
transformed data{
  vector[n] zeros;
  matrix<lower = 0>[n, n] D;
  {
    vector[n] W_rowsums;
    for (i in 1:n) {
      W_rowsums[i] <- sum(row(W, i));
    }
    D <- diag_matrix(W_rowsums);
  }
  zeros <- rep_vector(0, n);
}
parameters {
  vector[p] beta;
  vector[n] phi;
  real<lower = 0> tau;
  real<lower = 0, upper = 1> alpha;
}
model {
  phi ~ multi_normal_prec(zeros, tau * (D - alpha * W));
  beta ~ normal(0, 1);
  tau ~ gamma(2, 2);
  y ~ poisson_log(X * beta + phi + log_offset);
}
