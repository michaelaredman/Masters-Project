functions {
  /**
  * This part is taken from Max Joseph's write-up here: http://mc-stan.org/documentation/case-studies/mbjoseph-CARStan.html
  *
  * Return the log probability of a proper conditional autoregressive (CAR) prior 
  * with a sparse representation for the adjacency matrix
  *
  * @param phi Vector containing the parameters with a CAR prior
  * @param tau Precision parameter for the CAR prior (real)
  * @param alpha Dependence (usually spatial) parameter for the CAR prior (real)
  * @param W_sparse Sparse representation of adjacency matrix (int array)
  * @param n Length of phi (int)
  * @param W_n Number of adjacent pairs (int)
  * @param D_sparse Number of neighbors for each location (vector)
  * @param lambda Eigenvalues of D^{-1/2}*W*D^{-1/2} (vector)
  *
  * @return Log probability density of CAR prior up to additive constant
  */
  real sparse_car_lpdf(vector phi, real tau, real alpha, 
    int[,] W_sparse, vector D_sparse, vector lambda, int n, int W_n) {
      row_vector[n] phit_D; // phi' * D
      row_vector[n] phit_W; // phi' * W
      vector[n] ldet_terms;
    
      phit_D = (phi .* D_sparse)';
      phit_W = rep_row_vector(0, n);
      for (i in 1:W_n) {
        phit_W[W_sparse[i, 1]] = phit_W[W_sparse[i, 1]] + phi[W_sparse[i, 2]];
        phit_W[W_sparse[i, 2]] = phit_W[W_sparse[i, 2]] + phi[W_sparse[i, 1]];
      }
    
      for (i in 1:n) ldet_terms[i] = log1m(alpha * lambda[i]);
      return 0.5 * (n * log(tau)
                    + sum(ldet_terms)
                    - tau * (phit_D * phi - alpha * (phit_W * phi)));
  }
}
data {
    int<lower = 1> numRegions; // The number of regions
    int<lower = 1> nt; // The number of time points
    // matrix[numRegions, nt] observed;  // The observed counts at each point
    int observed[numRegions, nt];
    vector[numRegions] log_expected; // The expected number of counts based on demographics etc
    matrix<lower = 0, upper = 1>[numRegions, numRegions] W; // The adjacency matrix
    int W_n;
}
transformed data{
    int W_sparse[W_n, 2];   // adjacency pairs
    vector[numRegions] D_sparse;     // diagonal of D (number of neigbors for each site)
    vector[numRegions] lambda;       // eigenvalues of invsqrtD * W * invsqrtD
    
    { // generate sparse representation for W
	int counter;
	counter = 1;
	// loop over upper triangular part of W to identify neighbor pairs
	for (i in 1:(numRegions - 1)) {
	    for (j in (i + 1):numRegions) {
		if (W[i, j] == 1) {
		    W_sparse[counter, 1] = i;
		    W_sparse[counter, 2] = j;
		    counter = counter + 1;
		}
	    }
	}
    }
    for (i in 1:numRegions) D_sparse[i] = sum(W[i]);
    {
	vector[numRegions] invsqrtD;  
	for (i in 1:numRegions) {
	    invsqrtD[i] = 1 / sqrt(D_sparse[i]);
	}
	lambda = eigenvalues_sym(quad_form(W, diag_matrix(invsqrtD)));
    }
}
parameters {
    vector[nt] temporal;
    vector[numRegions] v; // Spatial smoothing component
    vector[numRegions] lmbda; 
    real<lower = 0> sigma_v; // Variance of spatial component, v
    real<lower = 0> sigma_temporal; // Variance of temporal component
    real<lower = 0> sigma_l;
    real<lower = 0> var_ind_temporal[numRegions]; // The variance of the individual temporal trends
    real<lower = 0, upper = 1> alpha; // The degree of spatial dependence - implicitly given flat prior
    
    matrix[numRegions, nt] ind_temporal; // The temporal trend of an individual region
    vector[numRegions] ind_constant;

    vector<lower=0, upper=1>[numRegions] prop_unusual;

    real a;
    real<lower = 0> b;


}
transformed parameters {
    matrix[numRegions, nt] mu_general;
    matrix[numRegions, nt] mu_specific;

    // reparameterize mu_general
    // lmbda(v, sigma_l) == v + N(0, 1) * sigma_l

    // change variance parameters to be Gamma(2, 0) maybe? as in http://www.stat.columbia.edu/~gelman/research/published/chung_etal_Pmetrika2013.pdf
    
    mu_general = rep_matrix(temporal, numRegions)' + rep_matrix(lmbda, nt) + rep_matrix(log_expected, nt);
    mu_specific = ind_temporal + rep_matrix(ind_constant, nt) + rep_matrix(log_expected, nt);
    
}
model {
    sigma_temporal ~ normal(0, 1);
    sigma_v ~ normal(0, 1);
    sigma_l ~ normal(0, 1);
    
    temporal[1] ~ normal(0, sigma_temporal); // 1d random walk prior on temporal component
    ind_temporal[,1] ~ normal(0, sqrt(var_ind_temporal));
    for (t in 2:nt) {
	temporal[t] ~ normal(temporal[t - 1], sigma_temporal);
	ind_temporal[,t] ~ normal(ind_temporal[, t - 1], sqrt(var_ind_temporal));
    }

    ind_constant ~ normal(0, 30); // non-informative prior on the constant term per region

    var_ind_temporal ~ lognormal(a, b);
    a ~ normal(0, 30);
    b ~ normal(0, 2.5);

    prop_unusual ~ beta(2, 40);
    
    v ~ sparse_car(sigma_v, alpha, W_sparse, D_sparse, lambda, numRegions, W_n);
    lmbda ~ normal(v, sigma_l);

    for (i in 1:numRegions){
	target += log_sum_exp(log(1.0 - prop_unusual[i]) + poisson_log_lpmf(observed[i,] | mu_general[i,]),
			      log(prop_unusual[i]) + poisson_log_lpmf(observed[i,] | mu_specific[i,]));
    }

}
