data {
    int<lower = 1> numRegions; // The number of regions
    int<lower = 1> nt; // The number of time points
    int observed[numRegions, nt];
    vector[numRegions] log_expected; // The expected number of counts based on demographics etc
}
parameters {
    real<lower = 0> var_ind_temporal[numRegions]; // The variance of the individual temporal trends
    
    matrix[numRegions, nt] ind_temporal; // The temporal trend of an individual region
    vector[numRegions] ind_constant;

    real a;
    real<lower = 0> b;
}
transformed parameters {
    matrix[numRegions, nt] mu_specific;
    
    mu_specific = ind_temporal + rep_matrix(ind_constant, nt) + rep_matrix(log_expected, nt);
}
model {
    
    ind_temporal[,1] ~ normal(0, sqrt(var_ind_temporal)); 
    for (t in 2:nt) {
	ind_temporal[,t] ~ normal(ind_temporal[, t - 1], sqrt(var_ind_temporal));
    }

    ind_constant ~ normal(0, 30); // non-informative prior on the constant term per region

    var_ind_temporal ~ lognormal(a, b);
    a ~ normal(0, 30);
    b ~ normal(0, 2.5);
    
    for (t in 1:nt) {
	observed[,t] ~ poisson_log(mu_specific[,t]);
    }

}
