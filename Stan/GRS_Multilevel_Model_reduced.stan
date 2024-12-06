
data {
  int<lower=0> n_obs; // number of fisherman responses
  int<lower=0> n_fisher; // number of fishermen in survey
  int<lower=0> tmax; // maximum number of time steps
  int<lower=0> year[n_obs]; // response years
  real<lower=0> CPUE[n_obs]; // adjusted CPUE data
  int<lower=0> n_boats[tmax]; // number of fishing boats
  int<lower=0> fisher_id[n_obs]; // unique id for each fisherman
  real<lower=0> K; // carrying capacity
  real midpoint; // midpoint of sigmoid fishing days curve
  real slope; // slope of sigmoid fishing days curve
  real lambda_alpha; // Shape of shifted prior distribution for lambda
  real lambda_beta; // Rate of shifted prior distribution for lambda
  real mu_dmax; // Mean of prior distribution for dmax
  real sd_dmax; // SD of prior distribution for dmax
}

parameters {
  real log_q; // log of overall catchability
  real<lower = 0> sigma_q; // standard deviation of fisher catchabilities
  real<lower = 0> q_fisher_std[n_fisher]; // standardized fisher catchability scores
  real<lower = 0> sigma_cpue; // standard deviation of observation error
  real<lower = 0> lambda_shift; // population finite growth rate
  real dmax_logit; // Proportion of days spent fishing, logit scale
}

transformed parameters {

  real N[tmax]; // Underlying population size
  real Np[tmax]; // No. individuals before fishing in each time step
  real mu_cpue[tmax, n_fisher]; // Expected CPUE for each fisher across time
  real q_fisher[n_fisher]; // catchability for each fisher
  real lambda = 1 + lambda_shift; // finite growth rate
  real dmax = inv_logit(dmax_logit) * 365; // maximum number of fishing days
  real q = exp(log_q);
  
  // Implies q_fisher ~ normal(q, sigma_q)
  for (i in 1:n_fisher) {
    q_fisher[i] = exp(log_q + sigma_q * q_fisher_std[i]);
  }
  
  N[1] = K; // Initialize population size at carrying capacity
  
  for (i in 1:tmax) {
    Np[i] = lambda*N[i] / (1 + ((lambda - 1)/K)*N[i]); 
    if (i < tmax) {
      N[i + 1] = Np[i] * exp(-q * dmax * n_boats[i]);
    }
    for (j in 1:n_fisher) {
      mu_cpue[i, j] = Np[i] * (1 - exp(-q_fisher[fisher_id[j]]));
    }
  }
  
}

model {
  log_q ~ normal(-5, 1);
  sigma_cpue ~ exponential(3);
  sigma_q ~ normal(0, 0.05);
  q_fisher_std ~ std_normal();
  dmax_logit ~ normal(mu_dmax, sd_dmax); // empirical prior
  lambda_shift ~ gamma(lambda_alpha, lambda_beta); // empirical prior

  for (i in 1:n_obs) {
    // lognormal likelihood for observed data
    CPUE[i] ~ lognormal(log(mu_cpue[year[i], fisher_id[i]]), sigma_cpue);
  }
}

generated quantities {
 
  real CPUE_pred[tmax, n_fisher];
  real log_lik[n_obs];
  
  for (i in 1:tmax) {
    for (j in 1:n_fisher) {
      CPUE_pred[i, j] = lognormal_rng(log(mu_cpue[i,j]), sigma_cpue);
    }
  }
  
  for (i in 1:n_obs) {
    log_lik[i] = lognormal_lpdf(CPUE[i] | log(mu_cpue[year[i], fisher_id[i]]), sigma_cpue);
  }
  
}
