data {
    int N; // number of obs (pregnancies)
    int M; // number of groups (women)
    int K; // number of predictors
    
    int y[N]; // outcome
    row_vector[K] x[N]; // predictors
    int g[N];    // map obs to groups (pregnancies to women)
    int test_N;
    int test_M;
    int test_y[test_N];
    row_vector[K] test_x[test_N];
    int test_g[test_N];
    real test_a[test_M];
}
parameters {
    real alpha;
    real a[M]; 
    vector[K] beta;
    real<lower=0,upper=10> sigma;  
}
model {
  sigma ~ gamma(1,1);
  alpha ~ normal(0,10);
  a ~ normal(0,sigma);
  beta ~ normal(0,10);
  for(n in 1:N) {
    y[n] ~ poisson(exp( alpha + a[g[n]] + x[n]*beta));
  }
}

generated quantities {

  real test_loglike;
  test_loglike <- 0;
  for(n in 1:test_N) {
     test_loglike += poisson_lpmf(test_y[n] | exp(alpha + test_a[test_g[n]] + test_x[n]*beta));
  }
}
