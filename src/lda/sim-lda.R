library("gtools");  # for rdirichlet
library("rstan")
library(parallel)
options(mc.cores = detectCores() - 1)
library(ggplot2)

args = commandArgs(trailingOnly=TRUE)
M <- args[1] # number of tarining documents

V <- 100; # words: river, stream, bank, money, loan
K0 <- 10; # topics: RIVER, BANK
K = 8 # number of topics we fit
avg_doc_length <- 100;
alpha <- c(rep(2,K), rep(0.5, K0-K));
beta <- rep(1,V);
phi <- rdirichlet(K0,beta); 
# dump(c("phi"), "truephi.data.R")
# phi = phi
# source("truephi.data.R")
truephi0 = phi
truephi = cbind(phi, rep(0, K0))
colnames(truephi)[101] = "topic"


# create test data
test_M = 1000
doc_length <- rpois(test_M,avg_doc_length);
test_N <- sum(doc_length);
theta <- rdirichlet(test_M,alpha); # local variables, topic dist per doc
#global variables, word dist per topic
test_w <- rep(NA,test_N);
test_doc <- rep(NA,test_N);
n <- 1;
for (m in 1:test_M) {
  for (i in 1:doc_length[m]) {
    z <- which(rmultinom(1,1,theta[m,]) == 1);
    test_w[n] <- which(rmultinom(1,1,phi[z,]) == 1);
    test_doc[n] <- m;
    n <- n + 1;
  }
}





# select the most important K components

truephi0 = phi[1:K,] # replace with the best model misspec

for (M in c(10)){
    # M <- 1000;  # docs
    doc_length <- rpois(M,avg_doc_length);
    N <- sum(doc_length);
    theta <- rdirichlet(M,alpha); #local variables, topic dist per doc
    #global variables, word dist per topic
    w <- rep(NA,N);
    doc <- rep(NA,N);
    n <- 1;
    for (m in 1:M) {
      for (i in 1:doc_length[m]) {
        z <- which(rmultinom(1,1,theta[m,]) == 1);
        w[n] <- which(rmultinom(1,1,phi[z,]) == 1);
        doc[n] <- m;
        n <- n + 1;
      }
    }
    alpha <- rep(1,K);
    truephi = cbind(truephi0, rep(0, K))
    test_theta = rdirichlet(test_M,alpha);

    data_file_name = paste(paste("lda_M", M, sep=""),".data.R", sep="")
    dump(c("K","V","M","N","z","w","doc","alpha","beta", "theta","phi", "truephi0", 
        "test_M", "test_N", "test_doc", "test_w", "test_theta"),data_file_name)

    output_file_csv = paste(paste("lda_M_hmc", M, sep=""),"output.csv", sep="")




    fitlda = stan(file = 'lda.stan', data = list(K = K, V = V, M = M, N = N, 
                w = w, doc = doc, alpha = alpha, beta = beta, 
                test_M=test_M, test_N=test_N, test_w=test_w, test_doc=test_doc, 
                test_theta=test_theta),
                iter = 1000, chains = 4, sample_file = output_file_csv)
    phifit = extract(fitlda)$phi
    topic_word_dist_hmc = truephi
    for(i in 1:K){
        temp = cbind(phifit[,i,], rep(i, 2000))
        topic_word_dist_hmc = rbind(topic_word_dist_hmc, temp)
    }
    output_file_name = paste(paste("lda_M_hmc", M, sep=""),"phi_output.csv", sep="")
    write.csv (phifit, output_file_name, row.names = F, quote = F)

    output_file_name_kl = paste(paste("lda_M_hmc_kl", M, sep=""),".csv", sep="")    
    write.csv(extract(fitlda)$klphi, output_file_name_kl, row.names = F, quote = F)
    print(summary(extract(fitlda)$klphi))


    output_file_name_test_loglike = paste(paste("lda_M_hmc_test_loglike", M, sep=""),".csv", sep="")    
    write.csv(extract(fitlda)$test_loglike, output_file_name_test_loglike, row.names = F, quote = F)
    print(summary(extract(fitlda)$test_loglike))


    ldam = stan_model(file = 'lda.stan', verbose = TRUE)
    source(data_file_name)
    output_file_csv = paste(paste("lda_M_vb", M, sep=""),"output.csv", sep="")
    ldafit = vb(ldam, data = list(K = K, V = V, M = M, N = N, 
                w = w, doc = doc, alpha = alpha, beta = beta), tol_rel_obj = 1e-3, 
                sample_file = output_file_csv)
    phifit = extract(ldafit)$phi
    topic_word_dist_vb = truephi
    for(i in 1:K){
        temp = cbind(phifit[,i,], rep(i, 1000))
        topic_word_dist_vb = rbind(topic_word_dist_vb, temp)
    }
    output_file_name = paste(paste("lda_M_vb", M, sep=""),"phi_output.csv", sep="")
    write.csv (phifit, output_file_name, row.names = F, quote = F)
    output_file_name_kl = paste(paste("lda_M_vb_kl", M, sep=""),".csv", sep="")    
    write.csv(extract(ldafit)$klphi, output_file_name_kl, row.names = F, quote = F)
    print(summary(extract(ldafit)$klphi))

    output_file_name_test_loglike = paste(paste("lda_M_vb_test_loglike", M, sep=""),".csv", sep="")    
    write.csv(extract(ldafit)$test_loglike, output_file_name_test_loglike, row.names = F, quote = F)
    print(summary(extract(ldafit)$test_loglike))
