# REDD intensity - Bayesian approach

# Following McElrath Statistical rethinking

# 1) Draw DAG
# 2) Write down the model
# 3) test model with simulated data
# 4) Fit model to real data

##### Setup ####
#install.packages(c("coda","mvtnorm","devtools","loo","dagitty","shape"))
#devtools::install_github("rmcelreath/rethinking")

library(rethinking)
library(dagitty)
library(dplyr)
library(MASS)

#### helper functions ####


#### Make DAG ####
dag <- dagitty("dag {

D_REDD -> Y
REDD_any -> D_REDD
X -> D_pre
U -> Y
U -> D_pre
D_pre -> D_REDD
D_pre -> Y

D_REDD [exposure]
REDD_any [adjusted]
Y [outcome]
X [adjusted]
D_pre [exposure]
U [unobserved]
               
}", layout = TRUE)

dagitty::coordinates(dag) <- list(
  x = c(TI_REDD = -1, REDD_any = -2, Def = 1,  X = -2, TI_pre = -1),
  y = c(TI_REDD = 1, REDD_any = 1, Def = 0,  X = -1, TI_pre = -1)
)

drawdag(dag)
# Check conditional independences
dagitty::impliedConditionalIndependencies(dag)


#### Build simple model ####

# true effect of T on Y can be Null, linear positive, or with squared term

# Simulate data

n <- 1000 # number of observations
true_linpos <- 5 # true linear positive effect
true_sqpos <- 10 # true squared positive effect
true_sqneg <- -10 # true squared negative effect

d.null <- data.frame(
  TI = sample(seq(0,1, 0.1), n, replace = T), # Treatment intensity
  Y = rnorm(n, 0, 1)   # Outcome variable
)

d.linpos <- data.frame(
  TI = sample(seq(0,1, 0.1), n, replace = T) # Treatment intensity
) |>
  mutate(Y = rnorm(n, 0, 1) + true_linpos * TI) # Outcome variable with linear effect

d.sqpos <- data.frame(
  TI = sample(seq(0,1, 0.1), n, replace = T) # Treatment intensity
) |>
  mutate(Y = rnorm(n, 0, 1) + true_linpos * TI + true_sqpos * TI^2) # Outcome variable with squared positive effect

d.sqneg <- data.frame(
  TI = sample(seq(0,1, 0.1), n, replace = T) # Treatment intensity
) |>
  mutate(Y = rnorm(n, 0, 1) + true_linpos * TI + true_sqneg * TI^2) # Outcome variable with squared negative effect

# fit same model to all datasets 

m.out <- lapply(
  list(d.null, d.linpos, d.sqpos, d.sqneg),
  function(data) {
    quap(
      alist(
        Y ~ dnorm(mu, sigma),
        mu <- a + b1 * TI + b2 * TI^2,
        a ~ dnorm(0, 1),
        b1 ~ dnorm(0, 1),
        b2 ~ dnorm(0, 1),
        sigma ~ dexp(1)
      ),
      data = data
    )
  }
)

# make plot with all datasets on Y axis, and respective plot(model) next to it on X
par(mfrow=c(4,2))
plot(d.null$TI, d.null$Y, main="Null Effect", xlab="Treatment Intensity (TI)", ylab="Outcome (Y)", pch=19, col=rgb(0.2, 0.5, 0.8, 0.5))
plot(m.out[[1]])
plot(d.linpos$TI, d.linpos$Y, main="Linear Positive Effect", xlab="Treatment Intensity (TI)", ylab="Outcome (Y)", pch=19, col=rgb(0.2, 0.5, 0.8, 0.5))
plot(m.out[[2]])
plot(d.sqpos$TI, d.sqpos$Y, main="Squared Positive Effect", xlab="Treatment Intensity (TI)", ylab="Outcome (Y)", pch=19, col=rgb(0.2, 0.5, 0.8, 0.5))
plot(m.out[[3]])
plot(d.sqneg$TI, d.sqneg$Y, main="Squared Negative Effect", xlab="Treatment Intensity (TI)", ylab="Outcome (Y)", pch=19, col=rgb(0.2, 0.5, 0.8, 0.5))
plot(m.out[[4]])


### Step 2: let TI be function of REDD_any and TI_pre, which are (negatively) correlated

# Simulate data with REDD_any and TI_pre
n <- 1000 # number of observations

# generate_groups <- function(x, k) {
#   # Input validation
#   if (!is.numeric(x) || any(x < 0) || any(x > 1)) {
#     stop("x must be a numeric vector with values between 0 and 1.")
#   }
#   if (length(x) %% 2 != 0) {
#     stop("The length of x must be an even number so that groups A and B can have equal size.")
#   }
#   if (k < -1 || k > 1) {
#     stop("k must be a value between -1 and 1 (inclusive).")
#   }
#   
#   total_n <- length(x)
#   n_half <- total_n / 2
#   
#   # Randomly select indices for Group B (renamed to C for control based on your code)
#   c_indices <- sample(1:total_n, size = n_half, replace = FALSE)
#   t_indices <- setdiff(1:total_n, c_indices) # Renamed to T for treatment
#   
#   # Assign preTI values and groups
#   final_preTI <- x
#   group_assignment <- rep("C", total_n)
#   group_assignment[t_indices] <- "T"
#   
#   # Generate skewed preTI values for Group T (formerly Group A)
#   # Adjust shape parameters based on k
#   if (k == 0) {
#     # Uniform distribution when k = 0 (alpha = 1, beta = 1)
#     shape1_T <- 1
#     shape2_T <- 1
#   } else if (k > 0) {
#     # Skewed to the right (higher values more probable) as k approaches 1
#     # For k=1, shape1=large, shape2=1
#     shape1_T <- 1 + k * 9 # Max shape1_T = 10 when k=1
#     shape2_T <- 1
#   } else { # k < 0
#     # Skewed to the left (lower values more probable) as k approaches -1
#     # For k=-1, shape1=1, shape2=large
#     shape1_T <- 1
#     shape2_T <- 1 + abs(k) * 9 # Max shape2_T = 10 when k=-1
#   }
#   
#   preTI_T_generated <- rbeta(n_half, shape1 = shape1_T, shape2 = shape2_T)
#   
#   # Assign generated preTI values to Group T
#   final_preTI[t_indices] <- preTI_T_generated
#   
#   # Combine results into a data frame, preserving original order
#   preTI_data <- data.frame(
#     TI_pre = final_preTI,
#     Group = factor(group_assignment, levels = c("T", "C"))
#   )
#   
#   # print shape parameters to console
#   cat("Shape parameters for Group T: shape1 =", shape1_T, ", shape2 =", shape2_T, "\n")
#   
#   return(preTI_data)
# }
# 
# generate_correlated_vector <- function(x, rho, mean_y = NULL, sd_y = NULL) {
#   # Input validation
#   if (!is.numeric(x) || length(x) < 2) {
#     stop("Input 'x' must be a numeric vector with at least two elements.")
#   }
#   if (!is.numeric(rho) || length(rho) != 1 || rho < -1 || rho > 1) {
#     stop("Input 'rho' must be a single numeric value between -1 and 1 (inclusive).")
#   }
#   
#   n <- length(x)
#   mu_x <- mean(x)
#   sd_x <- sd(x)
#   
#   # Determine mean and standard deviation for the new vector y
#   # If not specified, default to the same mean and SD as x
#   if (is.null(mean_y)) {
#     mean_y <- mu_x
#   }
#   if (!is.numeric(mean_y) || length(mean_y) != 1) {
#     stop("Input 'mean_y' must be a single numeric value.")
#   }
#   
#   if (is.null(sd_y)) {
#     sd_y <- sd_x
#   }
#   if (!is.numeric(sd_y) || length(sd_y) != 1 || sd_y <= 0) {
#     stop("Input 'sd_y' must be a single positive numeric value.")
#   }
#   
#   # --- Handle perfect correlation/anti-correlation cases separately ---
#   # This avoids potential floating-point precision issues with sqrt(1 - rho^2)
#   # when rho is extremely close to 1 or -1.
#   if (rho == 1) {
#     # Perfect positive correlation: Y is a direct linear transformation of X
#     # Y = mean_y + (X - mean_x) * (sd_y / sd_x)
#     y_correlated <- mean_y + (x - mu_x) * (sd_y / sd_x)
#     return(y_correlated)
#   } else if (rho == -1) {
#     # Perfect negative correlation: Y is an inverse linear transformation of X
#     # Y = mean_y - (X - mean_x) * (sd_y / sd_x)
#     y_correlated <- mean_y - (x - mu_x) * (sd_y / sd_x)
#     return(y_correlated)
#   }
#   
#   # --- General case using Cholesky decomposition principles ---
#   
#   # 1. Standardize the input vector x
#   # This converts x to a standard normal-like variable (mean ~0, sd ~1)
#   # This is equivalent to the first independent variable (Z1) in the Cholesky transformation
#   x_std <- (x - mu_x) / sd_x
#   
#   # 2. Generate an independent standard normal random variable for the 'noise' component
#   # This is equivalent to the second independent variable (Z2) in the Cholesky transformation
#   epsilon_std <- rnorm(n)
#   
#   # 3. Create the standardized correlated variable (y_std)
#   # This formula is derived directly from the Cholesky decomposition of a 2x2 correlation matrix
#   # R = [[1, rho], [rho, 1]] where the Cholesky factor L = [[1, 0], [rho, sqrt(1-rho^2)]]
#   # If we have [Z1, Z2]^T (where Z1=x_std, Z2=epsilon_std), then the correlated vector
#   # [Y1_std, Y2_std]^T = L %*% [Z1, Z2]^T
#   # Y1_std = Z1 (which is x_std, so x is preserved)
#   # Y2_std = rho * Z1 + sqrt(1 - rho^2) * Z2
#   y_std <- rho * x_std + sqrt(1 - rho^2) * epsilon_std
#   
#   # 4. Scale y_std back to the desired mean and standard deviation of y
#   y_correlated <- y_std * sd_y + mean_y
#   
#   # tranform to 0-1 range
#   y_correlated <- pnorm(scale(y_correlated))
#   
#   return(y_correlated)
# }
# 
# 
# simulate_data <- function(n = 1000, rho  = 0, k = 0) {
#   
#   # Check that k is between 0 and 1
#   if (k < -1 || k > 1) {
#     stop("k must be between -1 and 1.")
#   }
#   
#   # Check rho is between -1 and 1
#   if (rho < -1 || rho > 1) {
#     stop("rho must be between -1 and 1.")
#   }
#   # n must be a positive integer
#   if (n <= 0 || n != as.integer(n)) {
#     stop("n must be a positive integer.")
#   }
#   
#   # generate uniformly random variable truncated to 0-1
#   x1 <- pnorm(rnorm(n))
#   
#   # generate group allocation, can be random (k = 0)
#   # or skewed (x1 will be transformed such that C group remains uniform
#   # but T group will be skewed to lower (k<0) or upper (k>0) part of x1 
#   # representing treatment selection bias along some gradient, e.g. previous TI
#   df <- generate_groups(x1, k)
#   
#   # Now, for the treatment group (T), assign treatment intensities
#   # that may correlate with pre-treatment intensity (set rho) or not (rho=0)
#   df$TI_redd[df$Group == "T"] <- 
#     generate_correlated_vector(
#       df$TI_pre[df$Group == "T"], rho = rho)
#   
#   # fill missing values with zero
#   df$TI_redd[df$Group == "C"] <- 0
#   
#   # add TI_pre and TI_redd to total TI
#   df$TI_tot <- df$TI_pre + df$TI_redd
#   
#   return(df)
# }
library(MASS)
library(rethinking) # Make sure rethinking is loaded

dgp = function(n = 1000, rho = 0, e_lin = 1, e_sq = 0, lf = 0) {
  
  # two covarying variables using mvnorm
  mu = c(0, 0)
  Sigma = matrix(c(1, rho, rho, 1), nrow = 2)
  X = mvrnorm(n, mu, Sigma)
  
  # scale to 01 using pnorm
  X = pnorm(X)
  
  # transform to dataframe
  X = as.data.frame(X)
  colnames(X) = c("TI_pre", "TI_redd")
  
  # random group assigment
  X$treat = sample(c(1,0), n, replace = TRUE, prob = c(0.5, 0.5))
  
  # set ti_redd to 0 where treat = 0
  X$TI_redd[X$treat == 0] = 0
  
  # Calculate total treatment intensity based on DGP definition
  X$TI_tot <- (X$TI_pre + X$TI_redd) # This exact calculation is crucial
  
  
  X$Y <- rnorm(n, 0, 0.1)  + (lf * X$treat * X$TI_pre) + (X$TI_redd * e_lin) + (X$TI_redd^2 * e_sq) # Adding linear and squared effects
  
  # return df
  return(X)
}

# simulate data
set.seed(123)
n = 10000
rho = -0.5
e_lin = 3
e_sq = -0.2
lf = 0.5 # linear latent effect (unobserved, related to treatment)

sd1 = dgp(n, rho, e_lin, e_sq, lf = lf)

ggplot2::ggplot(sd1, aes(x = TI_pre, fill = factor(treat))) +
  geom_density(alpha = 0.5) +
  labs(title = "Density plot of preTI across REDD",
       x = "preTI",
       fill = "REDD") +
  theme_minimal()


# plot of preTI vs TI_redd
ggplot2::ggplot(sd1, aes(x = TI_redd, y = TI_pre, color = factor(treat))) +
  geom_point(alpha = 0.5) +
  labs(title = "Scatter plot of preTI vs TI_redd",
       x = "D_redd",
       y = "D_pre",
       color = "REDD") +
  theme_minimal()

ggplot2::ggplot(sd1, aes(x = TI_tot, y = Y, color = factor(treat))) +
  geom_point(alpha = 0.5) +
  labs(title = "Scatter plot of preTI vs TI_redd",
       x = "D_redd",
       y = "Y",
       color = "REDD") +
  theme_minimal()

# Statistical model
m1 <- alist(
  # final outcome of interest: Y
  Y ~ dnorm(mu, sigma),
  mu <- c0 + c1 * TI_redd + c2 * TI_redd^2 + c3 * TI_pre * treat,
  
  # 1. Likelihood for TI_redd (treatment selection process)
  TI_redd ~ dnorm(mu_redd, sigma_redd), 
  mu_redd <- b0 + b1 * treat + b2 * TI_pre +  b3 * treat * TI_pre, # Simplified from original
  
  # 4. Priors for parameters of Y model
  c0 ~ dnorm(0, 1), 
  c1 ~ dnorm(0, 1),   # To recover e_lin
  c2 ~ dnorm(0, 1),   # To recover e_sq
  c3 ~ dnorm(0, 1),
  sigma ~ dexp(1),    # To recover 0.1
  
  # 4. Priors for parameters of TI_redd model (as per original m1)
  b0 ~ dnorm(0, 1),
  b1 ~ dnorm(0, 1),
  b2 ~ dnorm(0, 1),
  b3 ~ dnorm(0, 1),
  sigma_redd ~ dexp(1)
  
)

m_recover_k <- quap(
  m1,
  data = sd1, # Data already has TI_tot calculated according to DGP
  start = list(
    c0 = 0,    # Start close to DGP's intercept for Y (which is 0)
    c1 = 0,     # Start for e_lin
    c2 = 0,     # Start for e_sq
    c3 = 0, 
    b1 = 0,
    b2 = 0,
    b3 = 0,
    
    sigma = 1,  # Start for Y's standard deviation
    sigma_redd = 1
  )
)

# To check the recovered parameters:
precis(m_recover_k)


# test naive lm model
sd1$TI_redd2 <- sd1$TI_redd^2
m_naive1 <- lm(Y ~ TI_redd + TI_redd2, data = sd1)
m_naive2 <- lm(Y ~ TI_redd + TI_redd2 + TI_pre, data = sd1)
m_naive3 <- lm(Y ~ TI_redd + TI_redd2 + TI_pre + treat, data = sd1)
m_naive4 <- lm(Y ~ TI_redd + TI_redd2 + TI_pre * treat, data = sd1)


screenreg(list(m_naive1, m_naive2, m_naive3, m_naive4))
htmlreg(list(m_naive1, m_naive2, m_naive3, m_naive4), 
        file = "./out/tbl/naive_models.html")


### showcase with some plots







