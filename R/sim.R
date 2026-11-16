## Simulation designs of Section 3 of Yu, Wang, Zeng (2023). X1..X5 are columns 1..5.

corr_block <- function(entries) {
  R <- diag(4)
  for (e in entries) { R[e[1], e[2]] <- e[3]; R[e[2], e[1]] <- e[3] }
  R
}
R_REG <- corr_block(list(c(1, 2, 0.4), c(1, 3, -0.3), c(2, 3, 0.5), c(3, 4, 0.2)))
R_CLS <- corr_block(list(c(1, 2, -0.2), c(1, 4, 0.2), c(2, 3, 0.5), c(2, 4, 0.3), c(3, 4, -0.4)))

draw_x <- function(n, p, R) {
  X <- matrix(rnorm(n * p), n, p)
  X[, 1:4] <- X[, 1:4] %*% chol(R)
  X
}

## Y = 0.9 X5^3 + 4 X1 X2 X3 + 2.3 exp(-X3) + 4 X4 + eps
sim_reg <- function(n, p) {
  X <- draw_x(n, p, R_REG)
  f <- 0.9 * X[, 5]^3 + 4 * X[, 1] * X[, 2] * X[, 3] + 2.3 * exp(-X[, 3]) + 4 * X[, 4]
  list(X = X, y = f + rnorm(n), f = f)
}

## P(Y = 1 | X) = 1 / (1 + exp{-0.25 + (X2 - 1.1 X3 + 0.3 X4)^3})
sim_cls <- function(n, p) {
  X <- draw_x(n, p, R_CLS)
  eta <- -0.25 + (X[, 2] - 1.1 * X[, 3] + 0.3 * X[, 4])^3
  prob <- 1 / (1 + exp(pmin(pmax(eta, -700), 700)))
  list(X = X, y = ifelse(runif(n) < prob, 1, -1), f = prob)
}

## pure pairwise interaction (no marginal signal for X1, X2), independent X:
## Y = 3 X1 X2 + 2 sin(1.5 X3) + eps
sim_int <- function(n, p) {
  X <- matrix(rnorm(n * p), n, p)
  f <- 3 * X[, 1] * X[, 2] + 2 * sin(1.5 * X[, 3])
  list(X = X, y = f + rnorm(n), f = f)
}

DESIGNS <- list(
  reg = list(sim = sim_reg, truth = 1:5, loss = "l2"),
  cls = list(sim = sim_cls, truth = 2:4, loss = "exp"),
  int = list(sim = sim_int, truth = 1:3, loss = "l2")
)
