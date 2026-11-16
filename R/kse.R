## KSE: kernel stepwise selection with REML and EBIC.
##
## Model:  y = b 1 + f + eps,  eps ~ N(0, s2 I),  f ~ GP(0, s2 * phi * K_S),
##   K_S(x, x') = prod_{m in S} { (1 - rho_m) + rho_m exp(-exp(u_m) d_m^2 / (2 s_m^2)) },
## s_m = per-coordinate median distance, rho_m in (0, 1] (gate), u_m (log inverse-bandwidth
## multiplier).  With gate = FALSE, rho_m = 1 (product Gaussian / ARD kernel).
## b and s2 are profiled out (REML), so the criterion only depends on (log phi, rho_S, u_S):
##
##   -2 l_R = (n - 1) log(r' A^{-1} r) + log det A + log(1' A^{-1} 1),   A = I + phi K_S,
##
## Selection: forward steps; at each step the score of every unselected coordinate at rho_m = 0
## (one GEMM with the Gaussian bank) ranks candidates, the `n_cand` best are refitted
## (new coordinate + log phi free), the best one is added and the full model is refitted.
## The models along the forward path are compared by EBIC.

## exp(-D~) for a coordinate: D~ = (x_i - x_j)^2 / (2 s^2)
kse_Dt <- function(x, z = x, s) outer(z, x, "-")^2 / (2 * s^2)

## parameter vector layout: c(logphi, rho_1..rho_k, u_1..u_k) for S = (S_1..S_k)
kse_unpack <- function(par, k, gate) {
  list(logphi = par[1],
       rho = if (gate) par[1 + seq_len(k)] else rep(1, k),
       u = par[1 + (gate) * k + seq_len(k)])
}

## REML value and gradient
kse_eval <- function(par, S, Dt, y, gate, need_grad = TRUE) {
  n <- length(y); k <- length(S)
  pp <- kse_unpack(par, k, gate)
  phi <- exp(pp$logphi)
  K <- matrix(1, n, n)
  E <- F <- vector("list", k)
  for (a in seq_len(k)) {
    E[[a]] <- exp(-exp(pp$u[a]) * Dt[[a]])
    F[[a]] <- (1 - pp$rho[a]) + pp$rho[a] * E[[a]]
    K <- K * F[[a]]
  }
  A <- phi * K
  diag(A) <- diag(A) + 1
  R <- chol(A)
  Ainv <- chol2inv(R)
  v <- rowSums(Ainv)                  # A^{-1} 1
  sv <- sum(v)
  b <- sum(v * y) / sv
  r <- y - b
  alpha <- drop(Ainv %*% r)
  q <- sum(r * alpha)
  val <- (n - 1) * log(q) + 2 * sum(log(diag(R))) + log(sv)
  out <- list(value = val, K = K, alpha = alpha, b = b, phi = phi, s2 = q / (n - 1))
  if (!need_grad) return(out)
  P <- Ainv - tcrossprod(v) / sv - (n - 1) / q * tcrossprod(alpha)
  PP <- phi * P                       # d(-2 l_R)/d theta = < PP, dK/d theta >
  Q <- K * PP
  g <- numeric(length(par))
  g[1] <- sum(Q)
  if (k > 0) {
    ## leave-one-out products prod_{b != a} F_b via prefix / suffix products
    pre <- vector("list", k + 1); suf <- vector("list", k + 1)
    pre[[1]] <- matrix(1, n, n); suf[[k + 1]] <- matrix(1, n, n)
    for (a in seq_len(k)) pre[[a + 1]] <- pre[[a]] * F[[a]]
    for (a in k:1) suf[[a]] <- suf[[a + 1]] * F[[a]]
    for (a in seq_len(k)) {
      W <- PP * pre[[a]] * suf[[a + 1]]
      if (gate) g[1 + a] <- sum(W * (E[[a]] - 1))
      g[1 + gate * k + a] <- -pp$rho[a] * exp(pp$u[a]) * sum(W * Dt[[a]] * E[[a]])
    }
  }
  if (!all(is.finite(g)) || !is.finite(val)) stop("non-finite REML")
  out$grad <- g
  out$Q <- Q
  out
}

kse_bounds <- function(k, gate) {
  list(lower = c(-10, if (gate) rep(1e-3, k), rep(-4, k)),
       upper = c(15, if (gate) rep(1, k), rep(4, k)))
}

## optimise the free entries of par (indices `free`), others fixed
kse_optim <- function(par, free, S, Dt, y, gate, maxit = 200) {
  bd <- kse_bounds(length(S), gate)
  cache <- new.env()
  f_full <- function(sub) {
    pp <- par; pp[free] <- sub
    key <- paste(sub, collapse = ",")
    if (!identical(cache$key, key)) {
      ev <- tryCatch(kse_eval(pp, S, Dt, y, gate), error = function(e) NULL)
      cache$key <- key; cache$ev <- ev
    }
    cache$ev
  }
  fn <- function(sub) { ev <- f_full(sub); if (is.null(ev)) 1e300 else ev$value }
  gr <- function(sub) { ev <- f_full(sub); if (is.null(ev)) rep(0, length(sub)) else ev$grad[free] }
  st <- pmin(pmax(par[free], bd$lower[free]), bd$upper[free])
  op <- optim(st, fn, gr, method = "L-BFGS-B", lower = bd$lower[free], upper = bd$upper[free],
              control = list(maxit = maxit, factr = 1e8))
  par[free] <- op$par
  list(par = par, value = op$value)
}

## insert a new coordinate into the parameter vector
kse_add_par <- function(par, k, gate, rho0 = 0.5, u0 = 0) {
  pp <- kse_unpack(par, k, gate)
  c(pp$logphi, if (gate) c(pp$rho, rho0), c(pp$u, u0))
}

## forward path.  Returns the list of models visited (S, par, value).
## pair step (pairs = TRUE): when the best single coordinate does not pass the EBIC(xi_stop)
## penalty, coordinates are ranked by the variance-heterogeneity statistic v' G_m v,
## v = e^2 - mean(e^2) (e = current residual), and all pairs among the top `n_pair` are fitted
## jointly; the best pair is added if it improves -2 l_R by more than two penalties.
kse_path <- function(X, y, gate = TRUE, Gm = NULL, sig = NULL, n_cand = 5, max_size = 15,
                     xi_stop = 0, patience = 2, pairs = TRUE, n_pair = 8, verbose = FALSE) {
  n <- nrow(X); p <- ncol(X)
  if (is.null(sig)) sig <- tpk_bandwidth(X, "coord")
  if (is.null(Gm)) Gm <- tpk_bank(X, sig)
  kpen <- (1 + gate) * log(n) + 2 * xi_stop * log(p)

  S <- integer(0); Dt <- list(); par <- 0
  cur <- kse_eval(par, S, Dt, y, gate)
  path <- list(list(S = S, par = par, value = cur$value))
  best_crit <- cur$value; bad <- 0
  while (length(S) < max_size) {
    ## score test for all unselected coordinates (rho_m = 0, u_m = 0)
    sc <- drop(crossprod(Gm, as.vector(cur$Q))) - sum(cur$Q)
    sc[S] <- Inf
    cand <- order(sc)[seq_len(min(n_cand, p - length(S)))]
    k <- length(S)
    best <- NULL
    for (m in cand) {
      S2 <- c(S, m)
      Dt2 <- c(Dt, list(kse_Dt(X[, m], s = sig[m])))
      par2 <- kse_add_par(par, k, gate)
      free <- c(1, if (gate) 1 + k + 1, 1 + gate * (k + 1) + k + 1)
      fit <- tryCatch(kse_optim(par2, free, S2, Dt2, y, gate, maxit = 60),
                      error = function(e) NULL)
      if (!is.null(fit) && (is.null(best) || fit$value < best$value))
        best <- list(m = m, par = fit$par, value = fit$value, Dt = Dt2)
    }
    if (is.null(best)) break
    if (pairs && cur$value - best$value < kpen && length(S) + 2 <= max_size) {
      e <- cur$alpha
      v <- e^2 - mean(e^2)
      hs <- drop(crossprod(Gm, as.vector(tcrossprod(v))))
      hs[S] <- -Inf
      top <- order(-hs)[seq_len(min(n_pair, p - k))]
      bp <- NULL
      for (i1 in seq_along(top)) for (i2 in seq_along(top)) {
        if (i2 <= i1) next
        m1 <- top[i1]; m2 <- top[i2]
        S2 <- c(S, m1, m2)
        Dt2 <- c(Dt, list(kse_Dt(X[, m1], s = sig[m1]), kse_Dt(X[, m2], s = sig[m2])))
        par2 <- kse_add_par(kse_add_par(par, k, gate), k + 1, gate)
        free <- c(1, if (gate) 1 + k + 1:2, 1 + gate * (k + 2) + k + 1:2)
        fit <- tryCatch(kse_optim(par2, free, S2, Dt2, y, gate, maxit = 60), error = function(e) NULL)
        if (!is.null(fit) && (is.null(bp) || fit$value < bp$value))
          bp <- list(m = c(m1, m2), par = fit$par, value = fit$value, Dt = Dt2)
      }
      if (!is.null(bp) && cur$value - bp$value > 2 * kpen && bp$value < best$value - kpen) {
        best <- bp
        if (verbose) cat(sprintf("  pair step: %s\n", paste(bp$m, collapse = ",")))
      }
    }
    S <- c(S, best$m); Dt <- best$Dt
    full <- kse_optim(best$par, seq_along(best$par), S, Dt, y, gate)
    par <- full$par
    cur <- kse_eval(par, S, Dt, y, gate)
    path[[length(path) + 1]] <- list(S = S, par = par, value = cur$value)
    crit <- cur$value + length(S) * kpen
    if (verbose) cat(sprintf("  add %d: -2lR=%.2f crit=%.2f\n", best$m, cur$value, crit))
    if (crit < best_crit - 1e-8) { best_crit <- crit; bad <- 0 } else {
      bad <- bad + 1
      if (bad >= patience) break
    }
  }
  list(path = path, sig = sig, gate = gate, n = n, p = p)
}

## choose a model on the path by EBIC with parameter xi
kse_select <- function(kp, xi) {
  kpen <- (1 + kp$gate) * log(kp$n) + 2 * xi * log(kp$p)
  crit <- vapply(kp$path, function(m) m$value + length(m$S) * kpen, 0)
  kp$path[[which.min(crit)]]
}

## predictor for a selected model
kse_predictor <- function(X, y, model, sig, gate) {
  S <- model$S
  if (length(S) == 0) { b <- mean(y); return(function(Z) rep(b, nrow(Z))) }
  Dt <- lapply(S, function(m) kse_Dt(X[, m], s = sig[m]))
  ev <- kse_eval(model$par, S, Dt, y, gate, need_grad = FALSE)
  pp <- kse_unpack(model$par, length(S), gate)
  function(Z) {
    Kz <- matrix(1, nrow(Z), nrow(X))
    for (a in seq_along(S)) {
      E <- exp(-exp(pp$u[a]) * kse_Dt(X[, S[a]], Z[, S[a]], sig[S[a]]))
      Kz <- Kz * ((1 - pp$rho[a]) + pp$rho[a] * E)
    }
    ev$b + ev$phi * drop(Kz %*% ev$alpha)
  }
}

## drop coordinate a from the parameter vector
kse_drop_par <- function(par, k, gate, a) {
  pp <- kse_unpack(par, k, gate)
  c(pp$logphi, if (gate) pp$rho[-a], pp$u[-a])
}

## backward elimination from `model` under EBIC(xi): repeatedly refit without each
## coordinate and drop the one giving the lowest criterion while it improves.
kse_backward <- function(X, y, model, sig, gate, xi) {
  n <- nrow(X); p <- ncol(X)
  kpen <- (1 + gate) * log(n) + 2 * xi * log(p)
  S <- model$S; par <- model$par
  crit <- model$value + length(S) * kpen
  repeat {
    if (length(S) == 0) break
    best <- NULL
    for (a in seq_along(S)) {
      S2 <- S[-a]
      par2 <- kse_drop_par(par, length(S), gate, a)
      if (length(S2) == 0) {
        val <- kse_eval(par2, S2, list(), y, gate, need_grad = FALSE)$value
        fit <- list(par = par2, value = val)
      } else {
        Dt2 <- lapply(S2, function(m) kse_Dt(X[, m], s = sig[m]))
        fit <- tryCatch(kse_optim(par2, seq_along(par2), S2, Dt2, y, gate), error = function(e) NULL)
      }
      if (!is.null(fit)) {
        c2 <- fit$value + length(S2) * kpen
        if (is.null(best) || c2 < best$crit) best <- list(S = S2, par = fit$par, value = fit$value, crit = c2)
      }
    }
    if (is.null(best) || best$crit >= crit - 1e-8) break
    S <- best$S; par <- best$par; crit <- best$crit; model <- best
  }
  list(S = S, par = par, value = model$value)
}

## full procedure: forward path, EBIC choice, backward elimination
kse_fit <- function(X, y, gate = TRUE, xi = c(0.5, 1), Gm = NULL, backward = TRUE, ...) {
  sig <- tpk_bandwidth(X, "coord")
  if (is.null(Gm)) Gm <- tpk_bank(X, sig)
  kp <- kse_path(X, y, gate = gate, Gm = Gm, sig = sig, xi_stop = min(xi), ...)
  lapply(setNames(xi, paste0("xi", xi)), function(x) {
    md <- kse_select(kp, x)
    if (backward && length(md$S) > 1) md <- kse_backward(X, y, md, sig, gate, x)
    md$predict <- kse_predictor(X, y, md, sig, gate)
    md
  })
}
