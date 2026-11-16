## Tensor-product-kernel feature selection (base R only).
##
## Kernel = prod_m F_m(t_m), F_m(t) = 1 + t * H_m (affine in t):
##   kernel = "orig" : F_m = 1 + lambda_m G_m,           H_m = G_m,     t in [0, M]
##   kernel = "gated": F_m = (1 - rho_m) + rho_m G_m,    H_m = G_m - 1, t in [0, 1]
## G_m = per-coordinate Gaussian matrix exp{-(x_im - x_jm)^2 / (2 sigma_m^2)}.
##
## Objective:
##   (1/n) sum_j w_j l(y_j, b + (K_t alpha)_j) + gamma1 alpha' K_t alpha + gamma2 sum_m t_m
## loss = "l2" (squared error) or "exp" (weighted exponential loss, y in {-1, 1}).
## See docs/derivation.md for the coordinate-wise formulas used below.

EXP_CLIP <- 500

## ---------------------------------------------------------------------------
## bandwidths and the Gaussian bank
## ---------------------------------------------------------------------------

## rule = "global": sigma^2 = median of full p-dim squared distances over all n^2 pairs
##                  (published MATLAB code), common to all coordinates.
## rule = "coord" : sigma_m = median_{i<j} |x_im - x_jm|.
tpk_bandwidth <- function(X, rule = c("coord", "global")) {
  rule <- match.arg(rule)
  n <- nrow(X); p <- ncol(X)
  if (rule == "global") {
    sq <- rowSums(X * X)
    Q <- outer(sq, sq, "+") - 2 * tcrossprod(X)
    return(rep(sqrt(median(Q)), p))
  }
  lt <- lower.tri(diag(n))
  ii <- row(lt)[lt]; jj <- col(lt)[lt]
  s <- vapply(seq_len(p), function(m) median(abs(X[ii, m] - X[jj, m])), 0)
  pmax(s, 1e-12)
}

## n^2 x p matrix; column m is vec(G_m) (column-major, entry (i, j) at i + n (j - 1)).
tpk_bank <- function(X, sigma) {
  n <- nrow(X); p <- ncol(X)
  Gm <- matrix(0, n * n, p)
  for (m in seq_len(p)) {
    x <- X[, m]
    d <- rep(x, times = n) - rep(x, each = n)
    Gm[, m] <- exp(-d * d / (2 * sigma[m]^2))
  }
  Gm
}

## cross kernel factor between rows of Z and rows of X for coordinate m
tpk_cross_G <- function(z, x, s) exp(-outer(z, x, "-")^2 / (2 * s^2))

## ---------------------------------------------------------------------------
## one-dimensional coordinate subproblem
##   min_{lo <= t <= hi} (1/n) sum_j w_j l(y_j, a_j + t B_j) + d t,  d = gamma1 c + gamma2
## ---------------------------------------------------------------------------

coord_solve <- function(loss, y, w, a, B, d, lo, hi) {
  n <- length(y)
  if (loss == "l2") {
    r <- y - a
    BB <- sum(B * B)
    num <- sum(r * B) - n * d / 2
    if (BB <= 0) return(if (num > 0) hi else lo)
    return(min(max(num / BB, lo), hi))
  }
  dh <- function(t) {
    e <- exp(pmin(-y * (a + t * B), EXP_CLIP))
    -sum(w * y * B * e) / n + d
  }
  if (dh(lo) >= 0) return(lo)
  if (dh(hi) <= 0) return(hi)
  l <- lo; u <- hi
  ## safeguarded Newton
  tt <- lo
  for (k in 1:100) {
    e <- w * exp(pmin(-y * (a + tt * B), EXP_CLIP))
    g <- -sum(y * B * e) / n + d
    if (g > 0) u <- tt else l <- tt
    h2 <- sum(B * B * e) / n
    tn <- if (h2 > 0) tt - g / h2 else NA
    if (!is.finite(tn) || tn <= l || tn >= u) tn <- 0.5 * (l + u)
    if (abs(tn - tt) <= 1e-10 * max(1, abs(tt)) || (u - l) <= 1e-12 * max(1, u)) { tt <- tn; break }
    tt <- tn
  }
  tt
}

## ---------------------------------------------------------------------------
## alpha step
## ---------------------------------------------------------------------------

## argmin (1/n)||y - b - K alpha||^2 + gamma1 alpha' K alpha   (b = 0 if !intercept)
alpha_l2 <- function(K, y, gamma1, intercept, jitter) {
  n <- length(y)
  A <- K
  diag(A) <- diag(A) + n * gamma1 + jitter
  R <- tryCatch(chol(A), error = function(e) NULL)
  slv <- if (is.null(R)) function(B) solve(A, B) else function(B) backsolve(R, forwardsolve(t(R), B))
  if (!intercept) return(list(alpha = drop(slv(y)), b = 0))
  S <- slv(cbind(y, 1))
  b <- sum(S[, 1]) / sum(S[, 2])
  list(alpha = S[, 1] - b * S[, 2], b = b)
}

## solve(A, rhs); on failure add an increasing ridge to the diagonal
robust_solve <- function(A, rhs) {
  out <- tryCatch(solve(A, rhs), error = function(e) NULL)
  eps <- 1e-10 * max(1, max(abs(diag(A))))
  while (is.null(out) || any(!is.finite(out))) {
    out <- tryCatch(solve(A + diag(eps, nrow(A)), rhs), error = function(e) NULL)
    eps <- eps * 100
    if (eps > 1e10) { out <- numeric(length(rhs)); break }
  }
  out
}

exp_obj <- function(K, y, w, alpha, b, gamma1) {
  Ka <- drop(K %*% alpha)
  mean(w * exp(pmin(-y * (Ka + b), EXP_CLIP))) + gamma1 * sum(alpha * Ka)
}

## damped Newton steps in (alpha, b) for the weighted exponential loss
alpha_exp <- function(K, y, w, alpha, b, gamma1, intercept, steps, jitter, tol = 1e-10) {
  n <- length(y)
  obj <- exp_obj(K, y, w, alpha, b, gamma1)
  obj0 <- exp_obj(K, y, w, numeric(n), 0, gamma1)
  if (!is.finite(obj) || obj > obj0) { alpha <- numeric(n); b <- 0; obj <- obj0 }
  for (s in seq_len(steps)) {
    f <- drop(K %*% alpha) + b
    u <- w * exp(pmin(-y * f, EXP_CLIP))
    g <- -(y * u) / n + 2 * gamma1 * alpha
    A <- (u / n) * K
    diag(A) <- diag(A) + 2 * gamma1 + jitter
    if (intercept) {
      A <- rbind(cbind(A, u / n), c(drop(u %*% K) / n, sum(u) / n))
      rhs <- c(-g, sum(y * u) / n)
    } else rhs <- -g
    del <- robust_solve(A, rhs)
    da <- del[1:n]; db <- if (intercept) del[n + 1] else 0
    step <- 1; moved <- FALSE
    for (k in 1:40) {
      na <- alpha + step * da; nb <- b + step * db
      no <- exp_obj(K, y, w, na, nb, gamma1)
      if (is.finite(no) && no <= obj) { moved <- TRUE; break }
      step <- step / 2
    }
    if (!moved) break
    dec <- obj - no
    alpha <- na; b <- nb; obj <- no
    if (dec <= tol * max(1, abs(obj))) break
  }
  list(alpha = alpha, b = b)
}

## ---------------------------------------------------------------------------
## main fit
## ---------------------------------------------------------------------------

## Gm     : bank from tpk_bank(X, sigma)
## screen : "kkt"     -> sweeps over the active set; at convergence all zero coordinates are
##                       screened (KKT) and violators are swept; stop when there is no violator
##          "forever" -> a coordinate that is zero after a sweep is never revisited
##                       (published MATLAB code)
## init   : optional list(t, alpha, b) for warm starts
## dfmax  : stop early (flag $aborted) when more than dfmax coordinates are active
tpk_fit <- function(X, y, Gm, sigma, kernel = c("gated", "orig"), loss = c("l2", "exp"),
                    gamma1, gamma2, M = 1e5, intercept = (kernel == "gated"),
                    screen = if (kernel == "gated") "kkt" else "forever",
                    max_iter = 500, tol_obj = 1e-3, tol_t = 1e-3, rel_tol = (kernel == "gated"),
                    newton_steps = 1, jitter = if (kernel == "orig") 1e-4 else 1e-8,
                    init = NULL, dfmax = Inf, order = NULL) {
  kernel <- match.arg(kernel); loss <- match.arg(loss)
  n <- nrow(X); p <- ncol(X)
  gated <- kernel == "gated"
  hi <- if (gated) 1 else M

  ymean <- 0
  if (loss == "l2") {
    w <- rep(1, n)
    if (!intercept) { ymean <- mean(y); y <- y - ymean }
  } else {
    npos <- sum(y > 0); nneg <- sum(y < 0)
    w <- ifelse(y > 0, n / (2 * max(npos, 1)), n / (2 * max(nneg, 1)))
  }
  if (is.null(order)) {
    cc <- abs(suppressWarnings(cor(X, y))); cc[is.na(cc)] <- 0
    order <- order(-cc)
  }
  rank <- integer(p); rank[order] <- seq_len(p)

  Gcache <- list()
  getG <- function(q) {
    key <- as.character(q)
    G <- Gcache[[key]]
    if (is.null(G)) { G <- matrix(Gm[, q], n, n); Gcache[[key]] <<- G }
    G
  }
  Fq <- function(q, tq) { G <- getG(q); if (gated) (1 - tq) + tq * G else 1 + tq * G }
  build_K <- function(t) {
    K <- matrix(1, n, n)
    for (q in which(t != 0)) K <- K * Fq(q, t[q])
    K
  }
  objective <- function(K, alpha, b, t) {
    Ka <- drop(K %*% alpha)
    f <- Ka + b
    fit <- if (loss == "l2") mean((y - f)^2) else mean(w * exp(pmin(-y * f, EXP_CLIP)))
    fit + gamma1 * sum(alpha * Ka) + gamma2 * sum(t)
  }
  alpha_step <- function(K, alpha, b, steps) {
    if (loss == "l2") alpha_l2(K, y, gamma1, intercept, jitter)
    else alpha_exp(K, y, w, alpha, b, gamma1, intercept, steps, jitter)
  }
  ## pseudo-residual z: h'(0) = -(1/n) z'b_q + gamma1 c_q + gamma2
  zvec <- function(f) if (loss == "l2") 2 * (y - f) else w * y * exp(pmin(-y * f, EXP_CLIP))

  ## KKT screening of all zero coordinates with one GEMM; returns violators
  screen_all <- function(K, alpha, b, t) {
    Ka <- drop(K %*% alpha)
    z <- zvec(Ka + b)
    KA <- K * rep(alpha, each = n)                 # K_ij alpha_j
    S <- crossprod(Gm, cbind(as.vector(z * KA), as.vector(alpha * KA)))
    zb <- S[, 1]; cq <- S[, 2]
    if (gated) { zb <- zb - sum(z * Ka); cq <- cq - sum(alpha * Ka) }
    viol <- which(t == 0 & zb > n * (gamma1 * cq + gamma2) * (1 + 1e-10))
    viol[order(rank[viol])]
  }

  ## exact sequential coordinate updates over set S (in `order`)
  sweep <- function(S, t, K, alpha, b) {
    for (q in S) {
      G <- getG(q)
      tq <- t[q]
      W <- if (tq == 0) K else K / (if (gated) (1 - tq) + tq * G else 1 + tq * G)
      u <- drop(W %*% alpha)
      v <- drop((W * G) %*% alpha)
      B <- if (gated) v - u else v
      cq <- sum(alpha * B)
      tn <- coord_solve(loss, y, w, b + u, B, gamma1 * cq + gamma2, 0, hi)
      t[q] <- tn
      K <- if (tn == 0) W else W * (if (gated) (1 - tn) + tn * G else 1 + tn * G)
    }
    list(t = t, K = K)
  }

  ## initial values
  if (!is.null(init)) {
    t <- init$t; alpha <- init$alpha; b <- init$b
  } else {
    t <- numeric(p); alpha <- numeric(n); b <- 0
    if (loss == "exp") t[order[1]] <- if (gated) 0.5 else 1
  }
  cand_alive <- rep(TRUE, p)          # "forever": coordinates still eligible
  need_screen <- TRUE
  obj_prev <- Inf; t_prev <- t
  converged <- FALSE; aborted <- FALSE; it <- 0
  K <- build_K(t)
  while (it < max_iter) {
    it <- it + 1
    K <- build_K(t)
    ab <- alpha_step(K, alpha, b, newton_steps); alpha <- ab$alpha; b <- ab$b
    act <- which(t != 0)
    if (screen == "forever") {
      if (!any(cand_alive)) { converged <- TRUE; break }
      V <- if (need_screen) { vv <- screen_all(K, alpha, b, t); vv[cand_alive[vv]] } else integer(0)
      S <- union(act, V); S <- S[order(rank[S])]
      did_screen <- need_screen; need_screen <- FALSE
    } else {
      did_screen <- need_screen
      V <- if (need_screen) screen_all(K, alpha, b, t) else integer(0)
      S <- union(act, V); S <- S[order(rank[S])]
      need_screen <- FALSE
    }
    sw <- sweep(S, t, K, alpha, b); t <- sw$t; K <- sw$K
    if (sum(t != 0) > dfmax) { aborted <- TRUE; break }
    obj <- objective(K, alpha, b, t)
    dobj <- abs(obj - obj_prev); if (rel_tol) dobj <- dobj / max(1, abs(obj))
    small <- is.finite(dobj) && dobj < tol_obj && sum(abs(t - t_prev)) < tol_t
    obj_prev <- obj; t_prev <- t
    if (screen == "forever") {
      if (did_screen) cand_alive[] <- FALSE
      cand_alive[t != 0] <- TRUE
      cand_alive[t == 0] <- FALSE
      if (small) { converged <- TRUE; break }
    } else {
      if (did_screen && length(V) == 0 && small) { converged <- TRUE; break }
      if (small) need_screen <- TRUE
    }
  }
  ## final alpha for the selected kernel
  K <- build_K(t)
  if (loss == "l2") {
    ab <- alpha_l2(K, y, gamma1, intercept, jitter)
  } else ab <- alpha_exp(K, y, w, alpha, b, gamma1, intercept, 100, jitter)
  structure(list(kernel = kernel, loss = loss, t = t, alpha = ab$alpha, b = ab$b, ymean = ymean,
                 X = X, sigma = sigma, support = which(t != 0), n_iter = it,
                 converged = converged, aborted = aborted,
                 objective = objective(K, ab$alpha, ab$b, t),
                 log10_maxK = log10(max(abs(K))),
                 gamma1 = gamma1, gamma2 = gamma2),
            class = "tpk")
}

## ---------------------------------------------------------------------------
## profiled solver: alpha (and b) are minimised out, the reduced objective
##   phi(t) = min_{alpha, b} L(alpha, b, t) + gamma2 sum t
## is smooth on the box [0, hi]^p.  By the envelope theorem
##   d phi / d t_q = < Omega, H_q / F_q > + gamma2,
##   Omega = K o (gamma1 alpha alpha' - z alpha' / n),   z = -n dLoss/df,
## so the gradient of every zero coordinate (F_q = 1) is one GEMM with the bank.
## Working-set scheme: L-BFGS-B on the active set, then KKT screening of all zero
## coordinates; the `add_max` most violating ones enter the working set.
## ---------------------------------------------------------------------------

tpk_fit_lbfgs <- function(X, y, Gm, sigma, kernel = c("gated", "orig"), loss = c("l2", "exp"),
                          gamma1, gamma2, M = 1e5, intercept = (kernel == "gated"),
                          jitter = if (kernel == "orig") 1e-4 else 1e-8,
                          init = NULL, dfmax = Inf, order = NULL, add_max = 10,
                          max_rounds = 50, kkt_tol = 1e-6, factr = 1e7, maxit = 500) {
  kernel <- match.arg(kernel); loss <- match.arg(loss)
  n <- nrow(X); p <- ncol(X)
  gated <- kernel == "gated"
  hi <- if (gated) 1 else M
  tzero <- if (gated) 1e-8 else 1e-6

  ymean <- 0
  if (loss == "l2") {
    w <- rep(1, n)
    if (!intercept) { ymean <- mean(y); y <- y - ymean }
  } else {
    npos <- sum(y > 0); nneg <- sum(y < 0)
    w <- ifelse(y > 0, n / (2 * max(npos, 1)), n / (2 * max(nneg, 1)))
  }
  if (is.null(order)) {
    cc <- abs(suppressWarnings(cor(X, y))); cc[is.na(cc)] <- 0
    order <- order(-cc)
  }

  Gcache <- list()
  getG <- function(q) {
    key <- as.character(q)
    G <- Gcache[[key]]
    if (is.null(G)) { G <- matrix(Gm[, q], n, n); Gcache[[key]] <<- G }
    G
  }
  Ffac <- function(q, tq) { G <- getG(q); if (gated) (1 - tq) + tq * G else 1 + tq * G }

  ## state at the last evaluated point
  st <- new.env()
  st$alpha <- if (is.null(init)) numeric(n) else init$alpha
  st$b <- if (is.null(init)) 0 else init$b
  st$key <- NULL
  nfeval <- 0

  evaluate <- function(A, tA) {
    key <- c(A, tA)
    if (!is.null(st$key) && length(st$key) == length(key) && all(st$key == key)) return(invisible())
    nfeval <<- nfeval + 1
    K <- matrix(1, n, n)
    for (k in seq_along(A)) if (tA[k] != 0) K <- K * Ffac(A[k], tA[k])
    if (loss == "l2") {
      ab <- alpha_l2(K, y, gamma1, intercept, jitter)
    } else {
      ab <- alpha_exp(K, y, w, st$alpha, st$b, gamma1, intercept, 100, jitter, tol = 1e-13)
    }
    Ka <- drop(K %*% ab$alpha)
    f <- Ka + ab$b
    if (loss == "l2") {
      fit <- mean((y - f)^2); z <- 2 * (y - f)
    } else {
      e <- w * exp(pmin(-y * f, EXP_CLIP)); fit <- mean(e); z <- y * e
    }
    st$K <- K; st$alpha <- ab$alpha; st$b <- ab$b
    st$val <- fit + gamma1 * sum(ab$alpha * Ka) + gamma2 * sum(tA)
    st$Omega <- K * outer(gamma1 * ab$alpha - z / n, ab$alpha)
    st$Omega[!is.finite(st$Omega)] <- 0
    st$key <- key
  }
  grad_active <- function(A, tA) {
    vapply(seq_along(A), function(k) {
      G <- getG(A[k])
      H <- if (gated) G - 1 else G
      sum(st$Omega * H / (1 + tA[k] * H))
    }, 0) + gamma2
  }
  grad_zero <- function() {
    g <- drop(crossprod(Gm, as.vector(st$Omega)))
    if (gated) g <- g - sum(st$Omega)
    g + gamma2
  }

  t <- if (is.null(init)) numeric(p) else init$t
  if (is.null(init) && loss == "exp") t[order[1]] <- if (gated) 0.5 else 1
  A <- which(t != 0)
  converged <- FALSE; aborted <- FALSE; rounds <- 0
  repeat {
    rounds <- rounds + 1
    evaluate(A, t[A])
    g <- grad_zero()
    gz <- g; gz[t != 0] <- Inf
    viol <- which(gz < -kkt_tol * max(1, abs(st$val)))
    if (length(viol) == 0 && rounds > 1) { converged <- TRUE; break }
    if (rounds > max_rounds) break
    if (length(viol) > add_max) viol <- viol[order(gz[viol])[seq_len(add_max)]]
    A <- sort(union(A, viol))
    if (length(A) == 0) { converged <- TRUE; break }
    fn <- function(tA) { evaluate(A, tA); if (is.finite(st$val)) st$val else 1e300 }
    gr <- function(tA) {
      evaluate(A, tA); g <- grad_active(A, tA); g[!is.finite(g)] <- 0; g
    }
    op <- tryCatch(optim(t[A], fn, gr, method = "L-BFGS-B", lower = 0, upper = hi,
                         control = list(factr = factr, maxit = maxit)),
                   error = function(e) list(par = t[A]))
    tA <- op$par
    tA[tA < tzero] <- 0
    t[A] <- tA
    A <- which(t != 0)
    if (length(A) > dfmax) { aborted <- TRUE; break }
  }
  evaluate(A, t[A])
  structure(list(kernel = kernel, loss = loss, t = t, alpha = st$alpha, b = st$b, ymean = ymean,
                 X = X, sigma = sigma, support = which(t != 0), n_iter = nfeval, rounds = rounds,
                 converged = converged, aborted = aborted, objective = st$val,
                 log10_maxK = log10(max(abs(st$K))), gamma1 = gamma1, gamma2 = gamma2),
            class = "tpk")
}

predict.tpk <- function(object, Z, type = c("link", "class"), ...) {
  type <- match.arg(type)
  Kz <- matrix(1, nrow(Z), nrow(object$X))
  for (m in object$support) {
    G <- tpk_cross_G(Z[, m], object$X[, m], object$sigma[m])
    tm <- object$t[m]
    Kz <- Kz * (if (object$kernel == "gated") (1 - tm) + tm * G else 1 + tm * G)
  }
  f <- drop(Kz %*% object$alpha) + object$b + object$ymean
  if (type == "class") ifelse(f >= 0, 1, -1) else f
}

## validation loss used for CV
## classification criterion: getOption("tpk.cls_cv", "misclass") = "misclass" or "exp"
## ("exp": class-balanced exponential loss on the validation fold)
cv_loss <- function(loss, y, f) {
  if (loss == "l2") return(mean((y - f)^2))
  if (getOption("tpk.cls_cv", "misclass") == "exp") {
    npos <- sum(y > 0); nneg <- sum(y < 0); n <- length(y)
    w <- ifelse(y > 0, n / (2 * max(npos, 1)), n / (2 * max(nneg, 1)))
    return(mean(w * exp(pmin(-y * f, EXP_CLIP))))
  }
  mean(ifelse(f >= 0, 1, -1) != y)
}

## ---------------------------------------------------------------------------
## cross-validation over a (gamma1, gamma2) grid
## ---------------------------------------------------------------------------

## For each fold the Gaussian bank is built once and reused across the whole grid.
## warm = TRUE: for each gamma1, gamma2 runs from largest to smallest with warm starts;
##              once a fit exceeds dfmax the remaining (smaller) gamma2 are skipped.
## relax = TRUE additionally records, for every grid point and fold, the validation error of
## the relaxed fit: the selected coordinates refitted with rho = 1 (product Gaussian kernel,
## intercept) at the same gamma1.
tpk_cv <- function(X, y, kernel, loss, bw, folds, grid1, grid2, warm, dfmax,
                   solver = c("lbfgs", "cd"), relax = FALSE, relax_scales = c(1, 2, 4), ...) {
  solver <- match.arg(solver)
  fitfun <- if (solver == "lbfgs") tpk_fit_lbfgs else tpk_fit
  E <- array(NA_real_, c(length(grid1), length(grid2), max(folds)))
  NS <- E                                  # number of selected coordinates per fold
  ER <- array(NA_real_, c(length(grid1), length(grid2), max(folds), length(grid1),
                         length(relax_scales)))
  g2_desc <- order(-grid2)
  for (k in seq_len(max(folds))) {
    tr <- folds != k
    Xtr <- X[tr, , drop = FALSE]; ytr <- y[tr]
    sig <- tpk_bandwidth(Xtr, bw)
    Gm <- tpk_bank(Xtr, sig)
    cc <- abs(suppressWarnings(cor(Xtr, ytr))); cc[is.na(cc)] <- 0
    ord <- order(-cc)
    rcache <- list()
    for (i in seq_along(grid1)) {
      init <- NULL; dead <- FALSE
      for (j in g2_desc) {
        if (dead) { E[i, j, k] <- NS[i, j, k] <- Inf; ER[i, j, k, , ] <- Inf; next }
        fit <- tryCatch(fitfun(Xtr, ytr, Gm, sig, kernel = kernel, loss = loss,
                               gamma1 = grid1[i], gamma2 = grid2[j], init = init,
                               dfmax = dfmax, order = ord, ...),
                        error = function(e) list(aborted = TRUE))
        if (fit$aborted) { E[i, j, k] <- NS[i, j, k] <- Inf; ER[i, j, k, , ] <- Inf; dead <- TRUE; next }
        e <- cv_loss(loss, y[!tr], predict(fit, X[!tr, , drop = FALSE]))
        E[i, j, k] <- if (is.finite(e)) e else Inf
        NS[i, j, k] <- length(fit$support)
        if (relax) {
          rkey <- paste(fit$support, collapse = ",")
          er <- rcache[[rkey]]
          if (is.null(er)) {
            er <- relaxed_errors(Xtr, ytr, X[!tr, , drop = FALSE], y[!tr], fit$support, Gm, sig,
                                 loss, grid1, relax_scales)
            rcache[[rkey]] <- er
          }
          ER[i, j, k, , ] <- er
        }
        if (warm) init <- list(t = fit$t, alpha = fit$alpha, b = fit$b)
      }
    }
    rm(Gm); gc(verbose = FALSE)
  }
  out <- cv_select(E, NS, grid1, grid2)
  if (relax) {
    ## joint choice of (selection gamma1, gamma2, refit gamma1); ties -> larger gamma2,
    ## then larger selection gamma1, then larger refit gamma1
    cvr <- apply(ER, c(1, 2, 4, 5), mean)
    best <- which(cvr == min(cvr), arr.ind = TRUE)
    best <- best[order(-grid2[best[, 2]], -grid1[best[, 1]], -grid1[best[, 3]]), , drop = FALSE][1, ]
    out$cvm_relax <- apply(cvr, c(1, 2), min)
    out$gamma1_relax <- grid1[best[1]]; out$gamma2_relax <- grid2[best[2]]
    out$gamma1_refit_relax <- grid1[best[3]]; out$scale_relax <- relax_scales[best[4]]
  }
  out
}

## validation errors of the relaxed fit on support S for every refit penalty in grid_r:
## product Gaussian kernel on S (rho = 1), intercept, training-fold bank Gm and bandwidths sig.
relaxed_errors <- function(Xtr, ytr, Xva, yva, S, Gm, sig, loss, grid_r, scales) {
  nr <- length(grid_r); ns <- length(scales)
  if (length(S) == 0) {
    f <- if (loss == "l2") rep(mean(ytr), length(yva)) else rep(if (mean(ytr > 0) >= 0.5) 1 else -1, length(yva))
    return(matrix(cv_loss(loss, yva, f), nr, ns))
  }
  n <- nrow(Xtr)
  ## squared distances scaled by 2 sigma_m^2, summed over S  (training and validation)
  Dtr <- matrix(0, n, n)
  for (m in S) Dtr <- Dtr - log(pmax(Gm[, m], 1e-300))
  Dva <- matrix(0, nrow(Xva), n)
  for (m in S) Dva <- Dva + outer(Xva[, m], Xtr[, m], "-")^2 / (2 * sig[m]^2)
  out <- matrix(Inf, nr, ns)
  for (si in seq_len(ns)) {
    K <- exp(-Dtr / scales[si]^2)
    Kv <- exp(-Dva / scales[si]^2)
    out[, si] <- vapply(grid_r, function(g) {
      ab <- tryCatch(refit_fit(K, ytr, loss, g), error = function(e) NULL)
      if (is.null(ab)) return(Inf)
      e <- cv_loss(loss, yva, drop(Kv %*% ab$alpha) + ab$b)
      if (is.finite(e)) e else Inf
    }, 0)
  }
  out
}

## min rule: smallest mean CV error; ties -> larger gamma2, then larger gamma1.
## 1-SE rule: among grid points with mean CV error <= min + SE(min), the one with the fewest
##            selected coordinates (averaged over folds); ties -> smaller CV error.
cv_select <- function(E, NS, grid1, grid2) {
  K <- dim(E)[3]
  cvm <- apply(E, c(1, 2), mean)
  cvse <- apply(E, c(1, 2), sd) / sqrt(K)
  nsm <- apply(NS, c(1, 2), mean)
  best <- which(cvm == min(cvm), arr.ind = TRUE)
  best <- best[order(-grid2[best[, 2]], -grid1[best[, 1]]), , drop = FALSE][1, ]
  thr <- cvm[best[1], best[2]] + cvse[best[1], best[2]]
  ok <- which(cvm <= thr, arr.ind = TRUE)
  ok <- ok[order(nsm[ok], cvm[ok], -grid2[ok[, 2]], -grid1[ok[, 1]]), , drop = FALSE][1, ]
  list(cvm = cvm, cvse = cvse, nsel = nsm,
       gamma1 = grid1[best[1]], gamma2 = grid2[best[2]],
       gamma1_1se = grid1[ok[1]], gamma2_1se = grid2[ok[2]])
}

## ---------------------------------------------------------------------------
## relaxed refit on a selected set: product Gaussian kernel (rho = 1) + intercept,
## gamma1 and a common bandwidth multiplier tuned by CV
## ---------------------------------------------------------------------------

refit_kernel <- function(Xa, Za, sig) {
  K <- matrix(1, nrow(Za), nrow(Xa))
  for (m in seq_len(ncol(Xa))) K <- K * tpk_cross_G(Za[, m], Xa[, m], sig[m])
  K
}

refit_fit <- function(K, y, loss, gamma1) {
  n <- length(y)
  if (loss == "l2") return(alpha_l2(K, y, gamma1, TRUE, 1e-8))
  npos <- sum(y > 0); nneg <- sum(y < 0)
  w <- ifelse(y > 0, n / (2 * max(npos, 1)), n / (2 * max(nneg, 1)))
  alpha_exp(K, y, w, numeric(n), 0, gamma1, TRUE, 100, 1e-8)
}

tpk_refit <- function(X, y, S, loss, folds, grid1, scales = c(1, 2, 4)) {
  if (length(S) == 0) {
    b <- if (loss == "l2") mean(y) else (if (mean(y > 0) >= 0.5) 1 else -1)
    return(list(predict = function(Z) rep(b, nrow(Z)), gamma1 = NA, scale = NA))
  }
  Xa <- X[, S, drop = FALSE]
  sig0 <- tpk_bandwidth(Xa, "coord")
  E <- array(NA_real_, c(length(grid1), length(scales), max(folds)))
  for (k in seq_len(max(folds))) {
    tr <- folds != k
    for (s in seq_along(scales)) {
      sig <- sig0 * scales[s]
      Ktr <- refit_kernel(Xa[tr, , drop = FALSE], Xa[tr, , drop = FALSE], sig)
      Kva <- refit_kernel(Xa[tr, , drop = FALSE], Xa[!tr, , drop = FALSE], sig)
      for (i in seq_along(grid1)) {
        ab <- refit_fit(Ktr, y[tr], loss, grid1[i])
        E[i, s, k] <- cv_loss(loss, y[!tr], drop(Kva %*% ab$alpha) + ab$b)
      }
    }
  }
  cvm <- apply(E, c(1, 2), mean)
  best <- which(cvm == min(cvm), arr.ind = TRUE)
  best <- best[order(-grid1[best[, 1]]), , drop = FALSE][1, ]
  sig <- sig0 * scales[best[2]]
  ab <- refit_fit(refit_kernel(Xa, Xa, sig), y, loss, grid1[best[1]])
  list(predict = function(Z) drop(refit_kernel(Xa, Z[, S, drop = FALSE], sig) %*% ab$alpha) + ab$b,
       gamma1 = grid1[best[1]], scale = scales[best[2]])
}
