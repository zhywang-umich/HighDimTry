#!/usr/bin/env Rscript
## One replicate of the simulation study.
##
##   Rscript run_one.R seed=<id> [design=reg,cls] [n=100,200,400] [p=200,400,1000]
##                     [methods=kse,kse_ard,gated,orig] [out=results] [ntest=10000] [cls_cv=exp|misclass]
##
## Every (design, n, p) combination is run for this seed; each method is saved to
##   <out>/<design>_n<n>_p<p>_seed<seed>_<method>.rds
## Existing files are skipped, so a killed job can simply be resubmitted, and methods can be
## run in separate jobs into the same directory (the data depend only on seed, design, n, p).
## With SLURM: seed defaults to $SLURM_ARRAY_TASK_ID.

get_script_dir <- function() {
  a <- commandArgs(FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) dirname(normalizePath(f)) else getwd()
}
here <- get_script_dir()
source(file.path(here, "R", "tpk.R"))
source(file.path(here, "R", "sim.R"))
source(file.path(here, "R", "kse.R"))

## ---- arguments -------------------------------------------------------------
args <- commandArgs(TRUE)
opt <- list(seed = Sys.getenv("SLURM_ARRAY_TASK_ID", "1"), design = "reg,cls",
            n = "100,200,400", p = "200,400,1000", methods = "kse,kse_ard,gated",
            out = "results", ntest = "10000", cls_cv = "exp")
for (a in args) {
  kv <- strsplit(a, "=", fixed = TRUE)[[1]]
  if (length(kv) != 2 || !(kv[1] %in% names(opt))) stop("bad argument: ", a)
  opt[[kv[1]]] <- kv[2]
}
spl <- function(s) strsplit(s, ",", fixed = TRUE)[[1]]
seed <- as.integer(opt$seed)
designs <- spl(opt$design)
ns <- as.integer(spl(opt$n)); ps <- as.integer(spl(opt$p))
methods <- spl(opt$methods)
ntest <- as.integer(opt$ntest)
options(tpk.cls_cv = opt$cls_cv)
dir.create(opt$out, showWarnings = FALSE, recursive = TRUE)

GRID <- 2^seq(-15, 15, by = 2)          # paper: 2^-15, 2^-13, ..., 2^15
DFMAX <- 30                             # gated: stop a gamma2 path beyond 30 active coordinates

## ---- one method ------------------------------------------------------------
## KSE (R/kse.R): forward REML path + EBIC + backward elimination; one row per xi
run_kse <- function(method, dat, test, des) {
  p <- ncol(dat$X)
  gate <- method == "kse"
  t0 <- proc.time()[3]
  fits <- kse_fit(dat$X, dat$y, gate = gate, xi = c(0, 0.5, 1))
  tt <- proc.time()[3] - t0
  lapply(names(fits), function(nm) {
    md <- fits[[nm]]
    S <- md$S
    fhat <- md$predict(test$X)
    k <- length(S)
    pp <- kse_unpack(md$par, k, gate)
    list(method = paste0(method, "_", nm), rule = nm, gamma1 = NA, gamma2 = NA,
         support = S, t_support = pp$rho, u_support = pp$u, logphi = pp$logphi,
         TPR = mean(des$truth %in% S), TNR = mean(!(setdiff(seq_len(p), des$truth) %in% S)),
         nsel = k,
         err = if (des$loss == "l2") mean((test$y - fhat)^2) else mean(ifelse(fhat >= 0, 1, -1) != test$y),
         err_f = if (des$loss == "l2") mean((test$f - fhat)^2) else NA,
         n_iter = NA, converged = TRUE, log10_maxK = 0, time_cv = 0, time_total = tt)
  })
}

run_method <- function(method, dat, test, des, folds) {
  if (method %in% c("kse", "kse_ard")) return(run_kse(method, dat, test, des))
  loss <- des$loss
  p <- ncol(dat$X)
  cfg <- switch(method,
    gated = list(kernel = "gated", bw = "coord", solver = "lbfgs", warm = TRUE, dfmax = DFMAX),
    orig  = list(kernel = "orig", bw = "global", solver = "cd", warm = FALSE, dfmax = Inf),
    orig_coord = list(kernel = "orig", bw = "coord", solver = "cd", warm = FALSE, dfmax = Inf),
    stop("unknown method ", method))
  t0 <- proc.time()[3]
  cv <- tpk_cv(dat$X, dat$y, kernel = cfg$kernel, loss = loss, bw = cfg$bw, folds = folds,
               grid1 = GRID, grid2 = GRID, warm = cfg$warm, dfmax = cfg$dfmax,
               solver = cfg$solver, relax = cfg$kernel == "gated")
  t_cv <- proc.time()[3] - t0
  sig <- tpk_bandwidth(dat$X, cfg$bw)
  Gm <- tpk_bank(dat$X, sig)
  fitfun <- if (cfg$solver == "lbfgs") tpk_fit_lbfgs else tpk_fit
  truth <- des$truth
  score <- function(fhat) {
    if (loss == "l2") list(err = mean((test$y - fhat)^2), err_f = mean((test$f - fhat)^2))
    else list(err = mean(ifelse(fhat >= 0, 1, -1) != test$y), err_f = NA)
  }
  out <- list(); fits <- list()
  rules <- if (cfg$kernel == "gated") c("min", "relax") else "min"
  for (rule in rules) {
    g1 <- if (rule == "min") cv$gamma1 else cv$gamma1_relax
    g2 <- if (rule == "min") cv$gamma2 else cv$gamma2_relax
    t1 <- proc.time()[3]
    key <- paste(g1, g2)
    fit <- fits[[key]]
    if (is.null(fit)) {
      fit <- fitfun(dat$X, dat$y, Gm, sig, kernel = cfg$kernel, loss = loss, gamma1 = g1, gamma2 = g2)
      fits[[key]] <- fit
    }
    S <- fit$support
    sc <- score(predict(fit, test$X))
    name <- if (rule == "min") method else paste0(method, "_relax")
    res <- list(method = name, rule = rule, gamma1 = g1, gamma2 = g2,
                support = S, t_support = fit$t[S],
                TPR = mean(truth %in% S), TNR = mean(!(setdiff(seq_len(p), truth) %in% S)),
                nsel = length(S), err = sc$err, err_f = sc$err_f,
                n_iter = fit$n_iter, converged = fit$converged, log10_maxK = fit$log10_maxK,
                time_cv = t_cv, time_total = t_cv + proc.time()[3] - t1)
    out[[length(out) + 1]] <- res
    if (cfg$kernel == "gated") {
      t2 <- proc.time()[3]
      rf <- tpk_refit(dat$X, dat$y, S, loss, folds, GRID)
      sc2 <- score(rf$predict(test$X))
      r2 <- res
      r2$method <- paste0(name, "_refit")
      r2$refit_gamma1 <- rf$gamma1; r2$refit_scale <- rf$scale
      r2$err <- sc2$err; r2$err_f <- sc2$err_f
      r2$time_total <- res$time_total + proc.time()[3] - t2
      out[[length(out) + 1]] <- r2
    }
  }
  rm(Gm); gc(verbose = FALSE)
  attr(out, "cv") <- list(method = method, cvm = cv$cvm, cvse = cv$cvse, nsel = cv$nsel,
                          cvm_relax = cv$cvm_relax, grid = GRID)
  out
}

## ---- loop over scenarios ---------------------------------------------------
scen <- expand.grid(p = ps, n = ns, design = designs, stringsAsFactors = FALSE)
for (k in seq_len(nrow(scen))) {
  d <- scen$design[k]; n <- scen$n[k]; p <- scen$p[k]
  files <- setNames(file.path(opt$out, sprintf("%s_n%d_p%d_seed%d_%s.rds", d, n, p, seed, methods)),
                    methods)
  todo <- methods[!file.exists(files)]
  for (m in setdiff(methods, todo)) cat("skip", files[[m]], "\n")
  if (!length(todo)) next
  des <- DESIGNS[[d]]
  ## data seed depends on (seed, design, n, p) only: every method sees the same data
  sid <- (seed * 100003 + match(d, names(DESIGNS)) * 7919 + n * 31 + p) %% 2147483647
  set.seed(sid)
  dat <- des$sim(n, p)
  test <- des$sim(ntest, p)
  folds <- sample(rep(1:3, length.out = n))
  bayes <- if (d == "cls") mean(ifelse(test$f > 0.5, 1, -1) != test$y) else mean((test$y - test$f)^2)
  for (m in todo) {
    cat(sprintf("[%s] %s n=%d p=%d seed=%d method=%s ... ", format(Sys.time(), "%H:%M:%S"), d, n, p, seed, m))
    r <- run_method(m, dat, test, des, folds)
    for (ri in r) cat(sprintf("%s: TPR=%.2f TNR=%.3f nsel=%d err=%.3f (%.0fs)  ",
                              ri$method, ri$TPR, ri$TNR, ri$nsel, ri$err, ri$time_total))
    cat("\n")
    cv <- attr(r, "cv"); if (!is.null(cv)) attr(r, "cv") <- NULL
    saveRDS(list(design = d, n = n, p = p, seed = seed, bayes_err = bayes, results = r, cv = cv,
                 cls_cv = opt$cls_cv),
            files[[m]])
  }
}
