#!/usr/bin/env Rscript
## Summarise all replicates written by run_one.R.
##
##   Rscript summarize.R [in=results] [out=summary]
##
## Writes to <out>/:
##   all_runs.csv        one row per (design, n, p, seed, method)
##   table_selection.csv TPR / TNR / Avg# per scenario and method   (cf. paper Table 1a / 2a)
##   table_error.csv     prediction error per scenario and method     (cf. paper Table 1b / 2b)
##   true_var_freq.csv   selection frequency of each true variable
##   weights.csv         distribution of the fitted lambda (orig) / rho (gated) on true and
##                       false positives
##   missing.csv         seeds absent for some scenario
##   plots.pdf           boxplots of prediction error and of the fitted weights
## and prints the two main tables.

args <- commandArgs(TRUE)
opt <- list(`in` = "results", out = "summary")
for (a in args) {
  kv <- strsplit(a, "=", fixed = TRUE)[[1]]
  if (length(kv) != 2 || !(kv[1] %in% names(opt))) stop("bad argument: ", a)
  opt[[kv[1]]] <- kv[2]
}
dir.create(opt$out, showWarnings = FALSE, recursive = TRUE)
TRUTH <- list(reg = 1:5, cls = 2:4)

files <- list.files(opt$`in`, pattern = "\\.rds$", full.names = TRUE)
if (!length(files)) stop("no .rds files in ", opt$`in`)

runs <- list(); wts <- list()
for (f in files) {
  r <- tryCatch(readRDS(f), error = function(e) NULL)
  if (is.null(r)) { warning("cannot read ", f); next }
  for (m in r$results) {
    runs[[length(runs) + 1]] <- data.frame(
      design = r$design, n = r$n, p = r$p, seed = r$seed, method = m$method,
      TPR = m$TPR, TNR = m$TNR, nsel = m$nsel, err = m$err,
      err_f = if (is.null(m$err_f)) NA else m$err_f, bayes = r$bayes_err,
      gamma1 = m$gamma1, gamma2 = m$gamma2, converged = m$converged, n_iter = m$n_iter,
      log10_maxK = m$log10_maxK, time_total = m$time_total,
      support = paste(m$support, collapse = " "), stringsAsFactors = FALSE)
    if (!grepl("_refit$", m$method) && length(m$support)) {
      wts[[length(wts) + 1]] <- data.frame(
        design = r$design, n = r$n, p = r$p, seed = r$seed, method = m$method,
        var = m$support, weight = m$t_support,
        true = m$support %in% TRUTH[[r$design]], stringsAsFactors = FALSE)
    }
  }
}
runs <- do.call(rbind, runs)
wts <- if (length(wts)) do.call(rbind, wts) else
  data.frame(design = character(), n = integer(), p = integer(), seed = integer(),
             method = character(), var = integer(), weight = numeric(), true = logical())
runs <- runs[order(runs$design, runs$p, runs$n, runs$method, runs$seed), ]
write.csv(runs, file.path(opt$out, "all_runs.csv"), row.names = FALSE)

key <- c("design", "p", "n", "method")
grp <- interaction(runs[key], drop = TRUE, lex.order = TRUE)
mad1 <- function(x) median(abs(x - median(x)))
agg <- function(fun) {
  out <- lapply(split(runs, grp), function(d) cbind(d[1, key], fun(d), row.names = NULL))
  out <- do.call(rbind, out); rownames(out) <- NULL
  out[order(out$design, out$p, out$n, out$method), , drop = FALSE]
}

## ---- feature selection ----------------------------------------------------
sel <- agg(function(d) data.frame(
  reps = nrow(d), TPR = mean(d$TPR), TNR = mean(d$TNR), avg_nsel = mean(d$nsel),
  exact = mean(d$TPR == 1 & d$TNR == 1), conv_rate = mean(d$converged),
  time_mean_s = mean(d$time_total)))
sel <- sel[!grepl("_refit$", sel$method), ]
write.csv(sel, file.path(opt$out, "table_selection.csv"), row.names = FALSE)

## ---- prediction error -----------------------------------------------------
## reg: test MSE against noisy y (and RMSE, the quantity reported in the published code);
## cls: test misclassification rate
er <- agg(function(d) data.frame(
  reps = nrow(d), err_mean = mean(d$err), err_median = median(d$err), err_mad = mad1(d$err),
  rmse_mean = if (d$design[1] == "reg") mean(sqrt(d$err)) else NA,
  err_f_mean = mean(d$err_f), bayes = mean(d$bayes)))
write.csv(er, file.path(opt$out, "table_error.csv"), row.names = FALSE)

## ---- per-variable selection frequency ---------------------------------------
freq <- agg(function(d) {
  tv <- TRUTH[[d$design[1]]]
  s <- strsplit(d$support, " ", fixed = TRUE)
  v <- sapply(tv, function(j) mean(vapply(s, function(z) as.character(j) %in% z, TRUE)))
  data.frame(var = paste0("X", tv), freq = v)
})
freq <- freq[!grepl("_refit$", freq$method), ]
write.csv(freq, file.path(opt$out, "true_var_freq.csv"), row.names = FALSE)

## ---- distribution of fitted weights ----------------------------------------
if (nrow(wts)) {
  wg <- interaction(wts[c(key, "true")], drop = TRUE, lex.order = TRUE)
  ws <- lapply(split(wts, wg), function(d) {
    q <- quantile(d$weight, c(0, 0.1, 0.25, 0.5, 0.75, 0.9, 1))
    data.frame(d[1, c(key, "true")], count = nrow(d),
               q00 = q[1], q10 = q[2], q25 = q[3], q50 = q[4], q75 = q[5], q90 = q[6], q100 = q[7],
               frac_ge_half_M = if (grepl("^orig", d$method[1])) mean(d$weight >= 5e4) else NA,
               frac_at_upper = if (grepl("^orig", d$method[1])) mean(d$weight >= 1e5 * (1 - 1e-8))
                               else mean(d$weight >= 1 - 1e-8))
  })
  ws <- do.call(rbind, ws); rownames(ws) <- NULL
  ws <- ws[order(ws$design, ws$p, ws$n, ws$method, -ws$true), ]
  write.csv(ws, file.path(opt$out, "weights.csv"), row.names = FALSE)
}

## ---- missing seeds ---------------------------------------------------------
sc <- unique(runs[c("design", "n", "p", "method")])
allseeds <- sort(unique(runs$seed))
miss <- do.call(rbind, lapply(seq_len(nrow(sc)), function(i) {
  s <- runs$seed[runs$design == sc$design[i] & runs$n == sc$n[i] & runs$p == sc$p[i] &
                 runs$method == sc$method[i]]
  m <- setdiff(allseeds, s)
  if (length(m)) data.frame(sc[i, ], missing = paste(m, collapse = " ")) else NULL
}))
if (!is.null(miss)) write.csv(miss, file.path(opt$out, "missing.csv"), row.names = FALSE)

## ---- plots -----------------------------------------------------------------
pdf(file.path(opt$out, "plots.pdf"), width = 11, height = 7)
for (d in unique(runs$design)) {
  sub <- runs[runs$design == d, ]
  sc2 <- unique(sub[c("p", "n")]); sc2 <- sc2[order(sc2$p, sc2$n), ]
  par(mfrow = c(ceiling(nrow(sc2) / 3), 3), mar = c(4, 4, 2, 1))
  for (i in seq_len(nrow(sc2))) {
    s <- sub[sub$p == sc2$p[i] & sub$n == sc2$n[i], ]
    boxplot(err ~ method, data = s, main = sprintf("%s  n=%d  p=%d", d, sc2$n[i], sc2$p[i]),
            ylab = if (d == "reg") "test MSE" else "misclassification", xlab = "", las = 1)
  }
  for (m in unique(wts$method[wts$design == d])) {
    s <- wts[wts$design == d & wts$method == m, ]
    par(mfrow = c(1, 1), mar = c(7, 5, 3, 1))
    lab <- sprintf("n%d p%d %s", s$n, s$p, ifelse(s$true, "TP", "FP"))
    lab <- factor(lab, levels = unique(lab[order(s$p, s$n, !s$true)]))
    yy <- if (grepl("^orig", m)) log10(s$weight) else s$weight
    boxplot(yy ~ lab, las = 2, xlab = "",
            ylab = if (grepl("^orig", m)) "log10(lambda)" else "rho",
            main = sprintf("%s / %s: fitted weights on selected variables", d, m))
  }
}
dev.off()

## ---- print -----------------------------------------------------------------
fmt <- function(df) {
  df[] <- lapply(df, function(x) if (is.numeric(x)) signif(x, 4) else x)
  print(df, row.names = FALSE)
}
cat("\n== Feature selection ==\n"); fmt(sel)
cat("\n== Prediction error ==\n"); fmt(er)
cat("\n== Selection frequency of true variables ==\n"); fmt(freq)
if (!is.null(miss)) cat("\nSome seeds are missing; see", file.path(opt$out, "missing.csv"), "\n")
cat("\nFiles written to", normalizePath(opt$out), "\n")
