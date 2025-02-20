
plotFn <- function(fn, from=-3, to=3, ylim=NULL) {
  name <- deparse(substitute(fn))
  par(cex=1.5)
  curve(fn(x), from=from, to=to, bty="n", lwd=3, ylab=sprintf("%s(x)", name), ylim=ylim)
}

plotFn(tanh, ylim=c(-1, 1))

ReLU <- function(x) {
  ifelse(x <= 0, 0, x)
}

plotFn(ReLU)

softplus <- \(x) log(1 + exp(x))

plotFn(softplus, to=3)
plotFn(softplus, to=100)

sigma <- \(x) 1/(1 + exp(-x))

plotFn(sigma, from=-5, to=5, ylim=c(0, 1))
