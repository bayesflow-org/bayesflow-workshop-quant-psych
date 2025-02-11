
dmix <- function(x, w, m, s) {
  w <- w/sum(w)
  out <- rep(0, length(x))
  for (i in seq_along(w)) {
    out <- out + w[i] * dnorm(x, m[i], s[i])
  }

  return(out)
}

par(oma=c(0,0,0,0), mar=c(0,0,0,0), mfrow=c(1, 1))

curve(
  dmix(x=x,c(5, 4, 2, 3), c(-2.5, -1, 1, 3), c(0.9, 0.3, 0.7, 0.5)),
  from=-5, to=5, n=1000, lwd=10,
  axes=FALSE, xlab="", ylab="")

curve(
  dmix(x=x,c(5, 4, 3), c(-1.8, -1, 2.5), c(1.1, 0.4, 0.7)),
  from=-5, to=5, n=1000, lwd=10,
  axes=FALSE, xlab="", ylab="")

curve(
  dmix(x=x,c(5, 3), c(-1, 3), c(1.5, 1)),
  from=-5, to=5, n=1000, lwd=10,
  axes=FALSE, xlab="", ylab="")

curve(
  dmix(x=x,1, 0, 1),
  from=-5, to=5, n=1000, lwd=10,
  axes=FALSE, xlab="", ylab="")

