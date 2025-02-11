
xmin<- -1
xmax<- 1
dens<-1/(xmax-xmin)
var <-"X = 2Z - 1"
par(mar=c(5, 5, 1, 1))
plot(0, type="n", xlim=c(xmin-0.5, xmax+0.5), ylim=c(0, 1.2), bty="n", axes=FALSE, xlab=var, ylab="Density", cex.lab=2)
axis(1, at=seq(xmin-0.5, xmax+0.5, by=0.5), outer=FALSE, cex.axis=1.5, lwd=2)
axis(2, las=1, cex.axis=1.5, lwd=2)

rect(xmin, 0, xmax, dens, col=adjustcolor("gray", alpha=0.7), lty=0)
lines(c(xmin-0.5, xmin), c(0, 0), lwd=5)
lines(c(xmin, xmax), c(dens, dens), lwd=5)
lines(c(xmax, xmax+0.5), c(0, 0), lwd=5)
lines(c(xmin, xmin), c(0, dens), lty=2, lwd=3)
lines(c(xmax, xmax), c(0, dens), lty=2, lwd=3)
points(c(xmin, xmax), c(0, 0), pch=21, bg="white", cex=1.5)
points(c(xmin, xmax), c(dens, dens), pch=19, cex=1.5)



