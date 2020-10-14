CreatePositionalHeatmap <- function(property,size)
{
  require(KernSmooth);
  require(raster);
  unlistedProperty <- matrix(unlist(property), ncol = 3, byrow = TRUE);
  flatPosition <- unlistedProperty[,c(1,3)];
  est <- bkde2D(flatPosition, bandwidth=c(0.1,0.1),gridsize=c(1000,1000),range.x=list(c(-size,size),c(-size,size)));
  
  est.raster = raster(list(x=est$x1,y=est$x2,z=est$fhat));
  projection(est.raster) <- CRS("+init=epsg:4326");
  xmin(est.raster) <- -size;
  xmax(est.raster) <- size;
  ymin(est.raster) <- -size;
  ymax(est.raster) <- size;
  plot(est.raster);
}