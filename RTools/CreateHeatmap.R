CreateHeatmap <- function ()
{
  test = JSONData$Stage0$Camera5952$position_Transform;
  require(MASS)
  unlisted <- matrix(unlist(test), ncol = 3, byrow = TRUE);
  dens <- kde2d(unlisted[,1], unlisted[,3], n=100);
  filled.contour(dens)
  #pointsMatrix = data.matrix(dataPoints);
}