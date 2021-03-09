CreateHeatmap <- function ()
{
  #test = TestData$Stage0$Camera16770$position_Transform;
  #eyeHitPoints = TestData$Stage0$EyeLookAtObject$EyeHitPoint;
  eyeHitPoints = TestData$Stage0$EyeLookAtObject$EyeDirection;
  #test = JSONData$Stage0$Camera5952$position_Transform;
  require(MASS)
  unlisted <- matrix(unlist(eyeHitPoints), ncol = 3, byrow = TRUE);
  dens <- kde2d(unlisted[,1], unlisted[,3], n=100);
  filled.contour(dens)
  #pointsMatrix = data.matrix(dataPoints);
}



