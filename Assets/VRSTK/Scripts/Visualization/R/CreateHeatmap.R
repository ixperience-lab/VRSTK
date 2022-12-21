CreateHeatmap <- function ()
{
  #test = TestData$Stage0$Camera16770$position_Transform;
  eyeHitPoints = testData$Stage0$EyeLookAtObject$EyeHitPoint;
  #eyeHitPoints = testData$Stage0$EyeLookAtObject$EyeDirection;
  #eyeHitPoints = TestData2$Stage0$EyeLookAtObject$EyeDirection;
  
  #eyeHitPoints = testData2$Stage0$EyeLookAt$position_Transform;
  
  #test = JSONData$Stage0$Camera5952$position_Transform;
  require(MASS)
  unlisted <- matrix(unlist(eyeHitPoints), ncol = 3, byrow = TRUE);
  dens <- kde2d(unlisted[,1], unlisted[,3], n=100);
  filled.contour(dens)
  #pointsMatrix = data.matrix(dataPoints);
}



