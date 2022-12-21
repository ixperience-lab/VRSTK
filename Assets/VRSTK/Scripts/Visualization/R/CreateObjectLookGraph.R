CreateObjectlookGraph <- function(input)
{
  objectLook <- input;
  objectNames <- objectLook$ObjectName;
  namesList <- objectNames[1];
  for (o in objectNames)
  {
    foundName <- FALSE;
    for (n in namesList)
    {
      if (o == n)
      {
        foundName <- TRUE;
      }
    }
    
    if (foundName == FALSE)
    {
      namesList <- c(namesList,o);
    }
  }
  valuesList <- rep(0, length(namesList))
  index1 <- 1;
  for (n in namesList)
  {
    index2 <- 1;
    for (o in objectNames)
    {
      if (o == n)
      {
        valuesList[index1] <- valuesList[index1] + objectLook$Duration[index2];
      }
      index2 <- index2 + 1;
    }
    index1 <- index1 + 1;
  }
  finishedData <- data.frame(namesList,valuesList);
  barplot(valuesList, names.arg = namesList);
  return(finishedData);
}