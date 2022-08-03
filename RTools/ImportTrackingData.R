ImportTrackingData <- function(filePath)
{
  require(jsonlite);
  
  rawTrackingData <- fromJSON(filePath);
  
  return(rawTrackingData);
}


#rawTrackingData <- ImportTrackingData('C:/My_JSON_Data/7-26_10-39-55.json')
