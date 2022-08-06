ImportTrackingData <- function(filePath)
{
  require(jsonlite);
  
  rawTrackingData <- fromJSON(filePath);
  
  return(rawTrackingData);
}


#rawTrackingData <- ImportTrackingData('C:/My_JSON_Data/EEGundBitalinoStage08-4_13-24-22.json')
#rawTrackingData <- ImportTrackingData('C:/My_JSON_Data/8-5_22-42-16.json')
