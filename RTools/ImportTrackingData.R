ImportTrackingData <- function(filePath)
{
  require(jsonlite);
  
  rawTrackingData <- fromJSON(filePath);
  
  return(rawTrackingData);
}


#rawTrackingData <- ImportTrackingData('C:/My_JSON_Data/EEGundBitalinoStage08-4_13-24-22.json')
#rawTrackingData <- ImportTrackingData('C:/My_JSON_Data/8-5_22-42-16.json')
#rawTrackingData <- ImportTrackingData('C:/My_JSON_Data/8-9_13-4-46.json')
#rawTrackingData <- ImportTrackingData('C:/My_JSON_Data/8-9_20-59-55.json')