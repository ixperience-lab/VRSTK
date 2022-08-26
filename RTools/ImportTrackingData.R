library(psych)
library(stringi)
library(stringr)
library(jsonlite)
library(glue)
library(magrittr)
library(mnormt)

ImportTrackingData <- function(filePath)
{
  require(jsonlite);
  
  rawTrackingData <- fromJSON(filePath);
  
  return(rawTrackingData);
}


#rawTrackingData <- ImportTrackingData('Condition A/VRSTK/Proband-id-1-Condition-A_8-18_15-23-31.json')
#rawTrackingData <- ImportTrackingData('C:/My_JSON_Data/8-5_22-42-16.json')
#rawTrackingData <- ImportTrackingData('C:/My_JSON_Data/8-9_13-4-46.json')
#rawTrackingData <- ImportTrackingData('C:/My_JSON_Data/8-10_22-51-33.json')