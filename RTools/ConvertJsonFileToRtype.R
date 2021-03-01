ConvertJsonFileToRtype <- function(filePath)
{
  require(jsonlite);
  myJSON <- fromJSON(filePath)
  print(myJSON)
  for (i in 1:length(myJSON))
  {
    for (j in 1:length(myJSON[[i]]))
    {
      myJSON[[i]][j] = ConvertTypes(myJSON[[i]][j]);
    }
  }
  return(myJSON);
}

#ConvertJsonFileToRtype("N:\\TestJSonDatensatz\\12-16_17-21-18.json")
ConvertJsonFileToRtype("N:\\TestJSonDatensatz\\3-1_13-50-21.json")