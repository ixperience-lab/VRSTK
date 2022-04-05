SaveFilesToWorkspace <- function(filePath)
{
  require(jsonlite);
  myFiles = list.files(filePath, pattern="*.json", full.names=TRUE);
  myJSON <- lapply(myFiles, function(x) fromJSON(x));
  for (i in 1:length(myJSON))
  {
    for (j in 1:length(myJSON[[i]]))
    {
      myJSON[[i]][j] = ConvertTypes(myJSON[[i]][j]);
    }
  }
  return(myJSON);
}