ConvertJsonFileToRtype <- function(filePath)
{
  require(jsonlite);
  myJSON <- fromJSON(filePath)
  #print(myJSON);
  print("root");
  print(length(myJSON));
  
  for (i in 1:length(myJSON))
  {
    #print("sub j");
    #print(length((myJSON[[i]])));
    for (j in 1:length(myJSON[[i]]))
    {
      #print("sub k");
      #print(length(myJSON[[i]][j]));
      #for (k in 1:length(myJSON[[i]][j]))
      #{
        #print("sub l");
        #print(length(myJSON[[i]][j][[k]]));
        #for (l in 1:length(myJSON[[i]][j][[k]]))
        #{
          #print(length(myJSON[[i]][j][[k]][l]));
          #if (length(l[[i]][j]) == 1)
          #{}
        #}
      #}
      myJSON[[i]][j] = ConvertTypes(myJSON[[i]][j]);
    }
  }
  return(myJSON);
}

#ConvertJsonFileToRtype("N:\\TestJSonDatensatz\\12-16_17-21-18.json")
#ConvertJsonFileToRtype("C:\\My_JSON_Data\\9-22_12-59-38.json")
#ConvertJsonFileToRtype("C:\\My_JSON_Data\\9-22_12-59-38.json")
