ConvertTypes <- function(l)
{
  for (i in 1:length(l))
  {
    for (j in 1:length(l[[i]]))
    {
      if (length(l[[i]][j]) == 1)
      {
        if (!is.na(is.numeric(l[[i]][j])))
        {
          #l[[i]][j] = as.numeric(unlist(l[[i]][j]));
          #lapply(l[[i]][j][0], as.numeric);
          
          l[[i]][j] = as.numeric(l[[i]][j]);
        }
        if (l[[i]][j] == "True")
        {
          l[[i]][j] = TRUE;
        }
        if (l[[i]][j] == "False")
        {
          l[[i]][j] = FALSE;
        }
      }
    }
  }
  return(l);
}