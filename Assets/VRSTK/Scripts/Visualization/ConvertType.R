ConvertTypes <- function(l)
{
  for (i in 1:length(l))
  {
    for (j in 1:length(l[[i]]))
    {
      if (length(l[[i]][j]) == 1)
      {
        if (!is.na(as.numeric(l[[i]][j])))
        {
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