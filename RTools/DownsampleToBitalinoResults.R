# Downsampling to: the samplingrate of ecg and eda 
# Downsampling of: Eye-Tracking, Band-Power, Performance-Metric

countBandPowerSamples <- nrow(rawBandPowerDataFrameStage0)
countTransformedBitalinoSamples <- nrow(transformedBilinoECGDataFrameStage0)

factor <- countBandPowerSamples / countTransformedBitalinoSamples

#test <- NULL
#test <- downSample(rawBandPowerDataFrameStage0, transformedBilinoECGDataFrameStage0, yname = "DownSampled")

# Attach packages
#library(groupdata2)

# Create data frame
df <- data.frame(
  "participant" = factor(c(1, 1, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5)),
  "diagnosis" = factor(c(0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0)),
  "trial" = c(1, 2, 1, 1, 2, 3, 4, 1, 2, 1, 2, 3, 4),
  "score" = sample(c(1:100), 13)
)

test <- downSample(df, c(0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0), yname = "DownSampled")

# Using downsample()
#test <- downSample(df, cat_col = "diagnosis")

#test <- downSample(test, cat_col = "diagnosis")

# Using downsample() with id_method "n_ids"
# With column specifying added rows
#downsample(df, cat_col = "diagnosis", id_col = "participant", id_method = "n_ids"
#)

# Using downsample() with id_method "n_rows_c"
# With column specifying added rows
#downsample(df, cat_col = "diagnosis", id_col = "participant",  id_method = "n_rows_c")

# Using downsample() with id_method "distributed"
#downsample(df, cat_col = "diagnosis", id_col = "participant", id_method = "distributed")

# Using downsample() with id_method "nested"
#downsample(df, cat_col = "diagnosis", id_col = "participant", id_method = "nested")
