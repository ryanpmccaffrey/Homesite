# Homesite Kaggle Competiton - script used to hypertune xgboost parameters

# Clear variables and set working directory
rm(list=ls())
setwd("/Users/ryanmccaffrey/Documents/Data Science/Kaggle/Homesite")

# Initialize packages
library(readr)
library(xgboost)
library(caret)

# Load train and test datasets
train <- read_csv("train.csv")
test  <- read_csv("test.csv")

# Flag NA values
train[is.na(train)]   <- -999
test[is.na(test)]   <- -999

# Separate out the elements of the date column for the train set
train$month <- as.integer(format(train$Original_Quote_Date, "%m"))
train$year <- as.integer(format(train$Original_Quote_Date, "%y"))
train$day <- weekdays(as.Date(train$Original_Quote_Date))

# Remove the date column
train <- train[,-c(2)]

# Separate out the elements of the date column for the test set
test$month <- as.integer(format(test$Original_Quote_Date, "%m"))
test$year <- as.integer(format(test$Original_Quote_Date, "%y"))
test$day <- weekdays(as.Date(test$Original_Quote_Date))

# Remove the date column
test <- test[,-c(2)]

# Feature engineering applied to both train and test datasets
train$Total_NAs_Count = rowSums(train[,3:301]==-999)
test$Total_NAs_Count = rowSums(test[,2:300]==-999)

train$Total_Zeros_Count = rowSums(train[,3:301]==0)
test$Total_Zeros_Count = rowSums(test[,2:300]==0)

train$Field_NAs_Count = rowSums(train[,3:9]==-999)
test$Field_NAs_Count = rowSums(test[,2:8]==-999)

train$Field_Zeros_Count = rowSums(train[,3:9]==0)
test$Field_Zeros_Count = rowSums(test[,2:8]==0)

train$Coverage_NAs_Count = rowSums(train[,10:25]==-999)
test$Coverage_NAs_Count = rowSums(test[,9:24]==-999)

train$Coverage_Zeros_Count = rowSums(train[,10:25]==0)
test$Coverage_Zeros_Count = rowSums(test[,9:24]==0)

train$Sales_NAs_Count = rowSums(train[,26:42]==-999)
test$Sales_NAs_Count = rowSums(test[,25:41]==-999)

train$Sales_Zeros_Count = rowSums(train[,26:42]==0)
test$Sales_Zeros_Count = rowSums(test[,25:41]==0)

train$Personal_NAs_Count = rowSums(train[,43:125]==-999)
test$Personal_NAs_Count = rowSums(test[,42:124]==-999)

train$Personal_Zeros_Count = rowSums(train[,43:125]==0)
test$Personal_Zeros_Count = rowSums(test[,42:124]==0)

train$Property_NAs_Count = rowSums(train[,126:172]==-999)
test$Property_NAs_Count = rowSums(test[,125:171]==-999)

train$Property_Zeros_Count = rowSums(train[,126:172]==0)
test$Property_Zeros_Count = rowSums(test[,125:171]==0)

train$Geographic_NAs_Count = rowSums(train[,173:298]==-999)
test$Geographic_NAs_Count = rowSums(test[,172:297]==-999)

train$Geographic_Zeros_Count = rowSums(train[,173:298]==0)
test$Geographic_Zeros_Count = rowSums(test[,172:297]==0)

# Load only most the important variables
# These variables are calculated from a previous xgboost run
load("important_variables.RData")
feature.names <- names(train)[important_variables]

# Assume text variables are categorical and replace them with numeric IDs
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}


dtrain <- xgb.DMatrix(data=data.matrix(train[,feature.names]), label=train$QuoteConversion_Flag) 

# Find optimum combination of subsample, colsample and max depth
searchGridSubCol <- expand.grid(subsample = c(0.5, 0.75, 1), 
                                colsample_bytree = c(0.6, 0.8, 1),
                                max_depth = c(4, 6)
)

ntrees <- 200 

aucErrorsHyperparameters <- apply(searchGridSubCol, 1, function(parameterList){
  #Extract Parameters to test
  currentSubsampleRate <- parameterList[["subsample"]]
  currentColsampleRate <- parameterList[["colsample_bytree"]]
  currentMaxDepth <- parameterList[["max_depth"]]  # e.g. extra parameters ...
  
  currentParam <- list(objective = "binary:logistic",
                       booster = "gbtree",
                       eval_metric = "auc",
                       eta = as.numeric(4/ntrees), 
                       max_depth = currentMaxDepth, #4, 
                       subsample = currentSubsampleRate,
                       colsample_bytree = currentColsampleRate
  )
  
  xgboostModelCV <- xgb.cv(params = currentParam,
                           data =  dtrain, 
                           nrounds = ntrees, 
                           nfold = 5,             # number of folds in K-fold
                           prediction = TRUE,     # return the prediction using the final model 
                           showsd = TRUE,         # standard deviation of loss across folds
                           stratified = TRUE,     # sample is unbalanced; use stratified sampling
                           verbose = F,
                           print.every.n = 1, 
                           early.stop.round = 10
  )
  
  maxTrainAuc <- max(xgboostModelCV$dt$test.auc.mean)
  TrainAucMatrix <- c(maxTrainAuc, currentSubsampleRate, currentColsampleRate, currentMaxDepth)
  return(TrainAucMatrix)
  
})