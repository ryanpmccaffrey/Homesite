# Homesite Kaggle Competiton - script used to run randon forest model

# Clear variables and set working directory
rm(list=ls())
setwd("/Users/ryanmccaffrey/Documents/Data Science/Kaggle/Homesite")

# Initialize packages 
library(readr)
library(xgboost)
library(randomForest)

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

# Feature engineering
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

# Create vector of all feature names
feature.names <- names(train)[3:315]

# Assume text variables are categorical and replace them with numeric IDs
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

# Fit model
fit <- randomForest(train[,2] ~ ., data=train[,feature.names], importance=TRUE, ntree=1250)
varImpPlot(fit)

# Predict and save file
pred1 <- predict(fit, data.matrix(test[,feature.names]))
submission <- data.frame(QuoteNumber=test$QuoteNumber, QuoteConversion_Flag=pred1)
cat("saving the submission file\n")
write_csv(submission, "xgb_try1_rf_1250ntrees.csv")
