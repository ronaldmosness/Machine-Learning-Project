# Clear workspace
rm(list = ls())

# Load required libraries
library(randomForest)
library(caret)
library(kernlab)  # for SVM model

# Import training data
traindata <- read.table(file = "7406train.csv", sep = ",")
dim(traindata)

## Extract X1 and X2
X1 <- traindata[, 1]
X2 <- traindata[, 2]

# Calculate muhat and Vhat
muhat <- apply(traindata[, 3:202], 1, mean)
Vhat <- apply(traindata[, 3:202], 1, var)

# Create data frame for training data
data0 <- data.frame(X1 = X1, X2 = X2, muhat = muhat, Vhat = Vhat)

# Set up a 2x2 grid for plots
par(mfrow = c(2, 2))

# Plot 1: Scatter plot for X1 vs muhat
plot(X1, muhat, col = "blue", main = "Scatter Plot: X1 vs muhat", xlab = "X1", ylab = "muhat")

# Plot 2: Scatter plot for X2 vs muhat
plot(X2, muhat, col = "green", main = "Scatter Plot: X2 vs muhat", xlab = "X2", ylab = "muhat")

# Plot 3: Scatter plot for X1 vs Vhat
plot(X1, Vhat, col = "red", main = "Scatter Plot: X1 vs Vhat", xlab = "X1", ylab = "Vhat")

# Plot 4: Scatter plot for X2 vs Vhat
plot(X2, Vhat, col = "purple", main = "Scatter Plot: X2 vs Vhat", xlab = "X2", ylab = "Vhat")

# Reset the plotting parameters to default
par(mfrow = c(1, 1))


# Import testing data
testX <- read.table(file = "7406test.csv", sep = ",")

# Random Forest Model
library(randomForest)
model_rf_mu <- randomForest(muhat ~ X1 + X2, data = traindata)
model_rf_var <- randomForest(Vhat ~ X1 + X2, data = traindata)

# Predict values for testing data using Random Forest
testdata_rf <- data.frame(X1 = testX[, 1], X2 = testX[, 2])
testdata_rf$muhat <- round(predict(model_rf_mu, newdata = testdata_rf), 6)
testdata_rf$Vhat <- round(predict(model_rf_var, newdata = testdata_rf), 6)



# Cross-validation Results for Random Forest
cv_rf_mu <- train(muhat ~ X1 + X2, data = data0, method = "rf", trControl = trainControl(method = "cv", number = 5))
cv_rf_var <- train(Vhat ~ X1 + X2, data = data0, method = "rf", trControl = trainControl(method = "cv", number = 5))

# Display the cross-validation results
print("Cross-validation Results using RandomForest:")
print("Mean Squared Error (mu):")
print(cv_rf_mu$results$RMSE)
print("Mean Squared Error (V):")
print(cv_rf_var$results$RMSE)


# Linear Model for muhat
model_lm_mu <- lm(muhat ~ X1 + X2, data = data0)
model_lm_var <- lm(Vhat ~ X1 + X2, data = data0)

# Predict values for testing data using Linear Model
testdata_lm_mu <- data.frame(X1 = testX[, 1], X2 = testX[, 2])
testdata_lm_mu$muhat <- round(predict(model_lm_mu, newdata = testdata_lm_mu), 6)
testdata_lm_mu$Vhat <- round(predict(model_lm_var, newdata = testdata_lm_mu), 6)



# Cross-validation Results for Linear Model
cv_lm_mu <- train(muhat ~ X1 + X2, data = data0, method = "lm")
print("Cross-validation Results for Linear Model (mu) :")
print("Mean Squared Error (mu):")
print(cv_lm_mu$results$RMSE)


# Cross-validation Results for Linear Model
cv_lm_var <- train(Vhat ~ X1 + X2, data = data0, method = "lm")
print("Cross-validation Results for Linear Model (V) :")
print("Mean Squared Error (V):")
print(cv_lm_var$results$RMSE)



# Support Vector Machine Model
model_svm_mu <- ksvm(muhat ~ X1 + X2, data = data0)
model_svm_var <- ksvm(Vhat ~ X1 + X2, data = data0)

# Predict values for testing data using SVM
testdata_svm <- data.frame(X1 = testX[, 1], X2 = testX[, 2])
testdata_svm$muhat <- round(predict(model_svm_mu, newdata = testdata_svm), 6)
testdata_svm$Vhat <- round(predict(model_svm_var, newdata = testdata_svm), 6)



# Cross-validation Results for SVM
cv_svm_mu <- train(muhat ~ X1 + X2, data = data0, method = "svmRadial")
cv_svm_var <- train(Vhat ~ X1 + X2, data = data0, method = "svmRadial")
print("Cross-validation Results using SVM:")
print("Mean Squared Error (mu):")
print(cv_svm_mu$results$RMSE)
print("Mean Squared Error (V):")
print(cv_svm_var$results$RMSE)


# Generalized Additive Model
library(mgcv)
model_gam_mu <- gam(muhat ~ s(X1) + s(X2), data = data0)
model_gam_var <- gam(Vhat ~ s(X1) + s(X2), data = data0)

# Predict values for testing data using GAM
testdata_gam <- data.frame(X1 = testX[, 1], X2 = testX[, 2])
testdata_gam$muhat <- round(predict(model_gam_mu, newdata = testdata_gam), 6)
testdata_gam$Vhat <- round(predict(model_gam_var, newdata = testdata_gam), 6)



# Cross-validation Results for GAM
cv_gam_mu <- train(muhat ~ X1 + X2, data = data0, method = "gam")
cv_gam_var <- train(Vhat ~ X1 + X2, data = data0, method = "gam")
print("Cross-validation Results using GAM:")
print("Mean Squared Error (mu):")
print(cv_gam_mu$results$RMSE)
print("Mean Squared Error (V):")
print(cv_gam_var$results$RMSE)

# Display overall cross-validation results
print("Overall Cross-validation Results:")
print("Random Forest - Mean Squared Error (mu):")
print(cv_rf_mu$results$RMSE)
print("Random Forest - Mean Squared Error (V):")
print(cv_rf_var$results$RMSE)
print("Linear Model - Mean Squared Error (mu):")
print(cv_lm_mu$results$RMSE)
print("Linear Model - Mean Squared Error (V):")
print(cv_lm_var$results$RMSE)
print("SVM - Mean Squared Error (mu):")
print(cv_svm_mu$results$RMSE)
print("SVM - Mean Squared Error (V):")
print(cv_svm_var$results$RMSE)
print("GAM - Mean Squared Error (mu):")
print(cv_gam_mu$results$RMSE)
print("GAM - Mean Squared Error (V):")
print(cv_gam_var$results$RMSE)



# Write the predicted values of svm model to a CSV file
write.table(
  testdata_svm,
  file = "1.Mosness.Ronald.csv",
  sep = ",",
  col.names = FALSE,
  row.names = FALSE
)