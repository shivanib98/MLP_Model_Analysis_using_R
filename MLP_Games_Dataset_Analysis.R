#Original file is located at
#    https://colab.research.google.com/drive/1TUiBEbN2S6lyWuMRnHIMQmBSF_hRkKlz


# Multilayer Perceptron Application on publicly available Games details Dataset

#Steps to upload the preprocessed file if using Google Collab:
#1. On the left pane click on the files
#2. Click on upload to session storage
#3. Select Updatedgames_details.csv and click ok
# You are good to proceed now

# Install required packages
install.packages(c("magrittr", "dplyr","MASS","tensorflow","caret","ggplot2","keras","glmnet"))

# Load libraries
library(MASS) # Library for the "Boston" dataset
library(keras) # Interface to the "Keras" deep learning library for building neural networks
library(tensorflow) # Provides the backend for the "Keras" package, using TensorFlow for computation
library(magrittr) # Provides a way to chain functions together using the pipe operator %>%.
library(dplyr) # Package for data manipulation and transformation
library(ggplot2) # Library for creating visualizations and plots
library(glmnet) # Library for fitting generalized linear models via penalized maximum likelihood (Lasso and Ridge regression)

# Set the file path of the uploaded CSV file
file_path <- "/PATH/Updatedgames_details.csv"

# Read the CSV file
df <- read.csv(file_path)

# Print the first few rows of the data frame (head)
head(df)

# Print the summary statistics of the data frame
summary(df)

# Function to calculate model evaluation parameters:
# MSE - Mean Square Error: Measures the average squared difference between predicted and actual values. It quantifies the model's prediction accuracy, with lower values indicating better performance.
# MAE - Mean Absolute Error: Measures the average absolute difference between predicted and actual values. It provides an average of the absolute prediction errors and is less sensitive to outliers compared to MSE.
# RMSE - Root Mean Square Error: The square root of MSE, providing an interpretation of the error in the original units of the target variable.
# R_squared: Also known as the "coefficient of determination," it indicates the proportion of variance in the dependent variable explained by the model. A value closer to 1 suggests a better fit.
# Adjusted_R_squared: A modification of R_squared that considers the number of predictors in the model. It penalizes the inclusion of unnecessary predictors, and a higher value indicates a more parsimonious model.

calculate_metrics <- function(predictions, actual_values, num_predictors) {
  mse <- mean((actual_values-predictions)^2)
  mae <- mean(abs(actual_values-predictions))
  rmse <- sqrt(mse)
  rsquared <- 1 - sum((actual_values - predictions)^2) / sum((actual_values - mean(actual_values))^2)
  adj_rsquared <- 1 - (1 - rsquared) * ((length(actual_values) - 1) / (length(actual_values) - num_predictors - 1))
  return(list(MSE = mse, MAE = mae, RMSE = rmse, R_squared = rsquared, Adjusted_R_squared = adj_rsquared))
}

# Loading the games dataset
data <- df

#To ensure reproducibility
set.seed(500)

# Split the data into training and testing sets
index <- sample(1:nrow(data), round(0.70 * nrow(data)))

# Scaling the data using z-score method
scaled <- as.data.frame(scale(data[, -1], center = TRUE, scale = TRUE))

# Adding back the target variable 'crim' to the scaled data
scaled <- cbind(scaled, crim = data$PTS)

# Split the scaled data into training and testing sets
train_ <- scaled[index, ]
test_ <- scaled[-index, ]

# Convert the data to matrix format
x_train <- as.matrix(train_[, -1])
y_train <- train_$PTS
x_test <- as.matrix(test_[, -1])
y_test <- test_$PTS

# MLP
# Defining the MLP model
model_mlp <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = 'relu', input_shape = ncol(x_train)) %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dense(units = 1) # Output layer representing the predicted 'crim' values

# Compiling the MLP model
model_mlp %>% compile(
  optimizer = optimizer_adam(), # Adam optimizer used for gradient-based optimization
  loss = "mse", # Mean Squared Error used as the loss function to measure prediction accuracy
  metrics = list("mean_absolute_error") # Evaluation metric to track during training
)

# Training the MLP model
history_mlp <- model_mlp %>% fit(
  x_train,
  y_train,
  epochs = 200, # Number of epochs (iterations) to train the model
  validation_split = 0.2, # 20% of the training data is reserved for validation
  verbose = 0 # Setting verbose to 0 means no progress bar during training
)

# model_mlp represents the trained MLP model
model_mlp

# Loss and Mean Absolute Error Plots
plot(history_mlp)

# Evaluation of MLP model on test set
predictions_mlp <- model_mlp %>% predict(x_test)
metrics_mlp <- calculate_metrics(predictions_mlp, y_test, ncol(x_test) - 1)
cat("MLP:\n")
cat("MSE:", metrics_mlp$MSE, "\n")
cat("MAE:", metrics_mlp$MAE, "\n")
cat("RMSE:", metrics_mlp$RMSE, "\n")
cat("R-squared:", metrics_mlp$R_squared, "\n")
cat("Adjusted R-squared:", metrics_mlp$Adjusted_R_squared, "\n\n")

# Scatter Plot of Predicted vs. Actual Values
plot_df <- data.frame(Actual = y_test, Predicted = predictions_mlp)
ggplot(plot_df, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(x = "Actual Values", y = "Predicted Values", title = "MLP: Predicted vs. Actual")

# Defining the Ridge regularized MLP model
model_ridge_mlp <- keras_model_sequential() %>%
  # applying L2 regularization with a regularization parameter of 0.01 to the weights of layer
  layer_dense(units = 64, activation = 'relu',
              kernel_regularizer = regularizer_l2(0.01),
              input_shape = ncol(x_train)) %>%
  layer_dense(units = 32, activation = 'relu', kernel_regularizer = regularizer_l2(0.01)) %>%
  layer_dense(units = 1)

# Compile the Ridge regularized MLP model
model_ridge_mlp %>% compile(
  optimizer = optimizer_adam(),
  loss = "mse",
  metrics = list("mean_absolute_error")
)

# Train the Ridge regularized MLP model
history_ridge_mlp <- model_ridge_mlp %>% fit(
  x_train,
  y_train,
  epochs = 200,
  validation_split = 0.2,
  verbose = 0
)

# model_ridge_mlp represents the trained Ridge regularized MLP model.
model_ridge_mlp

# Loss and Mean Absolute Error Plots
plot(history_ridge_mlp)

# Evaluating the Ridge regularized MLP model on test set
predictions_ridge_mlp <- model_ridge_mlp %>% predict(x_test)
metrics_ridge_mlp <- calculate_metrics(predictions_ridge_mlp, y_test, ncol(x_test) - 1)
cat("Ridge Regression with MLP:\n")
cat("MSE:", metrics_ridge_mlp$MSE, "\n")
cat("MAE:", metrics_ridge_mlp$MAE, "\n")
cat("RMSE:", metrics_ridge_mlp$RMSE, "\n")
cat("R-squared:", metrics_ridge_mlp$R_squared, "\n")
cat("Adjusted R-squared:", metrics_ridge_mlp$Adjusted_R_squared, "\n\n")

# Scatter Plot of Predicted vs. Actual Values
plot_df <- data.frame(Actual = y_test, Predicted = predictions_ridge_mlp)
ggplot(plot_df, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(x = "Actual Values", y = "Predicted Values", title = "MLP with Ridge Regression: Predicted vs. Actual")

# Defining the Lasso regularized MLP model
model_lasso_mlp <- keras_model_sequential() %>%
  #applying L1 regularization with a regularization parameter of 0.01 to the weights of layer
  layer_dense(units = 32, activation = 'relu', input_shape = ncol(x_train), kernel_regularizer = regularizer_l1(0.01)) %>%
  layer_dense(units = 16, activation = 'relu', kernel_regularizer = regularizer_l1(0.01)) %>%
  layer_dense(units = 1)

# Compile the Lasso regularized MLP model
model_lasso_mlp %>% compile(
  optimizer = optimizer_adam(),
  loss = "mse",
  metrics = list("mean_absolute_error")
)

# Train the Lasso regularized MLP model
history_lasso_mlp <- model_lasso_mlp %>% fit(
  x_train,
  y_train,
  epochs = 200,
  validation_split = 0.2,
  verbose = 0
)

# The model_lasso_mlp now represents the trained Lasso regularized MLP model.
model_lasso_mlp

# Loss and Mean Absolute Error Plots
plot(history_lasso_mlp)

# Evaluating the Lasso regularized MLP model on test set
predictions_lasso_mlp <- model_lasso_mlp %>% predict(x_test)
metrics_lasso_mlp <- calculate_metrics(predictions_lasso_mlp, y_test, ncol(x_test) - 1)
cat("Lasso Regression with MLP:\n")
cat("MSE:", metrics_lasso_mlp$MSE, "\n")
cat("MAE:", metrics_lasso_mlp$MAE, "\n")
cat("RMSE:", metrics_lasso_mlp$RMSE, "\n")
cat("R-squared:", metrics_lasso_mlp$R_squared, "\n")
cat("Adjusted R-squared:", metrics_lasso_mlp$Adjusted_R_squared, "\n\n")

# Scatter Plot of Predicted vs. Actual Values
plot_df <- data.frame(Actual = y_test, Predicted = predictions_lasso_mlp)
ggplot(plot_df, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(x = "Actual Values", y = "Predicted Values", title = "MLP with Lasso Regression: Predicted vs. Actual")

# Defining the Elastic Net regularized MLP model
model_enet_mlp <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = 'relu', input_shape = ncol(x_train), kernel_regularizer = regularizer_l1_l2(l1 = 0.01, l2 = 0.01)) %>%
  layer_dense(units = 32, activation = 'relu', kernel_regularizer = regularizer_l1_l2(l1 = 0.01, l2 = 0.01)) %>%
  layer_dense(units = 1)

# Compile the Elastic Net regularized MLP model
model_enet_mlp %>% compile(
  optimizer = optimizer_adam(),
  loss = "mse",
  metrics = list("mean_absolute_error")
)

# Train the Elastic Net regularized MLP model
history_enet_mlp <- model_enet_mlp %>% fit(
  x_train,
  y_train,
  epochs = 200,
  validation_split = 0.2,
  verbose = 0
)

# Trained Elastic Net regularized MLP model
model_enet_mlp

# Loss and Mean Absolute Error Plots
plot(history_enet_mlp)

# Evaluating the Elastic Net regularized MLP model on test set
predictions_enet_mlp <- model_enet_mlp %>% predict(x_test)
metrics_enet_mlp <- calculate_metrics(predictions_enet_mlp, y_test, ncol(x_test) - 1)
cat("Elastic Net with MLP:\n")
cat("MSE:", metrics_enet_mlp$MSE, "\n")
cat("MAE:", metrics_enet_mlp$MAE, "\n")
cat("RMSE:", metrics_enet_mlp$RMSE, "\n")
cat("R-squared:", metrics_enet_mlp$R_squared, "\n")
cat("Adjusted R-squared:", metrics_enet_mlp$Adjusted_R_squared, "\n\n")

# Scatter Plot of Predicted vs. Actual Values
plot_df <- data.frame(Actual = y_test, Predicted = predictions_enet_mlp)
ggplot(plot_df, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(x = "Actual Values", y = "Predicted Values", title = "MLP with Elastic Net Regression: Predicted vs. Actual")

#Metrics table for each model
model_names <- c("MLP", "Ridge MLP", "Lasso MLP", "Elastic Net MLP")
metrics <- list(metrics_mlp, metrics_ridge_mlp, metrics_lasso_mlp, metrics_enet_mlp)
metric_names <- c("MSE", "MAE", "RMSE", "R-squared", "Adjusted R-squared")

metrics_table <- matrix(NA, nrow = length(model_names), ncol = length(metric_names))
colnames(metrics_table) <- metric_names
rownames(metrics_table) <- model_names

for (i in 1:length(model_names)) {
  metrics_table[i, ] <- unlist(metrics[[i]])
}

# Printing the comparison table
print(metrics_table)
# Create a bar plot to compare metrics
metrics_df <- data.frame(Model = rep(model_names, each = length(metric_names)), Metric = metric_names,
                         Value = unlist(metrics))

# Define the custom order of the model names
custom_order <- c("MLP", "Ridge MLP", "Lasso MLP", "Elastic Net MLP")  # Replace with your desired order

# Reorder the "Model" factor variable based on the custom order
metrics_df$Model <- factor(metrics_df$Model, levels = custom_order)

# Create the bar plot with the custom sorted order
ggplot(metrics_df, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Model", y = "Value", title = "Comparison of Different Models") +
  theme(legend.position = "top")

# Find the index of the model with the lowest MSE
best_model_index <- which.min(metrics_table[, "MSE"])

# Get the name of the best model
best_model_name <- model_names[best_model_index]

# Get the metrics of the best model
best_model_metrics <- metrics_table[best_model_index, ]

# Print the name and metrics of the best model
cat("Best Model on the basis of prediction accuracy:", best_model_name, "\n")
cat("MSE:", best_model_metrics["MSE"], "\n")
cat("MAE:", best_model_metrics["MAE"], "\n")
cat("RMSE:", best_model_metrics["RMSE"], "\n")
cat("R-squared:", best_model_metrics["R-squared"], "\n")
cat("Adjusted R-squared:", best_model_metrics["Adjusted R-squared"], "\n")

# Find the index of the model with the lowest R-squared
best_model_index <- which.min(metrics_table[, "R-squared"])

# Get the name of the best model
best_model_name <- model_names[best_model_index]

# Get the metrics of the best model
best_model_metrics <- metrics_table[best_model_index, ]

# Print the name and metrics of the best model
cat("Best Model on the basis of model fit:", best_model_name, "\n")
cat("MSE:", best_model_metrics["MSE"], "\n")
cat("MAE:", best_model_metrics["MAE"], "\n")
cat("RMSE:", best_model_metrics["RMSE"], "\n")
cat("R-squared:", best_model_metrics["R-squared"], "\n")
cat("Adjusted R-squared:", best_model_metrics["Adjusted R-squared"], "\n")
