#NBA Games data preprocessing

library(tidyverse)
# Pleas uncomment the below line and specify the file path 
#file_path <-"path/games_details.csv"
# Read the CSV file
df <- read.csv(file_path)

# Print the number of rows and columns
num_rows <- nrow(df)
num_cols <- ncol(df)
cat("Number of rows:", num_rows, "\n")
cat("Number of columns:", num_cols, "\n")

# Filter rows with PLAYER_NAME as "LeBron James"
df<- subset(df, PLAYER_NAME == "LeBron James")

# Print the number of rows and columns after filtering
num_rows <- nrow(df)
num_cols <- ncol(df)
cat("Number of rows after filtering:", num_rows, "\n")
cat("Number of columns after filtering:", num_cols, "\n")

# Remove rows with missing values (NAs)
df <- na.omit(df)

#Display the dataset
view(df)

#Drop the mentioned columns
df <- df[, !(names(df) %in% c("GAME_ID","TEAM_ID","TEAM_CITY","REB",
                              "PLAYER_ID","PLAYER_NAME","NICKNAME","COMMENT"))]

#Converting the TEAM_ABBREVIATION column from String to numeric
df$TEAM_ABBREVIATION<- ifelse(df$TEAM_ABBREVIATION == "CLE", 1,
                              ifelse(df$TEAM_ABBREVIATION == "MIA", 2,
                                     ifelse(df$TEAM_ABBREVIATION == "LAL", 3,
                                            NA)))

#Converting the START_POSITION column from Character to numeric
df$START_POSITION <- ifelse(df$START_POSITION == "C", 1,
                            ifelse(df$START_POSITION == "F", 2,
                                   ifelse(df$START_POSITION == "G", 3,
                                   NA)))
#Dropping rows with START_POSITION = NA
df <- subset(df, START_POSITION != "NA")

# Display the structure of the data frame
str(df)
summary(df)

# Print the number of rows and columns after removing outliers
num_rows <- nrow(df)
num_cols <- ncol(df)
cat("Number of rows after data processing:", num_rows, "\n")
cat("Number of columns after data processing:", num_cols, "\n")

view(df)

# Specify the file path and name for saving the CSV file
output_file <- "path/Updatedgames_details.csv"

# Save the dataframe as a CSV file
write.csv(df, file = output_file,row.names = FALSE)

# Print a message indicating successful saving of the CSV file
cat("Dataframe saved as CSV:", output_file, "\n")
