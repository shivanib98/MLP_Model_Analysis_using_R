# MLP MODEL ANALYSIS USING R
ML project: Applied MLP Model on NBA Games datasets for analysis &amp; comparison with regularization techniques.

1. Language Used - R 
2. MLP Library used - keras

<p align="justify">In this machine learning (ML) project, I conducted a comprehensive analysis and comparison of the Multilayer Perceptron (MLP) model on the NBA Games dataset. The primary focus of the project was to evaluate the performance of the MLP algorithm with and without the implementation of regularization techniques.</p>

Regularisation applied:
1. Ridge Regression (l2)
2. Lasso Regression (l1)
3. Elastic Net Regression (l1 & l2)

<p align="justify">I used the publicly accessible NBA Games dataset [https://www.kaggle.com/datasets/nathanlauga/nba-games], and prior to executing the data preprocessing code, an essential modification was applied to the `MIN` column using Microsoft Excel. Specifically, the date-time representation was skillfully transformed into a more coherent format expressed in minutes, fortifying the dataset for subsequent processing and comprehensive analysis. During data preprocessing, a stringent filter was employed to isolate rows corresponding solely to the esteemed player, LeBron James. Additionally, any records containing missing values (NAs) were diligently removed through the effective implementation of the `na.omit()` function. Moreover, several columns, such as "GAME_ID," "TEAM_ID," "TEAM_CITY," "REB," "PLAYER_ID," "PLAYER_NAME," "NICKNAME," and "COMMENT," were deemed superfluous to the analytical goals, thus warranting their exclusion from the dataset. Furthermore, the `TEAM_ABBREVIATION` and `START_POSITION` columns underwent a transformation from string-based representations to numeric values, facilitated by the judicious application of nested `ifelse()` statements. Upon the successful conclusion of data preprocessing, the resultant data frame was aptly preserved as a new CSV file entitled "Updatedgames_details," boasting an enriched dataset comprising 1681 rows and 21 columns.</p>

