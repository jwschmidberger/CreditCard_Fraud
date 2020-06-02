
# title: "Model Development for Prediction of Creditcard Fraud"
# author: "Jason Schmidberger"
# date: "29/05/2020"


# The following script is a watered down version of my research project into the 2013 European creditcard
# fraud dataset problem. It only includes key scripts that are necessary to perform dataset download, 
# as well as all machine learning models culminating in the final model and a presentation of the table 
# of results. Much is missing here that is present in the report. All parameterisation screens are left 
# out, as well at the verbose discussions of results and conclusions. If you wish to see these, please 
# look at my report in pdf or Rmd format. 



### Here are the essential packages that are necessary to run this analysis of the creditcard dataset. 
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(Rborist)) install.packages("Rborist", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(DMwR)) install.packages("DMwR", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(MLmetrics)) install.packages("MLmetrics", repos = "http://cran.us.r-project.org")
if(!require(dMod)) install.packages("dMod", repos = "http://cran.us.r-project.org")


# Here is the code to download the dataset and open it as an object ccds. 
dl <- tempfile()
download.file("https://www.dropbox.com/s/rg6r68kdyual1wv/creditcard.csv.zip?dl=1", dl)
ccds <- read.csv(unzip(dl, "creditcard.csv"))
dim(ccds)

# I have introduced a new column in the dataset that was used in some data visualisation in the report,
# as well as inclusion in some machine learning models. This new column is a measure of the time that has
# passed from the time of the current transaction to the time of the previous transaction. 
ccds <- ccds %>%
  mutate(diff = Time - lag(Time))
# This process introduced one NA value for the first row, for which there is no previous row. This 
# should be converted to 0 for simplicity. 
ccds[is.na(ccds)] <- 0


# Time to divide the dataset into training and testing subsets. 90% and 10% respectively. 
set.seed(1, sample.kind = "Rejection")
test_index <- createDataPartition(y = ccds$Class, times = 1, p = 0.1, list = FALSE)
cc_train <- ccds[-test_index,]
cc_test <- ccds[test_index,]


# I need to set up some further data partitions for some basic cross validation. 
set.seed(1, sample.kind = "Rejection")
test_index2 <- createDataPartition(y = cc_train$Class, times = 1, p = 0.1, list = FALSE)
cc_train_wk <- ccds[-test_index2,]
cc_train_cv <- ccds[test_index2,]


# Introduction of an RMSE metric that is used at least for the regression modelling. 
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Here is the first model. It is the benchmark model used for comparative purposes. It is a linear 
# regression model. 
fit_lm <- lm(Class ~ ., data = cc_train_wk)
y_hat_lm <- predict(fit_lm, newdata = cc_train_cv)
y_hat_lm <- ifelse(y_hat_lm >= 0.07, 1,0)
acc_lm <- mean(y_hat_lm == cc_train_cv$Class)
rmse_lm <- RMSE(cc_train_cv$Class, y_hat_lm)
lm_tab <- table(y_hat_lm, cc_train_cv$Class)
rownames(lm_tab) <- c("Pred 0", "Pred 1")
colnames(lm_tab) <- c("Actual 0", "Actual 1")

# This initiates a table called model_results that contains all relevant statistics on each model.
# It begins involving an RMSE value that is not used in latter models. Also towards the end of the 
# project a new metric (CSL) is introduces and added to the table. 
model_results <- tibble(method = "linear regression", Accuracy = acc_lm, RMSE = rmse_lm, 
                       false_pos = lm_tab[2,1], false_neg = lm_tab[1,2])

# This is a modified linear regression model that has reduced number of predictors driving the model.
# The intention was to look at the effect of overtraining the model. A screen of threshold cut off 
# values was performed in the report but is left out here. The optimal cut off was 0.05.
fit_lm2 <- lm(Class ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V9 + V10 + V11 +
                V12 + V14 + V15 + V16 + V17 + V18, data = cc_train_wk)
y_hat_lm2 <- predict(fit_lm2, newdata = cc_train_cv)
y_hat_lm2 <- ifelse(y_hat_lm2 >= 0.05, 1,0)
acc_lm2 <- mean(y_hat_lm2 == cc_train_cv$Class)
rmse_lm2 <- RMSE(cc_train_cv$Class, y_hat_lm2)
lm_tab2 <- table(y_hat_lm2, cc_train_cv$Class)
rownames(lm_tab2) <- c("Pred 0", "Pred 1")
colnames(lm_tab2) <- c("Actual 0", "Actual 1")

model_results <- bind_rows(model_results,
                          data_frame(method = "LM lim Preds",
                                     Accuracy = acc_lm2, RMSE = rmse_lm2,
                                     false_pos = lm_tab2[2,1], false_neg = lm_tab2[1,2]))


# Here is the next type of model used. A logistic regression approach using all predictors. A screen of 
# threshold cut off values was performed in the report but is left out here. The optimal cut off was
# 0.07.
set.seed(1, sample.kind = "Rejection")
glm.fit <- glm(Class ~ ., data = cc_train_wk, family = binomial)
y_hat_log <- predict(glm.fit, newdata = cc_train_cv, type = "response")
y_hat_log <- ifelse(y_hat_log >= 0.07, 1,0)
acc_log <- mean(y_hat_log == cc_train_cv$Class)
log_tab <- table(y_hat_log, cc_train_cv$Class)
rmse_log <- RMSE(cc_train_cv$Class, y_hat_log)

model_results <- bind_rows(model_results,
                          data_frame(method = "Logistic Regression",
                                     Accuracy = acc_log, RMSE = rmse_log,
                                     false_pos = log_tab[2,1], false_neg = log_tab[1,2]))



# From this part forward, the response variable (Class) is changed to a two level factor for ease of use
# in subsequent classification models. 
cc_train_wk$Class <- as.factor(cc_train_wk$Class)
cc_train_cv$Class <- as.factor(cc_train_cv$Class)


# The first of the non-regression models. I am employing a random forest approach using Rborist. This 
# program is choses as the dataset is very large. A screen optimising both predFixed and minNode values
# was performed in the report but is left out here. The optimal values were 20 and 15 respectively. 
set.seed(1, sample.kind = "Rejection")
rf_1 <- Rborist(cc_train_wk[,c(1:30)], cc_train_wk$Class, predFixed = 20, minNode = 15)
rf_pred1 <- predict(rf_1, newdata = cc_train_cv[,c(1:30)])
acc_rf1 <- mean(rf_pred1$yPred == cc_train_cv$Class)
rf_tab1 <- table(rf_pred1$yPred, cc_train_cv$Class)
rmse_rf1 <- RMSE(cc_train_cv$Class, rf_pred1$yPred)
model_results <- bind_rows(model_results,
                          data_frame(method = "RF Rborist",
                                     Accuracy = acc_rf1, RMSE = rmse_rf1,
                                     false_pos = rf_tab1[2,1], false_neg = rf_tab1[1,2]))

# The imbalanced nature of the dataset was now addressed. 
# SMOTE was applied to balance the data. Preparing final SMOTE dataset. A lot of screening was performed 
# to find optimal seed values as well as perc.over and perc.under parameters. The optimal seed for 
# data balancing was 6, with perc.over = 400, and perc_under = 700. 
set.seed(6, sample.kind = "Rejection")
smt_final <- SMOTE(Class ~ ., data = cc_train_wk, perc.over = 400, k = 5, perc.under = 700,
                   learner=NULL)

# For comparative purposes, the same Rborist method applied to the whole dataset was applied to this 
# now balanced dataset. 
set.seed(1, sample.kind = "Rejection")
rf_2 <- Rborist(as.matrix(smt_final[,c(1:30)]), smt_final$Class, predFixed = 20, minNode = 15)
rf_pred2 <- predict(rf_2, newdata = cc_train_cv[,c(1:30)])
rf_tab2 <- table(rf_pred2$yPred, cc_train_cv$Class)
rownames(rf_tab2) <- c("Pred 0", "Pred 1")
colnames(rf_tab2) <- c("Actual 0", "Actual 1")




# The final model now employs randomForest. Having am much reduced dataset meant reduced computational 
# demands and the advantage of Rborist as being the more computationally efficient RF method no longer 
# matters as much. 
set.seed(2, sample.kind = "Rejection")
fit_RF <- randomForest(as.matrix(smt_final[,c(1:30)]), smt_final$Class, ntree = 30, mtry = 5)
y_hat <- predict(fit_RF, newdata = cc_train_cv[,c(1:30)])
f1 <- F1_Score(cc_train_cv$Class, y_hat, positive = 1)
F_pos <- table(y_hat, cc_train_cv$Class)[2,1]
F_neg <- table(y_hat, cc_train_cv$Class)[1,2]
CSLf <- (table(y_hat, cc_train_cv$Class)[2,1] + (400*(table(y_hat, cc_train_cv$Class)[1,2])))
Result <- c(f1, F_pos, F_neg, CSLf)


# Cost Sensitive Learning metrics have been applied with the SMOTE work above. Here I append and CSL 
# score onto previous model outcomes for reporting in the model_results table. 
model_results2 <- model_results %>% mutate(CSL = c(lm_tab[2,1] + (400*lm_tab[1,2]),
                                                 lm_tab2[2,1] + (400*lm_tab2[1,2]),
                                                 log_tab[2,1] + (400*log_tab[1,2]),
                                                 rf_tab1[2,1] + (400*rf_tab1[1,2])))

# Adding the stats of RF Rborist algorithm when applied to the SMOTE output. 
model_results2 <- bind_rows(model_results2,
                           data_frame(method = "Rborist with SMOTE",
                                      Accuracy = mean(rf_pred2$yPred == cc_train_cv$Class),
                                      false_pos = rf_tab2[2,1], 
                                      false_neg = rf_tab2[1,2],
                                      CSL = rf_tab2[2,1] + (400*rf_tab2[1,2])))
# Adding stats for the final model when applied to cc_train_cv. 
model_results2 <- bind_rows(model_results2,
                           data_frame(method = "Final Model vs cc_train_cv",
                                      Accuracy = mean(y_hat == cc_train_cv$Class),
                                      false_pos = F_pos, 
                                      false_neg = F_neg,
                                      CSL = CSLf))
model_results2 %>% knitr::kable()

# Application of the final method to the cc_test test dataset that has been kept aside until now. 
y_hat_t <- predict(fit_RF, newdata = cc_test[,c(1:30)])
f1_t <- F1_Score(cc_test$Class, y_hat_t, positive = 1)
F_pos_t <- table(y_hat_t, cc_test$Class)[2,1]
F_neg_t <- table(y_hat_t, cc_test$Class)[1,2]
CSL_t <- (table(y_hat_t, cc_test$Class)[2,1] + (400*(table(y_hat_t, cc_test$Class)[1,2])))
Result_test <- c(f1_t, F_pos_t, F_neg_t, CSL_t)




# Adding stats for the final model when applied to cc_test.
model_results2 <- bind_rows(model_results2,
                           data_frame(method = "Final Model vs cc_test",
                                      Accuracy = mean(y_hat_t == cc_test$Class),
                                      false_pos = F_pos_t, 
                                      false_neg = F_neg_t, CSL = CSL_t))

# This will product the summary statistics on all models, culminating with the final model tested
# against both cc_train_cv, and cc_test datasets. 
model_results2 %>% knitr::kable()










