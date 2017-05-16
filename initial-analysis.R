##Adding packages
##
library(ggplot2)
library(dplyr)
library(caret)
library(xgboost)
library(dummies)
library(stats)
library(nnet)
library(randomForest)
library(formattable)
library(gbm)
library(adabag)
##
##Read csv files
##
df_train <- read.csv("train_users_2.csv")
df_test_final <- read.csv("test_users.csv")
df_age_gender_bkts <- read.csv("age_gender_bkts.csv")
##
##Handle NA in age. Replace them with mean of age of same signup method
##
df_train$age[which(df_train$age > 100)] <- 0
mean_age_google <-
  round(mean(df_train$age[which(df_train$signup_method ==
                                  "google" &
                                  is.na(df_train$age) == FALSE)]), digits = 0)
mean_age_facebook <-
  round(mean(df_train$age[which(df_train$signup_method ==
                                  "facebook" &
                                  is.na(df_train$age) == FALSE)]), digits = 0)
mean_age_basic <-
  round(mean(df_train$age[which(df_train$signup_method ==
                                  "basic" &
                                  is.na(df_train$age) == FALSE)]), digits = 0)
##
##
df_train$age[which(is.na(df_train$age) &
                     df_train$signup_method == "google")] <-
  mean_age_google
##
df_train$age[which(is.na(df_train$age) &
                     df_train$signup_method == "facebook")] <-
  mean_age_facebook
##
df_train$age[which(is.na(df_train$age) &
                     df_train$signup_method == "basic")] <-
  mean_age_basic
##
df_train$age[which(df_train$age == 0 &
                     df_train$signup_method == "google")] <-
  mean_age_google
##
df_train$age[which(df_train$age == 0 &
                     df_train$signup_method == "facebook")] <-
  mean_age_facebook
##
df_train$age[which(df_train$age == 0 &
                     df_train$signup_method == "basic")] <-
  mean_age_basic
##
##Reconciling data types and cleaning
##
df_train$signup_flow <- as.factor(df_train$signup_flow)
df_train$date_account_created <-
  as.Date(df_train$date_account_created)
##
##Create column date_first_active from timestamp column
##
df_train$date_first_active <-
  gsub(" ", "", paste((
    substr(df_train$timestamp_first_active, 1, 4)
  ), "-",
  (
    substr(df_train$timestamp_first_active, 5, 6)
  ), "-",
  (
    substr(df_train$timestamp_first_active, 7, 8)
  )))
df_train$date_first_active <- as.Date(df_train$date_first_active)
##
##Handle missing date and tracking information
##
df_train$date_first_booking <-
  as.character(df_train$date_first_booking)
df_train$date_first_booking[which(df_train$date_first_booking == "")] <-
  "1970-01-01"
df_train$date_first_booking <-
  gsub(" ", "", df_train$date_first_booking)
df_train$date_first_booking <- as.Date(df_train$date_first_booking)
df_train$first_affiliate_tracked[which(df_train$first_affiliate_tracked == "")] <-
  "untracked"
##
##Fill up unknown first booking date with average values
##
df_time_diff <-
  data.frame(df_train$date_first_active[which(df_train$date_first_booking != "1970-01-01")],
             df_train$date_first_booking[which(df_train$date_first_booking != "1970-01-01")])
names(df_time_diff) <- c("first_active", "first_booking")
mean_time_diff <-
  round(mean(df_time_diff$first_booking - df_time_diff$first_active))
df_train$date_first_booking[which(df_train$date_first_booking == "1970-01-01")] <-
  df_train$date_first_active[which(df_train$date_first_booking == "1970-01-01")] + mean_time_diff
##
##Reformatting dataset
##
df_train$timeto_first_book <-
  df_train$date_first_booking - df_train$date_first_active
cols.drop <-
  c(
    "date_account_created",
    "timestamp_first_active",
    "date_first_booking",
    "date_first_active"
  )
df_train <- select(df_train, -one_of(cols.drop))
df_train$destination <-
  as.integer(interaction(df_train$country_destination, drop = TRUE)) - 1
df_train$timeto_first_book <- as.integer(df_train$timeto_first_book)
df_train$age <- as.integer(df_train$age)
##
##Sampling the data
##
set.seed(1)
train_index <-
  createDataPartition(df_train$country_destination, p = 0.70, list = FALSE)
df_train1 <- df_train[train_index,]
df_test <- df_train[-train_index,]
##
##Creating dummy variables
##
dummy.colnames <-
  c(
    "gender",
    "signup_method",
    "signup_flow",
    "language",
    "affiliate_channel",
    "affiliate_provider",
    "first_affiliate_tracked",
    "signup_app",
    "first_device_type",
    "first_browser"
  )
df_dum <-
  dummy.data.frame(
    data = df_train1,
    names = dummy.colnames,
    drop = TRUE,
    sep = "."
  )
##
##Multinomial Logistic Regression
##
set.seed(1)
df.multinom <-
  select(df_train1,-one_of(c("id", "country_destination")))
df.multinom$destination <- as.factor(df.multinom$destination)
model.multinom <-
  multinom(destination ~ .,
           data = df.multinom,
           MaxNWts = 2000,
           maxit = 1000)
##
##Predict for training data and compute confusion matrix
##
predicted.class.prob <-
  predict(model.multinom, newdata = df.multinom, type = "probs")
predicted_class <- rep(0, nrow(df.multinom))
for (i in 1:nrow(df.multinom)) {
  predicted_class[i] <- which.is.max(predicted.class.prob[i, ]) - 1
}
df.err.multinom <-
  data.frame(df_train1[, c("id", "country_destination", "destination")], predicted_class)
cm <-
  confusionMatrix(df.err.multinom$predicted_class,
                  df.err.multinom$destination)
cm
##
##Predict for test dataset and compute confusion matrix
##
df.multinom.test <-
  select(df_test,-one_of(c("id", "country_destination")))
df.multinom.test$destination <-
  as.factor(df.multinom.test$destination)
predicted.class.prob <-
  predict(model.multinom, newdata = df.multinom.test, type = "probs")
predicted_class <- rep(0, nrow(df.multinom.test))
for (i in 1:nrow(df.multinom.test)) {
  predicted_class[i] <- which.is.max(predicted.class.prob[i, ]) - 1
}
df.err.multinom <-
  data.frame(df_test[, c("id", "country_destination", "destination")], predicted_class)
cm <-
  confusionMatrix(df.err.multinom$predicted_class,
                  df.err.multinom$destination)
cm
##
##Boosting with AdaBoost
##
set.seed(1)
cols.exclude <- c("id", "country_destination")
df.ada <- select(df_train1,-one_of(cols.exclude))
df.ada$destination <- as.factor(df.ada$destination)
ada.model <- adabag::boosting(destination ~ ., data = df.ada, mfinal = 5, coeflearn = "Zhu", 
                              control = rpart.control(cp = 0.5))
##
##The X G Boosting !!!
##
set.seed(1)
cols.exclude <- c("destination", "id", "country_destination", "language", "affiliate_provider",
                  "signup_flow", "first_browser")
output_vector <- as.numeric(df_train1[, "destination"])
df.boost <-
  as.matrix(sapply(select(df_train1,-one_of(cols.exclude)), as.numeric))
no.of.classes <- as.numeric(length(unique(output_vector)))
param <- list(objective = "multi:softmax",
              eta = 0.15,
              gamma = 0.05,
              min_child_width = 4,
              num_class = no.of.classes,
              verbose = 1)
model.cv <- xgb.cv(data = df.boost, label = output_vector, params = param, nfold = 5, nrounds = 50)
model <-
  xgboost(
    data = df.boost,
    label = output_vector,
    params = param,
    nrounds = 100)
##
##Predicting and generating confusion matrix to evaluate on training data
##
predicted_class <- predict(model, df.boost)
df.error.calc <-
  data.frame(df_train1[, c("id", "country_destination", "destination")], predicted_class)
colnames(df.error.calc) <-
  c("id", "destination", "actual", "predicted")
cm1 <-
  confusionMatrix(df.error.calc$predicted, df.error.calc$actual)
cm1
##
##Predicting on test dataset
##
df.test.boost <-
  as.matrix(sapply(select(df_test,-one_of(cols.exclude)), as.numeric))
predicted_class <- predict(model, df.test.boost)
df.error.calc1 <-
  data.frame(df_test[, c("id", "country_destination", "destination")], predicted_class)
colnames(df.error.calc1) <-
  c("id", "destination", "actual", "predicted")
cm2 <-
  confusionMatrix(df.error.calc1$predicted, df.error.calc1$actual)
cm2
##
##Importance of predictors
##
imp.table.boost <- xgb.importance(colnames(df.boost), model = model)
imp.table <- data.frame(Feature = imp.table.boost$Feature,
                        Importance = round(imp.table.boost$Gain * 100, 2))
tab1 <- formattable(imp.table)
tab1
##
##Random Forests multiclass prediction
##
set.seed(1)
df.rf <- select(df_train1,-one_of(c("id", "country_destination")))
df.rf$first_browser <- as.character(df.rf$first_browser)
df.rf$first_browser[which(df.rf$first_browser == "-unknown-")] <-
  "Unknown"
df.rf$first_browser <- as.factor(df.rf$first_browser)
df.rf$destination <- as.factor(df.rf$destination)
df.rf.dum <- dummy.data.frame(df.rf, names = "first_browser")

rf.model <-
  randomForest(destination ~ . -language -affiliate_provider -signup_flow -first_browser, data = df.rf)
##
##Predict on training data and compute confusion matrix
##
rf.output <- predict(rf.model, newdata = df_train1)
cm <- confusionMatrix(rf.output, df_train1$destination)
cm
##
##Predicting and computing confusion matrix on test data
rf.output <- predict(rf.model, newdata = df_test)
cm <- confusionMatrix(rf.output, df_test$destination)
cm




