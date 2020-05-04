#clear the r environment
rm(list=ls())

#setting the working directory
setwd("D:/Gaikwad/Data/Live Project 1/Customer transaction prediction/path")

#verifying the directory
getwd()

#loading the train and test data
cust_train = read.csv("train.csv", header = T , stringsAsFactors = F)
cust_test = read.csv("test.csv", header = T , stringsAsFactors = F)

#after loading train data we have 2 lac observations and 202 variables
#out of 202 variables 201 are independent and 1 is dependent i.e target variable
#in test data we have 2 lac observations and 201 variables excluding target column

#let's look at names of the 2 datasets
names(cust_train)
names(cust_test)

#Explore the data
str(cust_train)
str(cust_test)

summary(cust_train)
summary(cust_test)

#make a copy of data
df = cust_train
df1 = cust_test


#lets convert the target int type to factor
df$target= as.factor(df$target)

#check the structure of data
str(df)

#lets check the count of target classes
require(gridExtra)
table(df$target)

#percentage count of target classes
table(df$target)/length(df$target)*100

#barplot of count of target classes
library(ggplot2)
plot1 = ggplot(df,aes(target))+theme_bw()+geom_bar(stat='count',fill='green')

#violin with jitterplots of target classes
plot2=ggplot(df, aes(x= target, y= 1:nrow(df)))+ theme_bw()+ geom_violin(fill = 'blue')+facet_grid(df$target)+geom_jitter(width = 0.02)+ labs(y='Index')
grid.arrange(plot1, plot2, ncol=2)

#histogram of train data
df$target= as.numeric(df$target)
hist(df$target , col = "blue")

#Loading the library
library(DMwR)
library(rpart)
library(ggplot2)
library(corrgram)
library(ggpubr)
library(C50)
library(MASS)
library(dplyr)
library(RODBC)
library(outliers)
library(car)
library(Boruta)
library(Metrics)
library(randomForest)
library(ggthemes)

#distribution of mean values per row and column in train and test dataset
#Applying the function to find mean values per row in train and test data.
df_mean = apply(df[,-c(1,2)],MARGIN=1,FUN=mean)
df1_mean = apply(df1[,-c(1)],MARGIN=1,FUN=mean)
ggplot()+
  #Distribution of mean values per row in train data
  geom_density(data=df[,-c(1,2)],aes(x=df_mean),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
  #Distribution of mean values per row in test data
  geom_density(data=df1[,-c(1)],aes(x=df1_mean),kernel='gaussian',show.legend=TRUE,color='green')+
  labs(x='mean values per row',title="Distribution of mean values per row in train and test dataset")

#Applying the function to find mean values per column in train and test data.
df_mean = apply(df[,-c(1,2)],MARGIN=2,FUN=mean)
df1_mean = apply(df1[,-c(1)],MARGIN=2,FUN=mean)
ggplot()+
  #Distribution of mean values per column in train data
  geom_density(aes(x=df_mean),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
  #Distribution of mean values per column in test data
  geom_density(aes(x=df1_mean),kernel='gaussian',show.legend=TRUE,color='green')+
  labs(x='mean values per column',title="Distribution of mean values per row in train and test dataset")

#distribution of standard deviation values per row and column in train and test dataset
#Applying the function to find standard deviation values per row in train and test data.
df_sd = apply(df[,-c(1,2)],MARGIN=1,FUN=sd)
df1_sd = apply(df1[,-c(1)],MARGIN=1,FUN=sd)
ggplot()+
  #Distribution of sd values per row in train data
  geom_density(data=df[,-c(1,2)],aes(x=df_sd),kernel='gaussian',show.legend=TRUE,color='red')+theme_classic()+
  #Distribution of mean values per row in test data
  geom_density(data=df1[,-c(1)],aes(x=df1_sd),kernel='gaussian',show.legend=TRUE,color='blue')+
  labs(x='sd values per row',title="Distribution of sd values per row in train and test dataset")

#Applying the function to find sd values per column in train and test data.
df_sd = apply(df[,-c(1,2)],MARGIN=2,FUN=sd)
df1_sd = apply(df1[,-c(1)],MARGIN=2,FUN=sd)
ggplot()+
  #Distribution of sd values per column in train data
  geom_density(aes(x=df_sd),kernel='gaussian',show.legend=TRUE,color='red')+theme_classic()+
  #Distribution of sd values per column in test data
  geom_density(aes(x=df1_sd),kernel='gaussian',show.legend=TRUE,color='blue')+
  labs(x='sd values per column',title="Distribution of std values per column in train and test dataset")

#distribution of kurtosis values per row and column in train and test dataset
#Applying the function to find kurtosis values per row in train and test data.
library(e1071)
df_kurtosis = apply(df[,-c(1,2)],MARGIN=1,FUN=kurtosis)
df1_kurtosis = apply(df1[,-c(1)],MARGIN=1,FUN=kurtosis)
ggplot()+
  #Distribution of sd values per column in train data
  geom_density(aes(x=df_kurtosis),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
  #Distribution of sd values per column in test data
  geom_density(aes(x=df1_kurtosis),kernel='gaussian',show.legend=TRUE,color='red')+
  labs(x='kurtosis values per row',title="Distribution of kurtosis values per row in train and test dataset")

#Applying the function to find kurtosis values per column in train and test data.
df_kurtosis = apply(df[,-c(1,2)],MARGIN=2,FUN=kurtosis)
df1_kurtosis = apply(df1[,-c(1)],MARGIN=2,FUN=kurtosis)
ggplot()+
  #Distribution of sd values per column in train data
  geom_density(aes(x=df_kurtosis),kernel='gaussian',show.legend=TRUE,color='blue')+theme_classic()+
  #Distribution of sd values per column in test data
  geom_density(aes(x=df1_kurtosis),kernel='gaussian',show.legend=TRUE,color='red')+
  labs(x='kurtosis values per column',title="Distribution of kurtosis values per column in train and test dataset")

#distribution of skewness values per row and column in train and test dataset

#Applying the function to find skewness values per row in train and test data.
df_skew = apply(df[,-c(1,2)],MARGIN=1,FUN=skewness)
df1_skew = apply(df1[,-c(1)],MARGIN=1,FUN=skewness)
ggplot()+
  #Distribution of skewness values per row in train data
  geom_density(aes(x=df_skew),kernel='gaussian',show.legend=TRUE,color='green')+theme_classic()+
  #Distribution of skewness values per column in test data
  geom_density(aes(x=df1_skew),kernel='gaussian',show.legend=TRUE,color='blue')+
  labs(x='skewness values per row',title="Distribution of skewness values per row in train and test dataset")

#Applying the function to find skewness values per column in train and test data.
df_skew = apply(df[,-c(1,2)],MARGIN=2,FUN=skewness)
df1_skew = apply(df1[,-c(1)],MARGIN=2,FUN=skewness)
ggplot()+
  #Distribution of skewness values per column in train data
  geom_density(aes(x=df_skew),kernel='gaussian',show.legend=TRUE,color='green')+theme_classic()+
  #Distribution of skewness values per column in test data
  geom_density(aes(x=df1_skew),kernel='gaussian',show.legend=TRUE,color='blue')+
  labs(x='skewness values per column',title="Distribution of skewness values per column in train and test dataset")

############Missing value analysis######################

#Finding the missing values in train data
missing_val = data.frame(missing_val=apply(df,2,function(x){sum(is.na(x))}))
missing_val = sum(missing_val)
missing_val

#Finding the missing values in test data
missing_val = data.frame(missing_val=apply(df1,2,function(x){sum(is.na(x))}))
missing_val = sum(missing_val)
missing_val

#Now, we proceed to look for outliers, using cooks Distance
#first we build the cooks distance model 
#removing variable "ID_Code" from train and test dataset
df = df[, -1, drop = FALSE]
df1 = df[, -1, drop = FALSE]

df[] = lapply(df[], as.numeric)

mod = lm(target~., data = df)
cooksd = cooks.distance(mod)

plot(cooksd, pch = "*", cex = 2, main = "Effective points")
abline(h = 4*mean(cooksd, na.rm = T), col = "red")
text(x=1:length(cooksd)+1, y=cooksd, labels = ifelse(cooksd >4*mean(cooksd, na.rm = T), names(cooksd), ""), col = "red")
influential = as.numeric(names(cooksd)[(cooksd >4*mean(cooksd, na.rm = T))])
head(df[influential,])

#we perform outliers test using car package 
car:: outlierTest(mod)

#it show that 94184 in this row has the most extreme values 

#imputation of the outliers we are using the capping function
x = as.data.frame(df)
caps = data.frame(apply(df,2, function(x){
  quantiles <- quantile(x, c(0.25, 0.75))
  x[x < quantiles[1]] <- quantiles[1]
  x[x > quantiles [2]] <- quantiles[2]
}))

caps

#########################Correlation analysis#########################
#Correlation in train data
#convert factor to int
df$target = as.numeric(df$target)
df_correlations = cor(df[,c(1:201)])
df_correlations

#Correlation in test data
df1_correlations = cor(df1[,c(1:200)])
df1_correlations

#We can observe that the correlation between the test attributes is very small
###############Feature engineering######################
########variable imortance############
#variable importance is used to see top features in the dataset
#lets build a simple model to find important features
#Split the training data using simple random sampling
df_index = sample(1:nrow(df),0.75*nrow(df))
#train data
train_data = df[df_index,]
#validation data
valid_data = df[-df_index,]
#dimension of train and validation data
dim(train_data)
dim(valid_data)

#Random forest classifier

#Training the Random forest classifier
set.seed(2732)
#convert to int to factor
train_data$target = as.factor(train_data$target)
#setting the mtry
mtry<-floor(sqrt(200))
#setting the tunegrid
tuneGrid<-expand.grid(.mtry=mtry)
#fitting the random forest
rf = randomForest(target~.,train_data[,-c(1)],mtry=mtry,ntree=10,importance=TRUE)

#Feature importance by random forest

#Variable importance
VarImp = importance(rf,type=2)
VarImp

#we can observe that by variable importance var_12,var_22,var_26,var_53,var_81,var_109,var_110,var_139,var_146,var_166,var_174,var_198 are higher and top most important features
#Partial dependence plots
#Let us see impact of the main features which are discovered in the previous section by using pdp package
#We will plot "var_13"
library(pdp)
par.var_13 = partial(rf, pred.var = c("var_13"), chull = TRUE)
plot.var_13 = autoplot(par.var_13, contour = TRUE)
plot.var_13

#We will plot "var_12"
par.var_12 = partial(rf, pred.var = c("var_12"), chull = TRUE)
plot.var_12 = autoplot(par.var_12, contour = TRUE)
plot.var_12

#simple Logistic regression model.

#Split the data using CreateDataPartition
set.seed(689)

train.index = sample(1:nrow(train_df),0.8*nrow(train_df))
#train data
train.data = train_df[train.index,]
#validation data
valid.data = train_df[-train.index,]
#dimension of train data
dim(train.data)
#dimension of validation data
dim(valid.data)
#target classes in train data
table(train.data$target)
#target classes in validation data
table(valid.data$target)

#Logistic Regression model

#Training dataset
X_t = as.matrix(train.data[,-c(1,2)])
y_t = as.matrix(train.data$target)
#validation dataset
X_v = as.matrix(valid.data[,-c(1,2)])
y_v = as.matrix(valid.data$target)
#test dataset
test = as.matrix(test_df[,-c(1)])
#Logistic regression model
set.seed(667) # to reproduce results
lr_model  = glmnet(X_t,y_t, family = "binomial")
summary(lr_model)

#Cross validation prediction
set.seed(8909)
cv_lr = cv.glmnet(X_t,y_t,family = "binomial", type.measure = "class")
cv_lr

#Plotting the missclassification error vs log(lambda) where lambda is regularization parameter

#Minimum lambda
cv_lr$lambda.min
#plot the auc score vs log(lambda)
plot(cv_lr)

#We can observe that miss classification error increases as increasing the log(Lambda).


#Model performance on validation dataset
set.seed(5363)
cv_predict.lr = predict(cv_lr,X_v,s = "lambda.min", type = "class")
cv_predict.lr

#Accuracy of the model is not the best metric to use when evaluating the imbalanced datasets as it may be misleading. So, we are going to change the performance metric.

#Confusion matrix
set.seed(689)
#actual target variable
target = valid.data$target
#convert to factor
target = as.factor(target)
#predicted target variable
#convert to factor
cv_predict.lr = as.factor(cv_predict.lr)
confusionMatrix(data=cv_predict.lr,reference=target)

#Reciever operating characteristics(ROC)-Area under curve(AUC) score and curve

#ROC_AUC score and curve
set.seed(892)
cv_predict.lr = as.numeric(cv_predict.lr)
roc(data=valid.data[,-c(1,2)],response=target,predictor=cv_predict.lr,auc=TRUE,plot=TRUE)

#Random Oversampling Examples(ROSE)

#It creates a sample of synthetic data by enlarging the features space of minority and majority class examples.

#Random Oversampling Examples(ROSE)
set.seed(699)
train.rose = ROSE(target~., data =train.data[,-c(1)],seed=32)$data
#target classes in balanced train data
table(train.rose$target)
valid.rose = ROSE(target~., data =valid.data[,-c(1)],seed=42)$data
#target classes in balanced valid data
table(valid.rose$target)

#Let us see how baseline logistic regression model performs on synthetic data points.

#Logistic regression model
set.seed(462)
lr_rose = glmnet(as.matrix(train.rose),as.matrix(train.rose$target), family = "binomial")
summary(lr_rose)

#Cross validation prediction
set.seed(473)
cv_rose = cv.glmnet(as.matrix(valid.rose),as.matrix(valid.rose$target),family = "binomial", type.measure = "class")
cv_rose

#We can observe that ROSE model is performing well on imbalance data compare to baseline logistic regression.