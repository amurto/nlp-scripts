# Load csv
dat = read.csv("adult.csv", header = TRUE)

# dat =dat[1:100,]

# Omit last column
data = dat[,1:14]

# Drop columns
data = subset(data, select=-c(education,fnlwgt,race))

# Add new column
data["isprofit"] <- NA

# Update values of new column
data <- within(data, isprofit[capital.gain > 0] <- 1)
data <- within(data, isprofit[capital.gain == 0] <- 0)
data <- within(data, isprofit[capital.loss > 0] <- -1)

# Drop columns
data = subset(data, select=-c(capital.gain,capital.loss))

# Add new column
data["age_set"] <- NA

# Update values of new column
data <- within(data, age_set[age <= 20] <- 1)
data <- within(data, age_set[age > 20] <- 2)
data <- within(data, age_set[age <= 60] <- 2)
data <- within(data, age_set[age > 60] <- 3)

# Drop column
data = subset(data, select=-c(age))

# Add column
data["hours_set"] <- NA

# Update values of new column
data <- within(data, hours_set[hours.per.week <= 20] <- 1)
data <- within(data, hours_set[hours.per.week > 20] <- 2)
data <- within(data, hours_set[hours.per.week <= 50] <- 2)
data <- within(data, hours_set[hours.per.week > 50] <- 3)

# Drop column
data = subset(data, select=-c(hours.per.week))

# Install package dummies for creating dummy variables
install.packages("dummies")
library(dummies)

# Rename Column
colnames(data)[1] <- "carrier"

# Create a dataframe with dummy variables
X_onehot <- dummy.data.frame(data, names = c("carrier","educational.num","marital.status","occupation","relationship","gender","native.country","isprofit","age_set","hours_set") , sep = ".")

# Add new column
dat["incomebin"] <- NA

# Update column values
dat <- within(dat, incomebin[income == '<=50K'] <- 0)
dat <- within(dat, incomebin[income == '>50K'] <- 1)

# Create a dataframe with last column
Y <- dat[16]

install.packages("gtools")
library(gtools)

ds <- X_onehot
ds["incomebin"] <- dat[16]

library(caTools)

# creates a value for dividing the data into train and test. In this case the value is defined as 80% of the number of rows in the dataset
smp_siz = floor(0.80*nrow(ds))  

# set seed to ensure you always have same random numbers generated
set.seed(123)      

# Randomly identifies therows equal to sample size ( defined in previous instruction) from  all the rows of Smarket dataset and stores the row number in train_ind
train_ind = sample(seq_len(nrow(ds)),size = smp_siz)

#creates the training dataset with row numbers stored in train_ind
train =ds[train_ind,]

# creates the test dataset excluding the row numbers mentioned in train_ind
test =ds[-train_ind,]

library (data.table)
install.packages("plyr")
library (plyr)
library (stringr)
install.packages("caret")
library(caret)
install.packages("ROCR")
library(ROCR)

model <- glm(incomebin ~.,family=binomial(link='logit'),data=train)
summary(model)
anova(model, test = 'Chisq')

log_predict <- predict(model,newdata = test,type = "response")
log_predict <- ifelse(log_predict > 0.5,1,0)
pr <- prediction(log_predict,test$incomebin)

perf <- performance(pr,measure = "tpr",x.measure = "fpr")
plot(perf)

auc(test$incomebin,log_predict)

misClasificError <- mean(log_predict != test$incomebin)
print(paste('Accuracy',1-misClasificError))


