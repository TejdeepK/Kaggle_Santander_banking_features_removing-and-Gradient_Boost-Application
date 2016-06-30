library(xgboost)
library(caret)
library(RCurl)
library(Metrics)
urlfile <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
x <- getURL(urlfile, ssl.verifypeer = FALSE)
adults <- read.csv(textConnection(x), header = FALSE)

names(adults)=c('age','workclass','fnlwgt','education','educationNum',
                'maritalStatus','occupation','relationship','race',
                'sex','capitalGain','capitalLoss','hoursWeek',
                'nativeCountry','income')
dim(adults)
adults$income <- ifelse(adults$income==' <=50K',0,1)
library(caret)
dmy <- dummyVars(" ~ .", data = adults)
adultsTrsf <- data.frame(predict(dmy, newdata = adults))
outcomeName <- c('income')
predictors <- names(adultsTrsf)[!names(adultsTrsf) %in% outcomeName]
nrow(adultsTrsf)
trainportion <- floor(nrow(adultsTrsf)*0.1)
trainset2 <- adultsTrsf[1:floor(trainportion/2),]
testset2 <- adultsTrsf[(floor(trainportion/2)+1):trainportion,]
smallesterror <- 100

boosting1 <- xgb.cv( data = as.matrix(trainset2[,predictors]), nrounds =400, nfold = 2, label = trainset2[,outcomeName],
                     showsd = TRUE, metrics = list(),
                    objective = "reg:linear", feval = NULL, 
                    verbose = T, print.every.n = 1L, early.stop.round = 100,
                    maximize = FALSE)
###############
for (depth in seq(1,10,1)) {
  for (rounds in seq(1,20,1)) {
    
    # train
    bst <- xgboost(data = as.matrix(trainset2[,predictors]),
                   label = trainset2[,outcomeName],
                   max.depth=depth, nround=rounds,
                   objective = "reg:linear", verbose=0)
    gc()
    
    # predict
    predictions <- predict(bst, as.matrix(testset2[,predictors]), outputmargin=TRUE)
    err <- rmse(as.numeric(testset2[,outcomeName]), as.numeric(predictions))
    
    if (err < smallesterror) {
      smallestError = err
      print(paste(depth,rounds,err))
    }     
  }
}  
depth1 = 4
rounds1 = 50
bst <- xgboost(data = as.matrix(trainset2[,predictors]),
               label = trainset2[,outcomeName],
               max.depth=depth1, nround=rounds1,
               objective = "reg:linear", verbose=0)
modelboost <-  xgb.dump(model = bst,with.stats = TRUE)
modelboost[1:10]
opted_names <- dimnames(as.matrix(trainset2[,predictors]))[[2]]
importance_matrix<- xgb.importance(opted_names,model = bst)
xgb.plot.importance(importance_matrix[1:15,])
xgb.plot.tree(feature_names = opted_names,model = bst,n_first_tree = 2)
adults <-read.csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header=F)
########
require(xgboost)
data(agaricus.train, package='xgboost') 
data(agaricus.test, package='xgboost') 
train = agaricus.train
test = agaricus.test