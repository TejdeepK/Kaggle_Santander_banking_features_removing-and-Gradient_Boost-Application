setwd("~/Downloads/Santander_Banking_Kaggle")
#
orig.train <- read.csv("train.csv", stringsAsFactors = F)
orig.test <- read.csv("test.csv", stringsAsFactors = F)
sample.submission <- read.csv("sample_submission.csv", stringsAsFactors = F)

# ---------------------------------------------------
# Merge
orig.test$TARGET <- -1
merged <- rbind(orig.train, orig.test)
for(i in names(merged))
{
if (class(merged[[i]]) == "integer")
{
  print(i)
}
}

# ---------------------------------------------------
# Convert
feature.train.names <- names(orig.train)[-1]
for (f in feature.train.names) {
  if (class(merged[[f]]) == "numeric") {
    merged[[f]] <- merged[[f]] / max(merged[[f]])
  } else if (class(merged[[f]]) == "integer") {
    u <- unique(merged[[f]])
    if (length(u) == 1) {
      merged[[f]] <- NULL
    } else if (length(u) < 200) {
      merged[[f]] <- factor(merged[[f]])
    }
  }
}

train1 <- merged[merged$TARGET != -1, ]
test1 <- merged[merged$TARGET == -1, ]
dim(train1)
dim(test1)
### Reverse Engineering Principle###
#

ncol0 <- ncol(train1)
ncol1 <- ncol(test1)
### train1 #################################################
### Removing constant features 
cat("removing constant features\n")
toRemove <- c()
feature.names <- names(train1)
for (f in feature.names) 
{
  if (sd(train1[[f]])==0) 
  {
    toRemove <- c(toRemove,f)
    cat(f,"is constant\n")
  }
}
train1.names  <- setdiff(names(train1), toRemove)
train1        <- train1[,train1.names]

toRemove
cat("-------------------------\n")
library("caret")
lin.comb <- findLinearCombos(train1)
train1 <- train1[, -lin.comb$remove]
dim(train1)
library("caret")
lin.corr <- findCorrelation(cor(train1), cutoff = .999, verbose = FALSE)
lin.corr
lin.corr = sort(lin.corr)
train1 = train1[,-c(lin.corr)]
names(train1)
dim(train1)
removed <- ncol0-ncol(train1)
cat("\n ",removed," features have been removed\n")
#test1 removal of variables.
###
### Removing constant features 
cat("removing constant features\n")
toRemove <- c()
feature.names <- names(test1)
for (f in feature.names) 
{
  if (sd(test1[[f]])==0) 
  {
    toRemove <- c(toRemove,f)
    cat(f,"is constant\n")
  }
}
test1.names   <- setdiff(names(test1), toRemove)
test1         <- test1[,test1.names]
toRemove
cat("-------------------------\n")
library("caret")
lin.comb <- findLinearCombos(test1)
test1 <- test1[, -lin.comb$remove]
dim(test1)
library("caret")
lin.corr <- findCorrelation(cor(test1), cutoff = .999, verbose = FALSE)
lin.corr
lin.corr = sort(lin.corr)
test1 = test1[,-c(lin.corr)]
dim(test1)
removed <- ncol1-ncol(test1)
cat("\n ",removed," features have been removed\n")
dim(train1)
removed1 <- ncol0-ncol(train1)
cat("\n ",removed1," features have been removed\n")
###
library(gbm)
gbm1 <-
  gbm(TARGET ~. ,         # formula
      data=train1,                   # dataset
      var.monotone= NULL , # -1: monotone decrease,
      # +1: monotone increase,
      #  0: no monotone restrictions
      distribution="gaussian",     # see the help for other choices
      n.trees=100,                # number of trees
      shrinkage=0.05,              # shrinkage or learning rate,
      # 0.001 to 0.1 usually work
      interaction.depth=3,         # 1: additive model, 2: two-way interactions, etc.
      bag.fraction = 0.5,          # subsampling fraction, 0.5 is probably best
      train.fraction = 0.5,        # fraction of data for training,
      # first train.fraction*N used for training
      n.minobsinnode = 10,         # minimum total weight needed in each node
      cv.folds = 3,                # do 3-fold cross-validation
      keep.data=TRUE,              # keep a copy of the dataset with the object
      verbose=FALSE,               # don't print out progress
      n.cores=1) 
names(gbm1)
summary(gbm1)
gbmout = data.frame(summary(gbm1))
gbmfinal = gbmout[1:30,]
barplot(gbmfinal$rel.inf, names = gbmfinal$var,
        xlab = "Variables", ylab = "Relative Influence",
        main = "Plot of relative importance of var")
library(gbm)
#Method OOB
best.iter <- gbm.perf(gbm1,method="OOB")
print(best.iter)
#Method Test
best.iter1 <- gbm.perf(gbm1,method="test")
print(best.iter1)
#Method CV
best.iter2 <- gbm.perf(gbm1,method="cv")
print(best.iter2)
summary(gbm1,n.trees=1)         # based on the first tree
summary(gbm1,n.trees=best.iter)

