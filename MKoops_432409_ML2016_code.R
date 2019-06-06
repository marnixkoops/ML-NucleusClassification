# Load Libraries
library(ggplot2)
library(data.table)
library(cluster)
library(fpc)
library(plyr)
library(factoextra)
library(FactoMineR)
library(party)
library(partykit)
library(randomForest)
library(rpart)
library(rpart.plot)
library(fclust)
library(clustrd)
library(corrplot)
library(e1071)
library(caret)
library(xgboost)
library(caTools)
library(pROC)
library(SVMMaj)
library(ada)

#_______________________________________#
#   DATA                                #
#---------------------------------------#

setwd("~/Documents/Studie/Master Econometrie/Machine Learning/Individual Assignment/Breast Cancer")
bcdata <-read.csv("bc_wisconsin_data.csv") # Read datafile
bcdata$id <- NULL # Remove ID column from dataset
bcdata$X <- NULL # Remove last column of dataset (empty column)

# Prepare data set for predictions
set.seed(999)
bcdata2 <- bcdata # use new dataframe for predictions
bcdata2$diagnosis <- as.factor(bcdata2$diagnosis)
perm.bcdata <-bcdata2[sample(nrow(bcdata2)),] # Randomly permute entire dataset
index <- sample(seq(1, 2),size = nrow(bcdata2), replace = TRUE, prob = c(0.7, 0.3)) # Split rougly 70/30
bc.train <- bcdata2[index == 1,] # Create training set
bc.test <- bcdata2[index == 2,] # Create testing set

# Some data plotting
ggplot(bcdata, aes(x = factor(""), fill = diagnosis)) + geom_bar() +
   coord_polar(theta = "y") + scale_x_discrete("")
ggplot(data = bcdata) +
  geom_point(mapping = aes(x=concave.points_worst, y=area_worst, color = diagnosis))
ggplot(data = bcdata) +
  geom_point(mapping = aes(x=concave.points_worst, y=concavity_se, color = diagnosis))
ggplot(data = bcdata) +
  geom_point(mapping = aes(x=area_worst, y=concavity_se, color = diagnosis))

#_______________________________________#
#   K-MEANS CLUSTERING                  #
#---------------------------------------#
# K - Means #
bc_kmeans <-
  kmeans(bcdata[, c(2:31)], centers = 2, nstart = 50) # Create clusters with K-Means
table(bc_kmeans$cluster, bcdata$diagnosis) # Confusion matrix for K-Means

#png(file="kmeans.png", units="in", width=3.8, height=2.4, res=300) # Export plot
fviz_cluster( # plot K-means clustering
  list(data = bcdata[, c(2:31)], cluster = bc_kmeans$cluster),
  geom = c("point"),
  # Plot clustering
  repel = TRUE,
  show.clust.cent = FALSE,
  frame = TRUE,
  frame.type = "convex",
  frame.level = 0.95,
  frame.alpha = 0.25,
  pointsize = 1,
  title = "K-Means")

# PCA #
bc.pca <- prcomp(bcdata[, c(2:31)], center = TRUE, scale = TRUE) # Calc PCA's
summary(bc.pca) # See results
fviz_screeplot(bc.pca, ncp = 10) # Scree plot of first 10 PCA's
fviz_pca_contrib(bc.pca, choice = "var", axes = 1:3) # Check variable contribution to PCA 1 and 2

# Correlaton plot #
#png(file="corrplot.png", units="in", width=3.8, height=2.4, res=300) # Export plot
bcdata.cor <- cor(bcdata[, c(2:4, 22:24)]) # Calculate correlations
corrplot(bcdata.cor,   # Plot correlogram
  method = "circle",
  type = "lower",
  diag = FALSE,
  addCoef.col = "white",
  addCoefasPercent = TRUE,
  tl.col = "black",
  tl.srt = 40,
  tl.cex = 0.8,
  number.cex = 0.8)

# Factorial K-Means #
bc_fkm <- cluspca(     # Calculate FKM
    data = bcdata[, c(2:31)], nclus=2, ndim=3,
    alpha = NULL,
    method = "FKM",
    center = TRUE,
    scale = TRUE,
    rotation = "none",
    nstart = 50,
    smartStart = NULL,
    seed = 999)
table(bc_fkm$cluID, bcdata$diagnosis) # Confusion matrix for FKM

#png(file="FKM.png", units="in", width=3.8, height=2.4, res=300) # Export plot
fviz_cluster( list(data = bcdata[, c(2:31)],  # Plot FKM clustering
  cluster = bc_fkm$cluID),
  geom = c("point"),
  repel = TRUE,
  show.clust.cent = FALSE,
  frame = TRUE,
  frame.type = "convex",
  frame.level = 0.95,
  frame.alpha = 0.25,
  pointsize = 1,
  title = "Factorial K-Means")

apply(bcdata[, c(2:31)], 2, var) # Calculate variance of variables

# Reduced K-Means #
bc_rkm <- cluspca(data = bcdata[, c(2:31)], nclus=2, ndim=3, # dimensionality 3
    alpha = NULL,
    method = "RKM",
    # Calculate RKM
    center = TRUE,
    scale = TRUE,
    rotation = "none",
    nstart = 50,
    smartStart = NULL,
    seed = 999)
table(bc_rkm$cluID, bcdata$diagnosis) # Confusion matrix for RKM

#png(file="RKM.png", units="in", width=3.8, height=2.4, res=300) # Export plot
fviz_cluster( list(data = bcdata[, c(2:31)],  # Plot RKM clustering
  cluster = bc_rkm$cluID),
  geom = c("point"),
  repel = TRUE,
  show.clust.cent = FALSE,
  frame = TRUE,
  frame.type = "convex",
  frame.level = 0.95,
  frame.alpha = 0.25,
  pointsize = 1,
  title = "Reduced K-Means")

#_______________________________________#
#   DECISION TREES                      #
#---------------------------------------#

#tree<-rpart(diagnosis~ ., data = bcdata[,c(1:31)]) # Cart tree
#rpart.plot(tree)
#printcp(tree)

#cond_tree <- party::ctree(formula = diagnosis~.,data = bcdata[,c(2:31)]) # Conditional tree
#plot(cond_tree)

#noprune<-rpart.control(minsplit = 2, minbucket = 1, cp = 0) # Control for no pruning
#rtree.fit<-rpart(formula = diagnosis~.,data = bcdata, control=noprune) # Unpruned Cart tre
#rpart.plot(rtree.fit)

#_______________________________________#
#   TREE BASED SUPERVISED LEARNING      #
#---------------------------------------#

# PREDICTIONS #
# Conditional Tree predictions 10x CV#
#rpart.fit<-rpart(formula = diagnosis~. ,data=bc.train) # CART Tree
cvfolds <-createMultiFolds(bc.train$diagnosis, k = 10, times = 2) # Create stratified sample for cv
cv.ctrl <- # Control object for CV
  trainControl(
    method = "repeatedcv",
    number = 10,
    repeats = 2,
    index = cvfolds)
ctree.fit <- # Fit model
  train(
    as.factor(diagnosis) ~ . ,
    data = bc.train,
    method = "ctree",
    trControl = cv.ctrl) # Conditional Tree
ctree.pred <-predict(ctree.fit, newdata = bc.test) # Predict using single tree
confusionMatrix(ctree.pred, bc.test$diagnosis, positive = "M") # Check results and set M as positive class
ctree.fit # check cv results
ctree.fit$finalModel # See tree used
# Random Forest pedictions #
#rf.fit <- train(as.factor(diagnosis) ~. , data=bc.train, importance=TRUE, ntree=500, method="rf") # Calculate RF
#rf.pred<-predict(rf.fit, newdata = bc.test, method = "rf", mtry=3) # Make predictions
#confusionMatrix(rf.pred, bc.test$diagnosis, positive="M") # Check results and set M as positive class
#rf.fit$finalModel # Check OOB error

# Random Forest predictions 10x CV#
cvfolds <-createMultiFolds(bc.train$diagnosis, k = 10, times = 2) # Create stratified sample for cv
ctrl <- # control object for CV
  trainControl(
    method = "repeatedcv",
    number = 10,
    repeats = 2,
    index = cvfolds,
    savePred = TRUE,
    classProbs = TRUE) # control object for cv
set.seed(99)
rf.fit <- # Fit model
  train(
    as.factor(diagnosis) ~ .,
    data = bc.train,
    importance = TRUE,
    ntree = 500,
    method = "rf",
    trControl = ctrl)
rf.pred <- predict(rf.fit, newdata = bc.test) # Make predictions
confusionMatrix(rf.pred, bc.test$diagnosis, positive = "M") # Check results and set M as positive class
rf.fit$finalModel # Check OOB error

# XGBoost predictions 10x CV#
#bc.xgb <- bc.train
#bc.test.xgb <- bc.test
#bc.xgb$diagnosis <- ifelse(bc.xgb$diagnosis == 'M', 1, 0) # Convert diagnosis to M=1 B=0
#bc.test.xgb$diagnosis <- ifelse(bc.test.xgb$diagnosis == 'M', 1, 0)
#xgb.fit <- xgboost(data = as.matrix(bc.xgb), label=as.matrix(bc.xgb$diagnosis), # train model
#                     nrounds = 10, prediction=TRUE)
#xgb.pred <- predict(xgb.fit, newdata=as.matrix(bc.test.xgb)) # Make predictions
#xgb.pred <- round(xgb.pred, digits = 0) # round chances (since even for classification it does not return class but prob.)
#confusionMatrix(round(xgb.pred, digits = 0), bc.test.xgb$diagnosis, positive = '1')# Check results

# XGBoost 10x CV #
set.seed(99)
cvfolds <- createMultiFolds(bc.train$diagnosis, k = 10, times = 2) # Create stratified sample for cv
cv.ctrl <- # control object for CV
  trainControl(
    method = "repeatedcv",
    repeats = 2,
    number = 10,
    summaryFunction = twoClassSummary,
    classProbs = TRUE,
    allowParallel = T,
    savePred = TRUE)
#xgb.grid <- expand.grid(nrounds = 1000, eta = c(0.1, 0.01, 0.001, 0.0001), max_depth = c(2, 4, 6, 8, 10),
#gamma = c(0, 0.5, 1), colsample_bytree=0.4, min_child_weight=1, subsample=1)
#xgb.grid <- expand.grid(nrounds=150, max_depth=2, eta=0.3, gamma=0, colsample_bytree=0.8, min_child_weight=1, subsample=0.5)
xgb.fit <- # Fit model
  train((diagnosis) ~ .,
        data = bc.train,
        method = "xgbTree",
        trControl = cv.ctrl,
        verbose = T,
        nthread = 3)
xgb.fit # See results of model training
xgb.pred <- predict(xgb.fit , newdata = bc.test) # Make predictions
confusionMatrix(xgb.pred, bc.test$diagnosis, positive = "M") # Check predictions and set M as positive class

# Adaboost 10x CV #
#ada.fit <- adaboost(diagnosis~., bc.train, 500)
#ada.pred <- predict(ada.fit ,newdata=bc.test) # Make predictions
#confusionMatrix(ada.pred$class, bc.test$diagnosis, positive="M") # Check predictions and set M as positive class
cvfolds <- createMultiFolds(bc.train$diagnosis, k = 10, times = 2) # Create stratified sample for cv
ctrl <- # control object for CV
  trainControl(
    method = "repeatedcv",
    number = 10,
    repeats = 2,
    index = cvfolds,
    savePred = TRUE)
ada.grid <- expand.grid(nu = 0.1, iter = 150, maxdepth = 3)
ada.fit <- # Fit model
  train(
    diagnosis ~ . ,
    data = bc.train,
    method = "ada",
    trControl = ctrl,
    tuneGrid = ada.grid)
ada.pred <- predict(ada.fit , newdata = bc.test) # Make predictions
confusionMatrix(ada.pred, bc.test$diagnosis, positive = "M") # Check results and set M as positive class
ada.fit$finalModel # Check model

# VARIABLE IMPORTANCE #
ctree.varImp <- varImp(ctree.fit, scale = TRUE) # Calculate variable importance ctree
#png(file="varimp_ctree.png", units="in", width=3.8, height=2.4, res=300) # Export plot
impVals <-round(head(arrange(ctree.varImp$importance, desc(B)), n = 10), digits =
          2) # Calculate and round values for label inside bar
ggplot(ctree.varImp, top = 10) + # plot Variable importance
  geom_bar(stat = "identity", fill = "#3cafda") + labs(title = "Conditional tree") +
  geom_text(aes(label = impVals[, c(1)]),vjust = 0.5,
    hjust = 1, color = "white")  # add numeric values in bars

rf.varImp <- varImp(rf.fit, scale = TRUE) # Calculate variable importance rf
#png(file="varimp_rf.png", units="in", width=3.8, height=2.4, res=300) # Export plot
impVals <-
  round(head(arrange(rf.varImp$importance, desc(B)), n = 10), digits = 2)
ggplot(rf.varImp, top = 10) + # plot Variable importance
  geom_bar(stat = "identity", fill = "#3cafda") + labs(title = "Random Forest") +
  geom_text(aes(label = impVals[, c(1)]), vjust = 0.5, hjust = 1,
    color = "white") # add numeric values in bar

ada.varImp <-varImp(ada.fit, scale = TRUE) # Calculate variable importance rf 10x cv
#png(file="varimp_ada.png", units="in", width=3.8, height=2.4, res=300) # Export plot
impVals <-
  round(head(arrange(ada.varImp$importance, desc(B)), n = 10), digits = 2)
ggplot(ada.varImp, top = 10) + # plot Variable importance
  geom_bar(stat = "identity", fill = "#3cafda") + labs(title = "Adaboost") +
  geom_text(aes(label = impVals[, c(1)]),vjust = 0.5, hjust = 1,
    color = "white") # add numeric values in bars

xgb.varImp <- varImp(xgb.fit, scale = TRUE) # Calculate variable importance rf 10x cv
#png(file="varimp_xgb.png", units="in", width=3.8, height=2.4, res=300) # Export plot
impVals <-
  round(head(arrange(xgb.varImp$importance, desc(Overall)), n = 10), digits =2)
ggplot(xgb.varImp, top = 10) + # plot Variable importance
  geom_bar(stat = "identity", fill = "#3cafda") + labs(title = "XGBoost") +
  geom_text(aes(label = impVals[, c(1)]), vjust = 0.5, hjust = 1,
    color = "white") # add numeric values in bars

# CALC & PLOT ROC's #
# Conditional tree ROC
auc.ctree <- roc(as.numeric(bc.test$diagnosis), as.numeric(ctree.pred),  ci = TRUE)
plot(auc.ctree, #plot AUC
  ylim = c(0, 1),
  print.thres = TRUE,
  main = paste('Conditional Tree AUC:', round(auc.ctree$auc[[1]], 3)),
  col = '#3cafda')

# Random Forest ROC
auc.rf <- roc(as.numeric(bc.test$diagnosis), as.numeric(rf.pred),  ci = TRUE)
plot(auc.rf, # Plot AUC
  ylim = c(0, 1),
  print.thres = TRUE,
  main = paste('Random Forest AUC:', round(auc.rf$auc[[1]], 3)),
  col = '#3cafda')

# Adaboost ROC
auc.ada <- roc(as.numeric(bc.test$diagnosis), as.numeric(ada.pred),  ci = TRUE)
plot(auc.ada, # Plot AUC
  ylim = c(0, 1),
  print.thres = TRUE,
  main = paste('Adaboost AUC:', round(auc.ada$auc[[1]], 3)),
  col = '#3cafda')

# XGBoost ROC
auc.xgb <- roc(as.numeric(bc.test$diagnosis), as.numeric(xgb.pred),  ci = TRUE)
plot(auc.xgb, # Plot AUC
  ylim = c(0, 1),
  print.thres = TRUE,
  main = paste('XGBoost AUC:', round(auc.xgb$auc[[1]], 3)),
  col = '#3cafda')

# ALL AUC's IN ONE PLOT
plot(auc.ctree, col = 'blue') # Add RF
par(new = TRUE)
plot(auc.rf, col = 'red') # Add RF
par(new = TRUE)
plot(auc.xgb, col = 'green') # Add XGBoost
par(new = TRUE)
plot(auc.ada, col = 'purple') # Add Adaboost

# scatter plot of the AUC against max_depth and eta
#ggplot(xgb.fit$results, aes(x = as.factor(eta), y = max_depth, size = ROC, color = ROC)) +
#  geom_point()

#_______________________________________#
#   SUPPORT VECTOR MACHINES             #
#---------------------------------------#
# (the tuning takes a while and not using seq to reduce calculations because my laptop is gonna melt)
# Linear SVM #
svmlin <- svm(formula = diagnosis ~ ., data = bc.train, kernel = "linear")
svmlin.pred <- predict(svmlin, newdata = bc.test) # Make predictions
# Tune Linear SVM #
svmlin.pars <- tune(svm, diagnosis ~ ., data = bc.train,kernel = "linear",
    # tune parameters using 10x cv
    ranges = list(epsilon = c(.0001, .001, .01, .1, .5),       # gamma not needed for lin
      cost = c(.0001, .001, .01, .1, .5, 1, 2.5, 5, 7.5, 10)))
svmlin.pars # Display optimal parameters
svmlin <- svm(formula = diagnosis ~ .,data = bc.train, kernel = "linear",
    epsilon = 0.0001, cost = 0.1, cross = 10) # Fit SVM with optimal parameters
svmlin.pred <- predict(svmlin, newdata = bc.test) # Make predictions
confusionMatrix(svmlin.pred, bc.test$diagnosis, positive = "M") # Check results and set M as positive class

# Tune Polynomial SVM #
svmpol.pars <- tune(
    svm, diagnosis ~ ., data = bc.train,
    kernel = "polynomial",
    # tune parameters using 10x cv
    ranges = list(epsilon = c(.0001, .001, .01, .1, .5),
      cost = c(.0001, .001, .01, .1, .5, 1, 2.5, 5, 7.5, 10),
      gamma = c(.01, .1, .25, .5)))
svmpol.pars # Display optimal parameters
svmpol <- svm(formula = diagnosis ~ .,data = bc.train, kernel = "polynomial",
    epsilon = 0.0001, cost = 1, gamma = 0.25, cross = 10) # Fit SVM with optimal parameters
svmpol.pred <- predict(svmpol, newdata = bc.test) # Make predictions
confusionMatrix(svmpol.pred, bc.test$diagnosis, positive = "M") # Check results and set M as positive class

# Tune Radial SVM #
svmrad.pars <- tune(svm, diagnosis ~ ., data = bc.train,
    kernel = "radial", ranges = list(
      epsilon = c(.0001, .001, .01, .1, .5),
      cost = c(.0001, .001, .01, .1, .5, 1, 2.5, 5, 7.5, 10),
      gamma = c(.01, .1, .25, .5)))
svmrad.pars # Display optimal parameters
svmrad <- svm(formula = diagnosis ~ .,data = bc.train, # Fit SVM with optimal parameters
    kernel = "radial", epsilon = 0.0001, cost = 7.5,
    gamma = 0.01, cross = 10) 
svmrad.pred <- predict(svmrad, newdata = bc.test) # Make predictions
confusionMatrix(svmrad.pred, bc.test$diagnosis, positive = "M") # Check results and set M as positive class

# Tune Sigmoid SVM #
svmsig.pars <- tune(svm, diagnosis ~ ., data = bc.train,
    kernel = "sigmoid", ranges = list(
      epsilon = c(.0001, .001, .01, .1, .5),
      cost = c(.0001, .001, .01, .1, .5, 1, 2.5, 5, 7.5, 10),
      gamma = c(.01, .1, .25, .5)))
#,coef0 = seq(-1,1,0.25)
svmsig.pars # Display optimal parameters
svmsig <- svm(formula = diagnosis ~ .,data = bc.train, # Fit SVM with optimal parameters
    kernel = "sigmoid", epsilon = 0.0001,
    cost = 2.5, gamma = 0.01, cross = 10)
svmsig.pred <- predict(svmsig, newdata = bc.test) # Make predictions
confusionMatrix(svmsig.pred, bc.test$diagnosis, positive = "M") # Check results and set M as positive class

plotWeights(svmsig)

# ROC'S #
# Linear SVM
auc.svmlin <- roc(as.numeric(bc.test$diagnosis), as.numeric(svmlin.pred),  ci = TRUE)
plot(
  auc.svmlin,
  ylim = c(0, 1),
  print.thres = TRUE,
  main = paste('Linear SVM AUC:', round(auc.svmlin$auc[[1]], 3)),
  col = '#3cafda'
)

# Polynomial SVM
auc.svmpol <- roc(as.numeric(bc.test$diagnosis), as.numeric(svmpol.pred),  ci = TRUE)
plot(
  auc.svmpol,
  ylim = c(0, 1),
  print.thres = TRUE,
  main = paste('Polynomial SVM AUC:', round(auc.svmpol$auc[[1]], 3)),
  col = '#3cafda'
)

# Radial SVM
auc.svmrad <- roc(as.numeric(bc.test$diagnosis), as.numeric(svmrad.pred),  ci = TRUE)
plot(
  auc.svmrad,
  ylim = c(0, 1),
  print.thres = TRUE,
  main = paste('Radial SVM AUC:', round(auc.svmrad$auc[[1]], 3)),
  col = '#3cafda'
)

# Sigmoid SVM
auc.svmsig <- roc(as.numeric(bc.test$diagnosis), as.numeric(svmsig.pred),  ci = TRUE)
plot(
  auc.svmsig,
  ylim = c(0, 1),
  print.thres = TRUE,
  main = paste('Sigmoid SVM AUC:', round(auc.svmsig$auc[[1]], 3)),
  col = '#3cafda'
)

# All ROC's in one plot
plot(auc.svmlin.tuned,
  ylim = c(0, 1),
  col = 'blue',
  main = paste('ROC Comparison'))
par(new = TRUE)
plot(auc.svmpol, col = 'red')
par(new = TRUE)
plot(auc.svmrad, col = 'green')
par(new = TRUE)
plot(auc.svmsig, col = 'purple')


#___________________________________________________
#Optimzing ntree and mtry values
#Trees <- seq(from = 500, to = 2000, by = 100)
#rfs <- lapply(noTrees, FUN = function(x){
#  randomForest(dummy.outcome ~ texture_mean + area_mean + symmetry_mean +
#                 smoothness_se + symmetry_se + fractal_dimension_se +
#                 smoothness_worst + symmetry_worst + fractal_dimension_worst,
#               data = bc.train, importance = TRUE, ntree = x)
#})

#Get errors
#ntrees.err <- as.data.frame(sapply(rfs, FUN = function(x){
#  confusion.df <- as.data.frame(x$confusion)
#  oob.error <- round((confusion.df[1,2] + confusion.df[2,1]) * 100 / sum(confusion.df[-3]), 2)
#  return (data.frame(x$ntree, oob.error))
#}))

#ntrees.err

#___________________________________________________
# Densities
#qplot(concave.points_worst, data=bcdata, geom="density", fill=diagnosis, alpha=I(0.7),
#          main="Distribution", xlab="Concave of Worst points",
#          ylab="Density")
#qplot(area_worst, data=bcdata, geom="density", fill=diagnosis, alpha=I(0.7),
#          main="Distribution", xlab="Area of Worst",
#         ylab="Density")

# 2D density plot 
#qplot(
#  concave.points_worst,
#  area_worst,
#  data = bcdata,
#  geom = "density2d",
#  main = "2D plot of",
#  col = diagnosis
#)
