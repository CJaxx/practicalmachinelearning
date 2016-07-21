# PML_Project
CJaxx  
18 July 2016  



## Introduction

This project is based on a dataset collected by Velloso et al (Velloso et al, 2013) as part of a study to evaluate whether data collected from on-body sensors could be used to assess the quality of the exercise being measured, to assess whether the exercise was conducted correctly or not.  In this project we are using their data to attempt to build a model which can predict whether the activity was done correctly of not.    

The data comes from sensors placed on the bicep, wrist and belt of the participants and on the dumbbell. There were six male participants (aged 20-28) and they were each asked to perform one set of 10 repetitions of a Unilateral Dumbbell Bicep Curl.  Each participant did one set correctly, then one set in four incorrect methods (corresponding to the most common mistakes for this type of exercise). The dumbbell used was 1.25kg, light enough to allow the participants to do the incorrect methods without the risk of injuring themselves (Velloso et al, 2013).

* Class A = The correct method - 
* Class B = incorrect method - throwing the elbows to the front
* Class C = incorrect method - lifting the dumbbell only half way
* Class D = Incorrect method - lowering the dumbbell only half way
* Class E = incorrect method - throwing the hips forward

https://en.wikipedia.org/wiki/Euler_angles#/media/File:Eulerangles.svg

Following the Components of a predictor from lecture 2, we have:

Question > input data > features > algorithm > parameters > evaluation

## Question 
First we need to define/clarify the question we are trying to answer.

Can we use on-body sensor data to accurately predict whether a participant did the Dumbbell bicep curl correctly made one of four common mistakes?
## Input data


### Load data

```r
pmlTraining <- read.csv("pml-training.csv", header=TRUE )
pmlValidation <- read.csv("pml-testing.csv", header=TRUE)
dim(pmlTraining)
```

```
## [1] 19622   160
```

```r
dim(pmlValidation)
```

```
## [1]  20 160
```
### Initial data analysis

```r
str(pmlTraining)
dim(pmlTraining)
```
Looking at the training data we can see that there are 19,622 observations with 160 variables recorded. There are a lot of blank fields and NAs that need to be dealt with. 

### Data cleaning
The datasets have a number of issues: 

1. There are a lot of fields with zero or near zero values
2. There are a lot of NAs
3. There are unnecessary fields relating the participant and the sensor that we don't need


```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
## Clean the data
pmlTraining <- pmlTraining[, colSums(is.na(pmlTraining))==0]
## Identify and remove near zero values with caret
nzv <- nearZeroVar(pmlTraining)
pmlTraining <- pmlTraining[ , -nzv]
## Remove the first 5 columns which identify the participant and the timestamp
pmlTraining <- pmlTraining[, -(1:5)]

## Clean the Validation test set
pmlValidation <- pmlValidation[, colSums(is.na(pmlValidation))==0]

dim(pmlTraining)
```

```
## [1] 19622    54
```

```r
dim(pmlValidation)
```

```
## [1] 20 60
```


## Feature selection

We need to use cross validation to pick the variables to include in the model, the type of prediction function to use and to pick the parameters in the predict function and to compare different predictors (lecture 8). The cross-validation approach was outlined as:

1. Use the training set
2. Split it into training/test sets
3. Build a model on the training set
4. Evaluate it on the test set
5. Repeat and average the estimate errors
6. Additional step here to test on the validation set for Coursera Submission


1-2. Split the training data so we can test model without using the final Validation dataset we need to submit (and to avoid overfitting). I'm using a 60/40 split as advised for medium size sample (then the 20 for the test data given = the validation set which is optional except in test situations).  


```r
set.seed(25)
library(caret)
inTrain <- createDataPartition(y=pmlTraining$classe, p=0.6, list=F)
Training <- pmlTraining[inTrain,] # use for training our model
Testing <- pmlTraining[-inTrain,] # use for testing our model
dim(Training)
```

```
## [1] 11776    54
```

```r
dim(Testing)
```

```
## [1] 7846   54
```
So now we have three datasets Training with 11,776 records, Testing with 7846 recrods, and the final validation (to be used only for the final test and submission (pmlValidation)). The Training and Testing datasets have 54 variables.

This leaves the pmlValidation dataset.

3. Build a model

### Model 1 - We'll start with a classification tree.
As this is a learning activity, I'm going to build 2 models, the first bootstrapped, the second cross-validated.

```r
library(rattle)
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 4.1.0 Copyright (c) 2006-2015 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
set.seed(123)
# Our first model (bootstrapped)
fitRpartBS <- train(classe ~., data=Training, method="rpart")
```

```
## Loading required package: rpart
```

```r
fancyRpartPlot(fitRpartBS$finalModel, palettes=c("Greens", "Greys"))
```

![](index_files/figure-html/unnamed-chunk-5-1.png)<!-- -->

```r
fitRpartBS
```

```
## CART 
## 
## 11776 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 11776, 11776, 11776, 11776, 11776, 11776, ... 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa     
##   0.04046037  0.5266549  0.38672854
##   0.05821864  0.3765645  0.14182833
##   0.11438064  0.3230480  0.05822745
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.04046037.
```

```r
# Second model - same thing but using cross validation
crossVal <- trainControl(method="cv", number=5)
fitRpartCV <- train(classe ~., data=Training, method="rpart", trControl=crossVal)
fancyRpartPlot(fitRpartCV$finalModel, palettes="YlGn")
```

![](index_files/figure-html/unnamed-chunk-5-2.png)<!-- -->

```r
fitRpartCV
```

```
## CART 
## 
## 11776 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 9421, 9422, 9421, 9420, 9420 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa     
##   0.04046037  0.5338021  0.39526566
##   0.05821864  0.4128683  0.20480316
##   0.11438064  0.3151286  0.04698248
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.04046037.
```

```r
## Use our models to make predictions
predictBS <- predict(fitRpartBS, Testing)
ConfMatrBS <- confusionMatrix(Testing$class, predictBS)
ConfMatrBS 
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2034   33  162    0    3
##          B  628  516  374    0    0
##          C  605   34  729    0    0
##          D  582  219  485    0    0
##          E  206  196  384    0  656
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5015          
##                  95% CI : (0.4904, 0.5127)
##     No Information Rate : 0.5168          
##     P-Value [Acc > NIR] : 0.9968          
##                                           
##                   Kappa : 0.3488          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.5016  0.51703  0.34161       NA  0.99545
## Specificity            0.9478  0.85368  0.88813   0.8361  0.89064
## Pos Pred Value         0.9113  0.33992  0.53289       NA  0.45492
## Neg Pred Value         0.6400  0.92383  0.78311       NA  0.99953
## Prevalence             0.5168  0.12720  0.27199   0.0000  0.08399
## Detection Rate         0.2592  0.06577  0.09291   0.0000  0.08361
## Detection Prevalence   0.2845  0.19347  0.17436   0.1639  0.18379
## Balanced Accuracy      0.7247  0.68536  0.61487       NA  0.94304
```

```r
predictCV <- predict(fitRpartCV, Testing)
ConfMatrCV <- confusionMatrix(Testing$class, predictCV)
ConfMatrCV 
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2034   33  162    0    3
##          B  628  516  374    0    0
##          C  605   34  729    0    0
##          D  582  219  485    0    0
##          E  206  196  384    0  656
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5015          
##                  95% CI : (0.4904, 0.5127)
##     No Information Rate : 0.5168          
##     P-Value [Acc > NIR] : 0.9968          
##                                           
##                   Kappa : 0.3488          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.5016  0.51703  0.34161       NA  0.99545
## Specificity            0.9478  0.85368  0.88813   0.8361  0.89064
## Pos Pred Value         0.9113  0.33992  0.53289       NA  0.45492
## Neg Pred Value         0.6400  0.92383  0.78311       NA  0.99953
## Prevalence             0.5168  0.12720  0.27199   0.0000  0.08399
## Detection Rate         0.2592  0.06577  0.09291   0.0000  0.08361
## Detection Prevalence   0.2845  0.19347  0.17436   0.1639  0.18379
## Balanced Accuracy      0.7247  0.68536  0.61487       NA  0.94304
```

Out of sample Error:

```r
## Work out the Out of Sample Error
OutSampleErrorBS <- 1-unname(ConfMatrBS$overall[1])
OutSampleErrorBS
```

```
## [1] 0.4984706
```

```r
OutSampleErrorCV <- 1-unname(ConfMatrCV$overall[1])
OutSampleErrorCV
```

```
## [1] 0.4984706
```
The accuracy for both models is 0.5015 (which is about as accurate as a coinflip) and the out of sample error is 0.4984706.  

So we need to try a different approach.


### Model 2 - Random Forest

Warning: The dataset isn't small, so the random forest model will take a few minutes to run (with any method). I'm using the randomForest packages rather than method="rf" in the caret package as it seems to be significantly faster.  I initially tried the random forest method using the train() function from the caret package with the trainControl() adding cross-validation and limiting the number of iterations 

Reading Xing Su's Practical Machine Learning Course Notes (See reference 3, page 46), he suggested that the randomForest() function/package is much faster so I tried it and got similar results in less time.


```r
# Uncomment the next line to install the randomForest package
#install.packages("randomForest")
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
# Model 2.1 - random forest model built with caret and 5-fold cross-validation 

crossVal <- trainControl(method="cv", number=5)
fitRF2.1 <- train(classe ~., data = Training, method="rf", prox=TRUE, trControl=crossVal)
fitRF2.1
```

```
## Random Forest 
## 
## 11776 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 9421, 9421, 9421, 9421, 9420 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9915927  0.9893640
##   27    0.9954142  0.9941994
##   53    0.9943104  0.9928026
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

```r
# Model 2.2 - random forest model built with randomForest() package
fitRF2.2 <- randomForest(classe ~., data = Training, prox=TRUE, ntree=800)
fitRF2.2
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = Training, prox = TRUE,      ntree = 800) 
##                Type of random forest: classification
##                      Number of trees: 800
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.36%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3346    1    0    0    1 0.0005973716
## B    7 2271    1    0    0 0.0035103115
## C    0    8 2045    1    0 0.0043816943
## D    0    0   17 1913    0 0.0088082902
## E    0    0    0    6 2159 0.0027713626
```

We'll test our models on the Testing dataset:

```r
predict2.1 <- predict(fitRF2.1, Testing)
ConfMatr2.1 <- confusionMatrix(Testing$class, predict2.1)
ConfMatr2.1
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    0    0    0    0
##          B    4 1512    2    0    0
##          C    0    5 1363    0    0
##          D    0    0    7 1278    1
##          E    0    0    0    2 1440
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9973          
##                  95% CI : (0.9959, 0.9983)
##     No Information Rate : 0.285           
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9966          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9967   0.9934   0.9984   0.9993
## Specificity            1.0000   0.9991   0.9992   0.9988   0.9997
## Pos Pred Value         1.0000   0.9960   0.9963   0.9938   0.9986
## Neg Pred Value         0.9993   0.9992   0.9986   0.9997   0.9998
## Prevalence             0.2850   0.1933   0.1749   0.1631   0.1837
## Detection Rate         0.2845   0.1927   0.1737   0.1629   0.1835
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9991   0.9979   0.9963   0.9986   0.9995
```

```r
predict2.2 <- predict(fitRF2.2, Testing)
ConfMatr2.2 <- confusionMatrix(Testing$class, predict2.2)
ConfMatr2.2
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    0    0    0    0
##          B    3 1514    1    0    0
##          C    0    4 1364    0    0
##          D    0    0   15 1270    1
##          E    0    0    0    3 1439
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9966         
##                  95% CI : (0.995, 0.9977)
##     No Information Rate : 0.2849         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9956         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9987   0.9974   0.9884   0.9976   0.9993
## Specificity            1.0000   0.9994   0.9994   0.9976   0.9995
## Pos Pred Value         1.0000   0.9974   0.9971   0.9876   0.9979
## Neg Pred Value         0.9995   0.9994   0.9975   0.9995   0.9998
## Prevalence             0.2849   0.1935   0.1759   0.1622   0.1835
## Detection Rate         0.2845   0.1930   0.1738   0.1619   0.1834
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9993   0.9984   0.9939   0.9976   0.9994
```

Accuracy of Random Forest model

```r
OutSampleError2.1 <- 1-unname(ConfMatr2.1$overall[1])
OutSampleError2.1
```

```
## [1] 0.002676523
```

```r
OutSampleError2.2 <- 1-unname(ConfMatr2.2$overall[1])
OutSampleError2.2
```

```
## [1] 0.003441244
```
So the models show:

Model 2.1
- accuracy 0.9973
- out of sample error 0.00268

Model 2.2
- accuracy 0.9966 
- out of sample error 0.00344



## Use the best model to make predictions on the final Validation data

So the caret random forest method has a slightly higher accuracy and smaller out of sample error so we will use this for our final prediction. Perhaps the randomForest() function would have higher accuracy if given a larger `ntree` value.


```r
predictFinal <- predict(fitRF2.1, pmlValidation)

FinalPrediction <- data.frame(pmlValidation$problem_id, predictFinal)
FinalPrediction
```

```
##    pmlValidation.problem_id predictFinal
## 1                         1            B
## 2                         2            A
## 3                         3            B
## 4                         4            A
## 5                         5            A
## 6                         6            E
## 7                         7            D
## 8                         8            B
## 9                         9            A
## 10                       10            A
## 11                       11            B
## 12                       12            C
## 13                       13            B
## 14                       14            A
## 15                       15            E
## 16                       16            E
## 17                       17            A
## 18                       18            B
## 19                       19            B
## 20                       20            B
```

## Conclusion
Within the framework of this assignment, it appears that the random forest model is the best predictor. The reference that goes with the data (Velloso, 2013) uses a 'sliding window' approach to feature extraction, which looks like an interesting technique - I've found another reference which might be interesting to look at: https://fedcsis.org/proceedings/2015/pliks/425.pdf 

We can now use our predictions `FinalPrediction` in the Coursera submission quiz.

## References

1. Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.;  Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013. Read more: http://groupware.les.inf.puc-rio.br/har#ixzz4EsDiTsfG


2. Caret package ref: http://topepo.github.io/caret/preprocess.html

3. Xing Su - Practical Machine Learning Course Notes. http://sux13.github.io/DataScienceSpCourseNotes/
