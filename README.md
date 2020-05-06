## xgboost use example  
It is a test task in 2019.  
Interviewer allowed me to put my solution into public access.  
  
## Task  
There is a table in train.csv  
The task is to predict last column and test model on test.csv  
  
## Solution  
See t001_read_data.py  
First I found that in train.csv last categorical feature is equal to target (if apply some redesignation)  
That is why I drop this feature as it does not add any information.  
Then I found that dataset is balanced.  
Then I found that there is no ommisions in train.csv but it exist in test.csv  
Then I install xgboost and trained XGBClassifier() on train.csv  
Made test of the model with 5-fold cross-validation on train.csv  
I found that all 5 accuracies are almost same.  
This mean that there is no any heterogeneity in the dataset.  
Then I traned an all train.csv.  
Then I filed ommisions in test.csv with mean values.  
Then I found accuracy on test.csv  
The result is 0.961  


