Preprocessing:
-Removed redundant columns/ renamed columns
-Converted Iron units to micrograms
-Converted excel file to csv
-Split data into approximately 80%-20% (train-test)



First Pass:
-Will treat each respective Copper,Iron,Lead 3-tuple as a data point and disregard time. We will try to see what can be predicted about lead from copper and iron.
-We first try a linear regression on these:
		-Regression on only Copper to predict Lead is inconclusive as expected, same for Iron to predict Lead 
		-Regression using both to predict lead was equally inconclusive
		-Regression on just iron performed best in terms of R^2 score, copper performed worse than even expected value
-We next try kernel svm models:
		-Our linear kernel svm (R^2 = .02 on train, .2 on test)
		-Our rbf kernel svm (R^2 = -.01 on train, -.1 on test)
		-Our polynomial(deg 3) kernel svm did not converge for some reason
		-Our sigmoid kernel svm (R^2 = -.018 on train, -.15 on test)
We next tried Neural network regression with multiple hidden layer widths/depths:
		-Our results were variable so we averaged over 5 train-test R^2 results
		-For two hidden neurons we got R^2 (.03 for train, .047 for test)
		-For 4 neurons our average R^2 (.06 for train, .06 for test)
		-For 8 hidden layer neuron R^2 (.06 train, .065 test)
		-For 16 hidden layer neuron R^2 (.068 train, .15 test)
		-For 16 neurons in first layer 8 in second R^2 (.06 train, .069 test)
		-For 32 neurons in first layer 16 in second R^2 (.07 train, .05 test)
We next try decision tree regressors:
		-Results not worth reporting, overfit dramatically