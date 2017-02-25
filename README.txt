Preprocessing:
-Removed redundant columns/ renamed columns
-Converted Iron units to micrograms
-Converted excel file to csv
-Split data into approximately 80%-20% (train-test)





First Pass:
Will treat each respective Copper,Iron,Lead 3-tuple as a data point and disregard time. We will try to predict the amount of lead from copper and iron.
-We try a linear regression on these:
		-Regression on only Copper to predict Lead is inconclusive as expected, same for Iron to predict Lead 
		-Regression using both to predict lead was equally inconclusive
		-Regression on just iron performed best in terms of R^2 score, copper performed worse than even expected value
-We next try kernel svm models:
		-Our linear kernel svm (R^2 = .02 on train, .2 on test)
		-Our rbf kernel svm (R^2 = -.01 on train, -.1 on test)
		-Our polynomial(deg 3) kernel svm did not converge for some reason
		-Our sigmoid kernel svm (R^2 = -.018 on train, -.15 on test)
We next tried a normal MLP Neural network regression with multiple hidden layer widths/depths:
		-Our results were variable so we averaged over 5 train-test R^2 results
		-For two hidden neurons we got R^2 (.03 for train, .047 for test)
		-For 4 neurons our average R^2 (.06 for train, .06 for test)
		-For 8 hidden layer neuron R^2 (.06 train, .065 test)
		-For 16 hidden layer neuron R^2 (.068 train, .15 test)
		-For 16 neurons in first layer 8 in second R^2 (.06 train, .069 test)
		-For 32 neurons in first layer 16 in second R^2 (.07 train, .05 test)
We next try decision tree regressors:
		-Results not worth reporting, overfit dramatically





Second Pass:
Now we are working with Copper, Iron, Chloride, Lead 4-tuple as a datapoint. We disregard the time series component again. We will try to predict lead content from the other 3. We will use cross validation when it is appropriate.

-We try a linear regression:

-We try kernel SVMs:

-We try Decision Trees:

-We try a regular MLP Neural Network regression with multiple hidden layer depths/widths:





Third Pass:
We now treat the data as time series. We will use the Copper,Iron, Chloride readings from each home to predict the lead at the corresponding time step.

-We try RNN of varying complexity:

-