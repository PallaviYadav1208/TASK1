# THE SPARKS FOUNDATION GRIP TASKS
## DATA SCIENCE AND BUSINESS ANALYTICS TASKS
### NAME: PALLAVI YADAV
TASK 1: PREDICTION USING SUPERVISED MACHINE LEARNING(SIMPLE LINEAR REGRESSION)<br>

AIM: TO PREDICT THE PERCENTAGE OF STUDENT BASED ON THE NO. OF HOURS OF STUDY AND TO PREDICT THE SCORE IF THE STUDENT STUDIES FOR 9.5 HRS/DAY

#### Importing all the required libraries<br>
>import pandas as pd<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>

##### reading data from web<br>
>url="http://bit.ly/w-data"<br>
data=pd.read_csv(url)<br>

![image](https://user-images.githubusercontent.com/97663851/173022920-a0c96410-691f-4e5a-957a-6e691df85f27.png)

> #checking the shape of dataset<br>

![image](https://user-images.githubusercontent.com/97663851/173022516-95068989-dc68-42c1-85a4-9df25a18c4d9.png)

>#checking for information of data<br>

![image](https://user-images.githubusercontent.com/97663851/173073239-9e2b302a-5016-44b5-94ae-75b2a142967c.png)

> #describing the data<br>

![image](https://user-images.githubusercontent.com/97663851/173073445-d4b6ef1f-27da-480a-925f-cb02264a5ea6.png)

>#cheching for missing value<br>

![image](https://user-images.githubusercontent.com/97663851/173073710-183a9d80-0b85-4927-9b01-010e465c4538.png)

Since there are no missing values in the dataset, Data Cleaning is not required.

## DATA VISUALIZATION
> #Plotting the graph between hours studied and Scores obtained<br>
x=data['Hours'].values.reshape(-1,1)<br>
y=data['Scores'].values.reshape(-1,1)<br>
plt.figure(figsize=(13,7))<br>
plt.scatter(x, y, color="firebrick",marker="o")<br>  
plt.title('Study Hours vs Score') <br> 
plt.xlabel('Hours Studied')<br>  
plt.ylabel('Score')<br>
plt.show()<br>

![image](https://user-images.githubusercontent.com/97663851/173074473-b5068f4f-a17e-4aa8-8625-e571d8602eb7.png)

From the scatter plot it is clear that as the number of hours of study increases, the student's score also increases.

Thus, there is a positive linear relationship between hours studied and score obtained.

Therefore, we build a Simple Linear Regression Model for the given dataset.

## SPLITTING THE DATASET INTO TRAINING AND TESTING DATASET

>from sklearn.model_selection import train_test_split <br> 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)<br>

![image](https://user-images.githubusercontent.com/97663851/173075184-53f185e8-9d71-49ad-bf8c-a7ff831a5560.png)
## TRAINING THE MODEL

>from sklearn.linear_model import LinearRegression  <br>
model = LinearRegression()  <br>
model.fit(x_train, y_train) <br>
model.intercept_<br>
model.coef_<br>

>#Plotting the regression line<br>
line = model.coef_*x_train+model.intercept_<br>

![image](https://user-images.githubusercontent.com/97663851/173075682-4c8c483c-e4ac-401d-9116-55d90c635a3e.png)

#Plotting for the given dataset<br>
plt.figure(figsize=(13,7))<br>
plt.scatter(x, y,label="Score",color="firebrick",marker="o")<br>
plt.plot(x_train, line,label="Regression Line",color="firebrick")<br>
plt.title("Study Hours  VS Score")<br>
plt.xlabel("Hours Studied")<br>
plt.ylabel("Score")<br>
plt.legend()<br>
plt.show()<br>

![image](https://user-images.githubusercontent.com/97663851/173076216-d45a7004-7a58-41cb-98ee-dbf8e03def9d.png)

## PREDICTION USING THE MODEL

>y_pred = model.predict(x_test).round(2) # Predicting the scores for testing dataset<br>
y_pred<br>


![image](https://user-images.githubusercontent.com/97663851/173076413-b0f3cdb9-9210-4a01-981a-07b89b69b34b.png)

>#Comparing Actual vs Predicted scores for test dataset<br>
df = pd.DataFrame(np.c_[x_test,y_test,y_pred],columns=["Hours","Actual Score","Predicted Score"])  <br>
df <br>
![image](https://user-images.githubusercontent.com/97663851/173076658-43571bb0-bb9f-4902-b635-c82064737968.png)


## PREDICTING THE SCORE WHEN THE STUDENT STUDIES FOR 9.25 HOURS/DAY.

>hours = 9.25<br>
pred = model.predict([[hours]])[0][0].round(2)<br>
print("No of Hours studied= ",hours)<br>
print("Predicted Score = ",pred)<br>

![image](https://user-images.githubusercontent.com/97663851/173076908-e9bde5c9-1a69-4d1a-be78-521bea29998e.png)

## EVALUATING THE MODEL
>#Checking the error<br>
from sklearn import metrics <br> 
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) <br>

![image](https://user-images.githubusercontent.com/97663851/173077166-ca77e820-1d6a-43c7-b967-d7646c54c7e8.png)

![image](https://user-images.githubusercontent.com/97663851/173077306-f6c00511-a4f3-481c-9104-9bffef349e35.png)<br>
Model-Score : 0.9735538080811826
This shows that our model gives 97.355% accurate results, which is good.

INTERPRETATION : If the student studies for 9.25 hrs/day, then according to our model he/she can get a score of 93.46%

THANK YOU



