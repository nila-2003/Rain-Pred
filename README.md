# Rain-Pred
Rainfall Prediction using Linear Regression
Dependent variable is the amount of rainfall and independent is the factors on which it depends such as humidity, temperature etc.
First collect historical data. It contains the amount of rainfall and the corresponding values of independent variables.
After collecting the data it is cleaned and pre-processed.  It is then split into two sets- training set, which is used to train the model and testing data set, which is used to evaluate the performance of the model.
The hypothesis in this case is a linear equation-  
Y= theta0+ theta1*x1 + theta2*x2 + theta3*x3……+thetan*xn
Thetas are the coefficients that are learned during training and y is the predicted amount of rainfall.
Theta should be such that it minimizes the differences. MSE
Some methods for data cleaning are: parsing, converting to one-hot- module used is pandas
NUN values in data
Read_csv is a function is pandas that reads a csv file and returns a dataframe object
Example: data=pd.read_csv(&quot
Filename.csv &quot)
&quot represents quotation mark without being interpreted as the end of html attribute value.

Data.drop()- to remove specified rows or columns from a dataframe.
Two parameters- column and axis. Axis specifies whether you want to remove rows or columns
Axis=1 means , we want to remove columns
Data.replace()- method that allows you to replace specified values in a dataframe. Two parameters, value you want to replace and second one is the value to replace with.
Ex-  data= data.replace(‘T’,0.0)
To save the data is a csv file- data.to_csv(‘filename.csv’)
The above code is for data cleaning. Now this data can be used. Linear approach to forming a relationship between a variable and many independent variables. Minimize errors. 
Import sklearn as sk
From sklearn.linear_model import LinearRegression
#read the cleaned data:
Data=pd.read_csv()
# the features or the 'x' values of the data
# these columns are used to train the model
# the last column, i.e, precipitation column
# will serve as the label
X=data.drop([‘’],axis=1)
Y= data[‘’]
#reshaping it into a 2d vector
Y=y.values.reshape(-1,1)
# consider a random day in the dataset
# we shall plot a graph and observe this
# day
Day_index=798
Days=[I for I in range(y.size)]
Clf=LinearRegression() – initialize the linear regression classifier
Train- clf.fit(x,y)
# give a sample input to test our model
# this is a 2-D vector that contains values
# for each column in the dataset.
inp = np.array([[74], [60], [45], [67], [49], [43], [33], [45],
                [57], [29.68], [10], [7], [2], [0], [20], [4], [31]])
inp = inp.reshape(1, -1)
print(clf.predict(inp))
print("the precipitation trend graph: ")
plt.scatter(days, Y, color='g')
plt.scatter(days[day_index], Y[day_index], color='r')
plt.title("Precipitation level")
plt.xlabel("Days")
plt.ylabel("Precipitation in inches")
plt.show()
x_vis = X.filter(['TempAvgF', 'DewPointAvgF', 'HumidityAvgPercent',
                  'SeaLevelPressureAvgInches', 'VisibilityAvgMiles',
                  'WindAvgMPH'], axis=1)
 
# plot a graph with a few features (x values)
# against the precipitation or rainfall to observe
# the trends
 
print("Precipitation vs selected attributes graph: ")
 
for i in range(x_vis.columns.size):
    plt.subplot(3, 2, i + 1)
    plt.scatter(days, x_vis[x_vis.columns.values[i][:100]],
                color='g')
 
    plt.scatter(days[day_index],
                x_vis[x_vis.columns.values[i]][day_index],
                color='r')
 
    plt.title(x_vis.columns.values[i])
 
plt.show()


