import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier

#Create data frame and read data
df = pd.read_csv('student_performance.csv')
print(df)

#Decide on features
y = df.Passed
features = ['StudyHours', 'AttendanceRate']
X = df[features]

#Create model
passStatus_model = DecisionTreeClassifier(random_state=42)
#Split data as training and validation
train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=42)
#Train the model
passStatus_model.fit(train_X,train_y)
#Prediction validation
val_pred = passStatus_model.predict(val_X)
print("Mean absolute error: ", mean_absolute_error(val_y, val_pred))

# Get values from user
studyHourOfStudent = float(input("Please enter your study hours: "))
attendanceRateOfStudent = float(input("Please enter your attendance rate (%): "))

# Create a new data
new_df = pd.DataFrame({
    'StudyHours': [studyHourOfStudent],
    'AttendanceRate': [attendanceRateOfStudent]
})

#Make a prediction using new data frame and print out the result
new_pred = passStatus_model.predict(new_df)
if new_pred == 1:
    print("Student predicted as passed!")
else:
    print("Student predicted as failed!")
