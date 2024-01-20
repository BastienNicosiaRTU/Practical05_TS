import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix



#load the dataset
df = pd.read_csv('SDN_traffic.csv', sep=';')
#print(df.columns)

#Treatment of the file : 
#I tested several things to make it work.

#df = pd.get_dummies(df, columns=['category'])
#non_numeric_columns = df.select_dtypes(include=['object']).columns
#print(non_numeric_columns)
#print(df.columns)

df = df.iloc[:, 4:]
df = df.drop('forward_bps_var', axis=1)
#object_columns = df.select_dtypes(include=['object']).columns

#if len(object_columns) > 0:
#    print("There are string columns:", object_columns)
#else:
#    print("No string columns left in the DataFrame.")


#Starting the training
# Split the data into features and target variable
X = df.drop('category', axis=1)  
y = df['category']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CART Decision Tree
cart_classifier = DecisionTreeClassifier(criterion='gini')
cart_classifier.fit(X_train, y_train)

# ID3 Decision Tree (approximated using entropy criterion)
id3_classifier = DecisionTreeClassifier(criterion='entropy')
id3_classifier.fit(X_train, y_train)


# Evaluate CART Classifier
cart_predictions = cart_classifier.predict(X_test)
print("CART Classifier Metrics:")
print(classification_report(y_test, cart_predictions, zero_division=1))
print("Confusion Matrix:")
print(confusion_matrix(y_test, cart_predictions))

# Evaluate ID3 Classifier
id3_predictions = id3_classifier.predict(X_test)
print("\nID3 Classifier Metrics:")
print(classification_report(y_test, id3_predictions, zero_division=1))
print("Confusion Matrix:")
print(confusion_matrix(y_test, id3_predictions))







