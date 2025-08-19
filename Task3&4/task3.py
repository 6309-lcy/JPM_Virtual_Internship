import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('Task 3 and 4_Loan_Data.csv')
target = data["default"]
data = data.drop(columns=["default"])

def create_decision_tree_model():
    data = pd.read_csv('Task 3 and 4_Loan_Data.csv')
    
    data = data.drop(columns=["default"])
    presplit = data.iloc[:, 1:7].drop('income', axis=1)
    presplit = StandardScaler().fit_transform(presplit)

    X = presplit[:, :-1]
    y = data['default'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    w_train = compute_sample_weight("balanced", y_train)
    dt = DecisionTreeClassifier(max_depth=5, random_state=35)
    dt.fit(X_train, y_train, sample_weight=w_train)
    return dt

def predict_expected_loss(input_data:list):
    # assume dont have customer_id and default in input_data
    # assume input_data is a list
    # assume input_data list structure is the same as the training data
    input_data = input_data.pop(3)
    
    prob_default = StandardScaler().fit_transform(input_data)
    model = create_decision_tree_model()
    prob = model.predict(prob_default)
    loan_outstanding = input_data[1]
    return prob * loan_outstanding * 0.9