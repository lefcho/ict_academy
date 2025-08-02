from sklearn import tree, ensemble, neural_network
from sklearn.model_selection import train_test_split
import numpy as np


def dtc_test(x_train, x_test, y_train, y_test):
    dtc_algthm = tree.DecisionTreeClassifier()
    dtc_algthm = dtc_algthm.fit(x_train, y_train)

    # prediction = dtc_algthm.predict(x_test)
    # # print(prediction)

    accuracy = dtc_algthm.score(x_test, y_test)
    # print(f"Decision Tree Classifier: {accuracy * 100:.2f} %")
    return accuracy


def rfc_test(x_train, x_test, y_train, y_test):
    rfc = ensemble.RandomForestClassifier()
    rfc.fit(x_train, y_train)
    
    accuracy = rfc.score(x_test, y_test)

    # print(f"Random Forest Classifier: {accuracy * 100:.2f} %")
    return accuracy


def mlp_test(x_train, x_test, y_train, y_test):
    mlp = neural_network.MLPClassifier(max_iter=2000,)
    mlp.fit(x_train, y_train)
    accuracy = mlp.score(x_test, y_test)

    # print(f"MLP Classifier: {accuracy * 100:.2f} %")
    return accuracy


data = np.loadtxt('M8/sonar.all-data', delimiter=',', dtype=str)
# print(data.shape)

x = data[:, :-1].astype(float)
y = data[:, -1]

# print(x_test)
# print(y_test)

dtc_scores = []
rfc_scores = []
mlp_scores = []

for i in range(100):
    x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y,
    train_size=0.8,
    shuffle=True,
)

    dtc_scores.append(dtc_test(x_train, x_test, y_train, y_test))
    rfc_scores.append(rfc_test(x_train, x_test, y_train, y_test))
    mlp_scores.append(mlp_test(x_train, x_test, y_train, y_test))


dtc_scores = np.array(dtc_scores)
rfc_scores = np.array(rfc_scores)
mlp_scores = np.array(mlp_scores)

print(f"Decision Tree accuracy: {dtc_scores.mean()*100:.2f}%")
print(f"Random Forest accuracy: {rfc_scores.mean()*100:.2f}%")
print(f"MLP Classifier accuracy: {mlp_scores.mean()*100:.2f}%")
