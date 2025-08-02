from sklearn import tree, ensemble, neural_network
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


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

    return accuracy


def mlp_test(x_train, x_test, y_train, y_test):
    mlp = neural_network.MLPClassifier(max_iter=2000,)
    mlp.fit(x_train, y_train)
    accuracy = mlp.score(x_test, y_test)

    return accuracy


data = np.loadtxt('M8/sonar.all-data', delimiter=',', dtype=str)
rows_num = data.shape[0]
np.random.shuffle(data)
# print(rows_num)
x = data[:, :-1].astype(float)
y = data[:, -1]

dtc_scores = []
rfc_scores = []
mlp_scores = []

for i in range(50, rows_num + 1, 10):
    x_sample = x[:i]
    y_sample = y[:i]
    # print(len(x_sample))

    dtc_loc = []
    rfc_loc = []
    mlp_loc = []

    for i in range(100):
        x_train, x_test, y_train, y_test = train_test_split(
        x_sample, y_sample, train_size=0.8, shuffle=True,
        )

        dtc_loc.append(dtc_test(x_train, x_test, y_train, y_test))
        rfc_loc.append(rfc_test(x_train, x_test, y_train, y_test))
        mlp_loc.append(mlp_test(x_train, x_test, y_train, y_test))

    dtc_scores.append(np.array(dtc_loc).mean())
    rfc_scores.append(np.array(rfc_loc).mean())
    mlp_scores.append(np.array(mlp_loc).mean())

dtc_scores = np.array(dtc_scores)
rfc_scores = np.array(rfc_scores)
mlp_scores = np.array(mlp_scores)


print(dtc_scores)
print(rfc_scores)
print(mlp_scores)


sample_sizes = np.arange(50, rows_num + 1, 10)


plt.plot(sample_sizes, dtc_scores, marker='o', linestyle='-', label='Decision Tree')
plt.plot(sample_sizes, rfc_scores, marker='s', linestyle='--', label='Random Forest')
plt.plot(sample_sizes, mlp_scores, marker='^', linestyle='-.', label='MLP')

plt.xlabel('Number of Samples')
plt.ylabel('Mean Accuracy')
plt.title('Classifier Accuracy vs. Sample Size')
plt.legend()
plt.grid(alpha=0.3)
plt.show()