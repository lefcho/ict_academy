from sklearn import tree


features = [[2, 2], [2, 2], [3, 4], [2, 2], [4, 4], [4, 4], [3, 2], [2, 2], [4, 4]]
labels = [0, 0, 1, 0, 1, 1, 0, 0, 1]

new_data = [[2, 2], [4, 4], [4, 3]]

algorithm = tree.DecisionTreeClassifier()
algorithm = algorithm.fit(features, labels)

print(algorithm.predict(new_data))
