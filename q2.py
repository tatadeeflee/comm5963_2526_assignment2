import sklearn
import sklearn.tree
model = sklearn.tree.DecisionTreeClassifier()
import matplotlib.pyplot as plt

from utils import load_train_test_datasets
from static import FEATURES, TARGET

def run_decision_tree():
    train_x, train_y, test_x, test_y = load_train_test_datasets()
    model = None
    # TODO: Run a classification by constructing a decision tree (Please set the random_state to 5963)
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(random_state=5963)
    model.fit(train_x, train_y)
    # TODO: Print the train and test accuracy of the model
    train_acc = model.score(train_x, train_y)
    test_acc = model.score(test_x, test_y)
    print(f'Train accuracy: {train_acc:.2%}')
    print(f'Test accuracy: {test_acc:.2%}')
    return model

def show_decision_tree(model_from_part1):
    # TODO: Visualize the decision tree
    from sklearn.tree import plot_tree
    plt.figure(figsize=(12,8))
    plot_tree(model_from_part1,feature_names = FEATURES,class_names = ['setosa','versicolor','virginica'],filled = True, rounded = True)
    plt.savefig('q2_part2.png')
    plt.show

def run_random_forest():
    train_x, train_y, test_x, test_y = load_train_test_datasets()
    # TODO: Run a classification by constructing a random forest (Please set the random_state to 5963)
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state = 5963)
    model.fit(train_x,train_y)
    # TODO: Print the train and test accuracy of the model
    train_acc = model.score(train_x,train_y)
    test_acc = model.score(test_x,test_y)
    print(f'Train accuracy:{train_acc:.2%}')
    print(f'Test accuracy:{test_acc:.2%}')


if __name__ == '__main__':
    print('[Q2][Part 1] Run Decision Tree')
    model = run_decision_tree()
    print('[Q2][Part 2] Visualize the Decision Tree')
    show_decision_tree(model_from_part1=model)
    print('[Q2][Part 3] Run Random Forest')
    run_random_forest()
