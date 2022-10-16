import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# plt.rc('text', usetex=True)

# pts = np.loadtxt('linpts.txt')
# X = pts[:, :2]
# Y = pts[:, 2].astype('int')
# X = np.array([[2, 0], [3, 1], [4, 2], [1, 2], [2, 3], [3, 4], [0.8, 2.5], [1.8, 3.5], [2.8, 4.5]])
# Y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])
X = np.array([[2, 0], [3, 1], [4, 2], [1, 2], [2, 3], [3, 4]])
Y = np.array([0, 0, 0, 1, 1, 1])
print(X, Y)
# Fit the data to a logistic regression model.

lr = LogisticRegression()
svm = LinearSVC()
lr.fit(X, Y)
svm.fit(X, Y)

xmin, xmax = 0, 4
ymin, ymax = -2, 7
plt.scatter(*X[3:, :].T, s=100, alpha=0.5, marker='o')
plt.scatter(*X[:3, :].T, s=100, alpha=0.5, marker='^')


# plt.scatter(*X[Y == 1].T, s=8, alpha=0.5)


# Retrieve the model parameters.
def plot_boundary(model, name, color):
    b = model.intercept_[0]
    w1, w2 = model.coef_.T
    # Calculate the intercept and gradient of the decision boundary.
    c = -b / w2
    m = -w1 / w2

    # Plot the data and the classification with the decision boundary.

    xd = np.array([xmin, xmax])
    yd = m * xd + c
    plt.plot(xd, yd, f'{color}--', lw=1, label=name)


plot_boundary(lr, "Logistic Regression", "b")
plot_boundary(svm, "Linear SVM", "r")
plt.arrow(-0.5, 0, 5, 0, head_width=0.1, head_length=0.15, color='k')
plt.arrow(0, -1, 0, 8, head_width=0.1, head_length=0.15, color='k')
plt.xticks([])
plt.yticks([])
plt.legend()
plt.title("HW1-6-A(a)")
plt.savefig("HW1-6-A1.png")
plt.show()
