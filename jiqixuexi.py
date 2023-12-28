
from ast import Index
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False

diabetes = pd.read_csv('diabetes.csv')
print(diabetes.columns)
Index(['Pregnancies', 'Glucose', 'BloodPressure','SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction','Age', 'Outcome'], dtype='object')
diabetes.head()
print("dimension of diabetes data: {}".format(diabetes.shape))
diabetes_dimensions = (768, 9)
print(diabetes.groupby('Outcome').size())

# 生成图表
import seaborn as sns
sns.countplot(x="Outcome", data=diabetes, label="Count")
plt.xlabel("结果(Outcome)")
plt.ylabel("数量(Count)")
plt.show()
diabetes.info()

#K近邻算法
#首先，让我们研究是否可以确认模型复杂性和准确性之间的联系：
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'], diabetes['Outcome'], stratify=diabetes['Outcome'], random_state=66)
from sklearn.neighbors import KNeighborsClassifier
training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    # build the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(knn.score(X_train, y_train))
    # record test set accuracy
    test_accuracy.append(knn.score(X_test, y_test))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("准确性(Accuracy)")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

#让我们检查k近邻算法预测糖尿病的准确性得分。
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)
print("检查K近邻算法预测准确性:")
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))
from sklearn.metrics import f1_score
# 在测试集上使用模型进行预测
y_pred = knn.predict(X_test)
# 计算并打印F1分数
f1 = f1_score(y_test, y_pred)
print("F1 Score: {:.2f}".format(f1))

print("决策树准确性：")
#决策树分类器
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
# 预测并计算F1分数
y_pred_default = tree.predict(X_test)
f1_default = f1_score(y_test, y_pred_default)
print("Default Decision Tree - F1 Score on test set: {:.3f}".format(f1_default))

#使用决策树分类器的训练集的准确性为100％，而测试集的准确性则差得多。这表明树过度拟合并且不能很好地归纳为新数据。
# 因此，我们需要对树进行预修剪。
#现在，我将通过设置max_depth = 3再次执行此操作，限制树的深度可减少过度拟合。这会导致训练集的准确性降低，但会提高测试集的准确性。
tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
# 预测并计算F1分数
y_pred_depth3 = tree.predict(X_test)
f1_depth3 = f1_score(y_test, y_pred_depth3)
print("Decision Tree with max_depth=3 - F1 Score on test set: {:.3f}".format(f1_depth3))

def plot_feature_importances_diabetes(model, diabetes_features):
    plt.figure(figsize=(8, 6))
    n_features = len(diabetes_features)  # 根据输入的特征名称列表确定特征数量
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(range(n_features), diabetes_features)  # 设置纵坐标的标签
    plt.xlabel("特征重要性(Feature importance)")
    plt.ylabel("特征名字(Feature Name)")
    plt.ylim(-1, n_features)
# 假设diabetes_features是一个包含特征名称的列表
diabetes_features = ['怀孕', '葡萄糖', '血压', '皮肤厚度', '胰岛素', '体重指标(BMI)','糖尿病谱系特征', '年龄']
plot_feature_importances_diabetes(tree, diabetes_features)

print("深度学习准确性：")
#深度学习预测
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)
print("Accuracy on training set: {:.2f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score(X_test, y_test)))
# 预测并计算F1分数
y_pred_default = mlp.predict(X_test)
f1_default = f1_score(y_test, y_pred_default)
print("Default MLP - F1 Score on test set: {:.3f}".format(f1_default))
plt.show()

#多层感知器（MLP）的准确性根本不如其他模型，这可能是由于数据缩放所致。
#深度学习算法还期望所有输入特征以相似的方式变化，理想情况下，平均值为0，方差为1。
#现在，我将重新缩放我们的数据，以使其能够满足这些要求，从而以良好的准确性预测糖尿病。

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
mlp = MLPClassifier(max_iter=10000,random_state=0)
mlp.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(
   mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))
y_pred_iter_10000 = mlp.predict(X_test_scaled)
f1_iter_10000 = f1_score(y_test, y_pred_iter_10000)
print("MLP with max_iter=10000 - F1 Score on test set: {:.3f}".format(f1_iter_10000))

# 现在，我们增加迭代次数，alpha参数，并为模型的权重添加更强的参数：
mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=0)
mlp.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(
   mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

y_pred_iter_1000_alpha_1 = mlp.predict(X_test_scaled)
f1_iter_1000_alpha_1 = f1_score(y_test, y_pred_iter_1000_alpha_1)
print("MLP with max_iter=1000 and alpha=1 - F1 Score on test set: {:.3f}".format(f1_iter_1000_alpha_1))

plt.figure(figsize=(10, 5))
diabetes_features = diabetes.columns[:-1]
# 通过改变图像的形状来改变高度和宽度
reshaped_coefs = mlp.coefs_[0].reshape(25, 32)
plt.imshow(reshaped_coefs, cmap='viridis', interpolation='none')
plt.yticks([2.2777,5.0554,7.8331,10.6108,13.3885,16.1662,18.9439,21.7216], diabetes_features)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()
plt.show()
