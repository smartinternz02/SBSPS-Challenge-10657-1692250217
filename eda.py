import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('/content/drive/MyDrive/Placement _data')

sns.countplot(x='gender', hue='status', data=data)
plt.title("Placement Status Distribution based on Gender")
plt.show()

sns.countplot(x='specialisation', hue='status', data=data)
plt.title("Placement Status Distribution based on Specialisation")
plt.xticks(rotation=45)
plt.show()

sns.boxplot(x='status', y='salary', data=data)
plt.title("Salary Distribution based on Placement Status")
plt.show()


sns.pairplot(data, hue='status', diag_kind='kde')
plt.suptitle("Pair Plot of Numerical Features")
plt.show()

