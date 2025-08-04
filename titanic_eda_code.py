import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from google.colab import files
uploaded = files.upload()
df = pd.read_csv('train.csv')

print("First 5 rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(['Cabin', 'Ticket'], axis=1, inplace=True)

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

print("\nSurvival Rate by Gender:\n", df.groupby('Sex')['Survived'].mean())
print("\nSurvival Rate by Passenger Class:\n", df.groupby('Pclass')['Survived'].mean())
print("\nSurvival by Embarked:\n", df.groupby('Embarked')['Survived'].mean())
print("\nSurvival by IsAlone:\n", df.groupby('IsAlone')['Survived'].mean())

sns.countplot(data=df, x='Sex', hue='Survived')
plt.title('Survival Count by Gender')
plt.xlabel('Gender'); plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title('Survival Count by Passenger Class')
plt.xlabel('Passenger Class'); plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

sns.countplot(data=df, x='Embarked', hue='Survived')
plt.title('Survival Count by Port of Embarkation')
plt.xlabel('Embarked'); plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='Age', hue='Survived', bins=30, kde=True)
plt.title('Age Distribution by Survival')
plt.show()

sns.barplot(data=df, x='FamilySize', y='Survived')
plt.title('Survival Rate by Family Size')
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

print("\n🔍 Final Key Insights:")
print("- Females had a much higher survival rate than males.")
print("- Passengers in 1st class were more likely to survive.")
print("- Those who traveled alone had lower chances of survival.")
print("- Children (younger age) had higher survival rates.")
