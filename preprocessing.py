import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess(path="train.csv"):
    df = pd.read_csv(path)

    # Drop columns
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    # Fill ALL missing values 
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Encode categorical
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    #  تأكيد عدم وجود NaN نهائي
    X = X.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)