import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# Функция для загрузки данных
def load_data():
    features_files = [
        'data_with_data_dep_JPCustom.csv',
        'data_with_data_dep_PSCustom.csv',
        'data_with_data_dep_ZeusCustom.csv'
    ]
    labels_files = [
        'cluster_labelsJPCustom.txt',
        'cluster_labelsPSCustom.txt',
        'cluster_labelsZeusCustom.txt'
    ]

    data = []
    labels = []

    for f_file, l_file in zip(features_files, labels_files):
        # Загрузка фич
        df = pd.read_csv(f_file, delimiter=',')
        df = df.iloc[:, 1:]  # Убираем первый столбец с датой и временем
        data.append(df)

        # Загрузка меток
        with open(l_file, 'r') as lf:
            label = []
            for line in lf:
                label.extend([int(x) for x in line.strip().split() if x.lstrip('-').isdigit()])
        labels.append(pd.Series(label, name='Cluster'))

    return pd.concat(data, ignore_index=True), pd.concat(labels, ignore_index=True)


# Загрузка данных
X, y = load_data()

# Обработка меток кластеров: удаление строк с метками `-1`
valid_indices = y != -1
X = X[valid_indices]
y = y[valid_indices]

# Разделение данных на обучающую, валидационную и тестовую выборки
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)  # 60% на обучение, 40% на временную тестовую выборку
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 20% на валидацию, 20% на тест

# Обучение модели XGBoost с валидацией
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', early_stopping_rounds=10)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# Оценка важности фич
importances = model.feature_importances_
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Оценка точности модели
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Визуализация важности фич с значениями важности
plt.figure(figsize=(10, 8))
plt.bar(feature_importances['Feature'], feature_importances['Importance'])
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.title('Feature Importances')

# Добавление значений важности рядом с графиками
for index, value in enumerate(feature_importances['Importance']):
    plt.text(index, value, f'{value:.4f}', ha='center', va='bottom')

# Отображение точности модели на графике
plt.text(len(feature_importances) - 1, max(feature_importances['Importance']), f'Accuracy: {accuracy:.4f}',
         ha='right', va='bottom', fontsize=12, color='red')

print(f'Accuracy: {accuracy:.4f}')

plt.show()
