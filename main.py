import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
import matplotlib.pyplot as plt
import os

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

# Функция для сохранения модели
def save_model(model, filename):
    model.save_model(filename)
    print(f"Модель сохранена в {filename}")

# Функция для загрузки модели
def load_model(filename):
    model = xgb.XGBClassifier()
    model.load_model(filename)
    print(f"Модель загружена из {filename}")
    return model

# Загрузка данных
X, y = load_data()

# Проверка распределения классов в исходных данных
print("Распределение классов в исходных данных:")
print(y.value_counts())

# Обработка меток кластеров: удаление строк с метками `-1` и `3`
valid_indices = y.isin([0, 1, 2])
X = X[valid_indices]
y = y[valid_indices]

# Проверка распределения классов после удаления аномальных данных
print("\nРаспределение классов после удаления аномальных данных:")
print(y.value_counts())

# Разделение данных на обучающую, валидационную и тестовую выборки
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)  # 60% на обучение, 40% на временную тестовую выборку
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 20% на валидацию, 20% на тест

# Проверка распределения классов в обучающей и тестовой выборках
print("\nРаспределение классов в обучающей выборке:")
print(y_train.value_counts())
print("\nРаспределение классов в валидационной выборке:")
print(y_val.value_counts())
print("\nРаспределение классов в тестовой выборке:")
print(y_test.value_counts())

# Проверяем, существует ли уже сохраненная модель
model_filename = 'xgb_model.json'
if os.path.exists(model_filename):
    model = load_model(model_filename)  # Загружаем модель, если она существует
else:
    # Определение параметров для поиска
    param_grid = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3]
    }

    # Поиск гиперпараметров
    grid_search = GridSearchCV(estimator=xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
                               param_grid=param_grid,
                               scoring='accuracy',
                               cv=3,
                               verbose=1,
                               n_jobs=-1)

    grid_search.fit(X_train, y_train)

    # Получение лучших параметров
    best_params = grid_search.best_params_
    print(f"Лучшие параметры: {best_params}")

    # Обучение модели с лучшими параметрами
    model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    save_model(model, model_filename)   # Сохраняем созданную модель

# Оценка важности фич
importances = model.feature_importances_
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Оценка точности и полноты модели
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='weighted')

# Вывод отчета по метрикам классификации в консоль
print("Classification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# Вывод важности фич в консоль
print("\nFeature Importances:\n")
print(feature_importances)

# Визуализация важности фич с значениями важности
plt.figure(figsize=(10, 8))
plt.bar(feature_importances['Feature'], feature_importances['Importance'])
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.title('Feature Importances')

# Добавление значений важности рядом с графиками
for index, value in enumerate(feature_importances['Importance']):
    plt.text(index, value, f'{value:.4f}', ha='center', va='bottom')

# Отображение точности и полноты модели на графике
plt.text(len(feature_importances) - 1, max(feature_importances['Importance']), f'Accuracy: {accuracy:.4f}\nRecall: {recall:.4f}',
         ha='right', va='top', fontsize=12, color='red')

plt.show()
