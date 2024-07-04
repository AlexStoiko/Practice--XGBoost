import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import os
from sklearn.inspection import PartialDependenceDisplay
import time

# Функция для загрузки данных
def load_data():
    features_files = [
        'data_with_data_dep_JPCustom.csv',
        'data_with_data_dep_PSCustom.csv',
        'data_with_data_dep_ZeusCustom.csv',
        'data_with_data_dep_1234Custom.csv'
    ]
    labels_files = [
        'cluster_labelsJPCustom.txt',
        'cluster_labelsPSCustom.txt',
        'cluster_labelsZeusCustom.txt',
        'cluster_labels1234Custom.txt'
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
    print(f"\nМодель сохранена в {filename}")

# Функция для загрузки модели
def load_model(filename):
    model = xgb.XGBClassifier()
    model.load_model(filename)
    print(f"\nМодель загружена из {filename}")
    return model

# Функция для сохранения гиперпараметров в файл
def save_hyperparameters(params, filename):
    with open(filename, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    print(f"Гиперпараметры сохранены в {filename}")

# Функция для чтения гиперпараметров из файла и их вывода в консоль
def read_hyperparameters(filename):
    if (os.path.exists(filename)):
        print("\nГиперпараметры модели:")
        with open(filename, 'r') as f:
            for line in f:
                print(line.strip())
    else:
        print("\nФайл с гиперпараметрами не найден.")

# Загрузка данных
X, y = load_data()

# Проверка распределения классов в исходных данных
print("Распределение классов в исходных данных:\n")
print(y.value_counts())

# Обработка меток кластеров: удаление строк с метками `-1`
valid_indices = y.isin([0, 1, 2, 3])
X = X[valid_indices]
y = y[valid_indices]

# Проверка распределения классов после удаления аномальных данных
print("\nРаспределение классов после удаления аномальных данных:")
print(y.value_counts())

# Разделение данных на обучающую, валидационную и тестовую выборки
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=int(time.time()))  # 60% на обучение, 40% на временную тестовую выборку
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=int(time.time()))  # 20% на валидацию, 20% на тест

# Проверка распределения классов в обучающей и тестовой выборках
print("\nРаспределение классов в обучающей выборке:")
print(y_train.value_counts())
print("\nРаспределение классов в валидационной выборке:")
print(y_val.value_counts())
print("\nРаспределение классов в тестовой выборке:")
print(y_test.value_counts())

# Проверяем, существует ли уже сохраненная модель
model_filename = 'xgb_model.json'
hyperparams_filename = 'hyperparameters.txt'

if os.path.exists(model_filename):
    model = load_model(model_filename)  # Загружаем модель, если она существует
else:
    # Определение параметров для поиска
    param_grid = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [70, 80, 90, 100, 150],
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

    # Сохранение гиперпараметров в файл
    save_hyperparameters(best_params, hyperparams_filename)

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
precision = precision_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Чтение и вывод гиперпараметров в консоль
read_hyperparameters(hyperparams_filename)

# Вывод отчета по метрикам классификации в консоль
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# Сохранение отчета в файл
report_filename = 'classification_report.txt'
with open(report_filename, 'w') as f:
    f.write(classification_report(y_test, y_pred, zero_division=0))
    f.write(f"\nAccuracy: {accuracy:.4f}\nRecall: {recall:.4f}\nPrecision: {precision:.4f}\nF1 Score: {f1:.4f}")
print(f"Отчет сохранен в {report_filename}")

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
plt.text(len(feature_importances) - 1, max(feature_importances['Importance']),
         f'Accuracy: {accuracy:.4f}\nRecall: {recall:.4f}\nPrecision: {precision:.4f}\nF1 Score: {f1:.4f}',
         ha='right', va='top', fontsize=12, color='red')

plt.show()

# Построение Partial Dependence Plots и Individual Conditional Expectation графиков для важных фич
important_features = feature_importances['Feature'].head(5).tolist()  # Берем топ-5 важных фич

# Partial Dependence Plots
fig, ax = plt.subplots(figsize=(12, 8))
display = PartialDependenceDisplay.from_estimator(model, X_train, important_features, target=0, ax=ax)  # Указываем целевой класс
fig.suptitle('Partial Dependence Plots')
plt.show()

# Individual Conditional Expectation (ICE) графики
fig, ax = plt.subplots(figsize=(12, 8))
display = PartialDependenceDisplay.from_estimator(model, X_train, important_features, target=0, kind='both', ax=ax)  # Указываем целевой класс
fig.suptitle('Individual Conditional Expectation (ICE) Plots')
plt.show()
