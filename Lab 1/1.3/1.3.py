from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

def my_lin_calculate_coefficients(data, x_ind, y_ind):
    n = len(data)-1
    y_sum = 0
    x_sum = 0
    square_x_sum = 0
    x_mult_y_sum = 0
    for i in data[1:]:
        y_sum+=i[y_ind]
        x_sum+=i[x_ind]
        x_mult_y_sum += i[x_ind]*i[y_ind]
        square_x_sum += i[x_ind]*i[x_ind]

    w1 = ( (x_sum*y_sum)/n - x_mult_y_sum )/( (x_sum*x_sum)/n - square_x_sum )

    w0 = y_sum/n - (w1 * x_sum)/n

    return[w0,w1]

def my_lin_regression(x_train, y_train, x_test):
    data= list()
    data.append([None,None])
    for i in range(len(x_train)):
        data.append([x_train[i], y_train[i]])

    coeffs = my_lin_calculate_coefficients(data,0,1)

    my_lin_predictions = list()
    for x in x_test:
        my_lin_predictions.append((coeffs[0] + coeffs[1]*x)[0])
    return [my_lin_predictions,coeffs]

def my_mean_absolute_error(y_test, y_pred):
    sum = 0
    for i in range(len(y_pred)):
        sum += abs(y_test[i]-y_pred[i])
    return sum/len(y_pred)

def my_r2_score(y_test, y_pred):
    SSE = 0
    SST = 0
    avg_y = sum(y_test)/len(y_test)

    for i in range(len(y_pred)):
        SSE += (y_test[i] - y_pred[i])**2
        SST += (y_test[i] - avg_y) ** 2

    return 1 - SSE/SST

def my_mean_absolute_percentage_error(y_test, y_pred):
    sum = 0
    for i in range(len(y_pred)):
        sum += abs(y_test[i] - y_pred[i])/y_test[i]
    return sum / len(y_pred)


def main():
    diabetes = datasets.load_diabetes()

    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target  # уровень глюкозы
    x = df[['bmi']].values
    y = df['target'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

    #Модель sklearn
    sl_reg = LinearRegression()
    sl_reg.fit(x_train, y_train)
    sl_predictions = sl_reg.predict(x_test)

    #Собственный алгоритм
    my_reg = my_lin_regression(x_train,y_train,x_test)
    my_lin_coeffs = my_reg[1]
    my_lin_predictions = my_reg[0]

    #Выводы
    print("Resulting model functions")
    print(f"sklearn: {sl_reg.intercept_:.2f} + x * {sl_reg.coef_[0]:.2f}")
    print(f"my lin reg: {my_lin_coeffs[0][0]:.2f} + x * {my_lin_coeffs[1][0]:.2f}")
    print()
    n = 30
    print(f"Prediction table, first {n} pieces:")
    print('x    real_y  sklearn_y   my_lin_y')
    for i in range(n):
        print(f'{x_test[i][0]:.2f}    {y_test[i]:.2f}  {sl_predictions[i]:.2f}   {my_lin_predictions[i]:.2f}')

    #Метрики
    print('\nSklearn model metrics:')
    print(f"MAE: {mean_absolute_error(y_test, sl_predictions):.2f}")
    print(f"R2: {r2_score(y_test, sl_predictions):.2f}")
    print(f"MAPE: {mean_absolute_percentage_error(y_test, sl_predictions)*100:.2f}%")

    print('\nMy model metrics:')
    print(f"MAE: {my_mean_absolute_error(y_test, my_lin_predictions):.2f}")
    print(f"R2: {my_r2_score(y_test, my_lin_predictions):.2f}")
    print(f"MAPE: {my_mean_absolute_percentage_error(y_test, my_lin_predictions) * 100:.2f}%")

    #Отрисовка
    plt.figure(figsize = (16,8))
    plt.title('Linear regressions')
    plt.xlabel('BMI')
    plt.ylabel('Glucose level')
    plt.scatter(x_train, y_train, color = 'black', label='Train values')
    plt.scatter(x_test, y_test, label = 'Test values')

    plt.axline((x_test[0][0], sl_predictions[0]), (x_test[1][0], sl_predictions[1]), label = 'sklearn regression', color = 'Red', linewidth = 3)
    plt.axline((x_test[0][0], my_lin_predictions[0]), (x_test[1][0], my_lin_predictions[1]), label = 'my linear regression', color = 'Blue',linestyle = 'dashed' , linewidth = 2)


    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
