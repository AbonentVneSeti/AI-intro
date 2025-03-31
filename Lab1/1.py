import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def read_csv(file_path : str) -> list[list]:
    data = list()

    with open(file_path) as f:
        reader = csv.reader(f)
        data.append(next(reader))
        for row in reader:
            data.append(list())
            for i in row:
                data[-1].append(float(i))
    return data

def stat_analyisis(data : list[list])-> dict:
    numvalues = len(data) - 1
    minvalues = dict()
    maxvalues = dict()
    averagevalues = dict()

    headers = data[0]

    for i in headers:
        minvalues[i] = 9999999.0
        maxvalues[i] = -9999999.0
        averagevalues[i] = 0


    for row in data[1:]:
        for i in range(len(headers)):
            if row[i] < minvalues[headers[i]]:
                minvalues[headers[i]] = row[i]
            if row[i] > maxvalues[headers[i]]:
                maxvalues[headers[i]] = row[i]
            averagevalues[headers[i]] += row[i]

    for i in headers:
        averagevalues[i]/= numvalues

    stats = {"numvalues": numvalues, "minvalues": minvalues, "maxvalues": maxvalues, "averagevalues": averagevalues}
    return stats

def calculate_coefficients(data : list[list],x_ind : int,y_ind : int) -> list[float]:
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

def func(x : float, line_coeffs : list) -> float:
    return line_coeffs[0] + x*line_coeffs[1]

def inv_func(y : float, line_coeffs : list) -> float:
    return (y - line_coeffs[0])/line_coeffs[1]

def get_rects(x : list ,y : list, line_coeffs : list)-> list[Rectangle]:
    rects = list()
    #rects.append(Rectangle((10,10),20,20,edgecolor = 'black',facecolor = 'None'))

    for i in range(len(x)):
        point = (x[i],y[i])
        delta_y = point[1] - func(point[0],line_coeffs)
        delta_x = abs(point[0] - inv_func(point[1],line_coeffs))
        if delta_y>0:
            rects.append(Rectangle(point,delta_x,-delta_y,edgecolor = 'black',facecolor = 'None'))
        elif delta_y < 0:
            rects.append(Rectangle(point,-delta_x,-delta_y,edgecolor = 'black',facecolor = 'None'))

    return rects

def draw(x : list ,y : list, line_coeffs : list, x_axis = "X", y_axis = "Y") -> None:
    plt.figure(figsize = (16,8))
    plt.subplots_adjust(hspace=0.3)

    plt.subplot(2,2,1)
    plt.scatter(x, y)
    plt.title("Все точки")
    ax = plt.gca()
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    plt.subplot(2,2,2)
    plt.scatter(x, y)
    plt.axline((1,line_coeffs[0]+line_coeffs[1]*1),(10,line_coeffs[0]+line_coeffs[1]*10),color = 'red')
    plt.title("Линейная регрессия")
    ax = plt.gca()
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    plt.subplot(2,2,3)
    plt.scatter(x, y)
    plt.axline((1, func(1,line_coeffs)), (10, func(10,line_coeffs)), color = 'red')
    for i in get_rects(x,y, line_coeffs):
        plt.gca().add_patch(i)
    plt.title("Линейная регрессия с квадратами MSE")
    ax = plt.gca()
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    plt.show()

def main():
    data = read_csv("student_scores.csv")
    print("Считанные данные:\n",data,sep='')

    stats = stat_analyisis(data)
    print("Статистика:\n",stats,sep='')

    x_ind = 0
    y_ind = 0

    print("\nДоступные столбцы:")
    for i in range( len(data[0])):
        print(f"{i} : {data[0][i]}")

    input_flag = True
    while(input_flag):
        #print("\nВыберите, какой столбец будет отвечать за x:")
        x_ind = int(input("Выберите, какой столбец будет отвечать за x: "))

        #print("\nВыберите, какой столбец будет отвечать за y:")
        y_ind = int(input("Выберите, какой столбец будет отвечать за y: "))

        if x_ind == y_ind or x_ind > len(data[0]) or x_ind < 0 or y_ind > len(data[0]) or y_ind < 0:
            print("Ввод некорректен, повторите попытку.")
        else:
            print("Принято!")
            input_flag = False

    x = [i[x_ind] for i in data[1:]]
    y = [i[y_ind] for i in data[1:]]

    #4
    coeffs = calculate_coefficients(data,x_ind,y_ind)
    print(f"Искомая функция: y = {coeffs[0]} + {coeffs[1]}*x")


    draw(x,y,coeffs,data[0][x_ind],data[0][y_ind])

if __name__ == "__main__":
    main()