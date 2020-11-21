import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
import collections
from sklearn.linear_model import LinearRegression

# Import and Clean Time Series
dataset = pd.read_csv("CumulativeCases.csv")

dates = dataset['Date']
colombia_dataset = dataset['Colombia']
belgium_dataset = dataset['Belgium']

# Create lists from the datasets.
list_colombia = list(colombia_dataset)
list_belgium = list(belgium_dataset)


#### Measures of Central Tendency

def central_tendency_spread():
    #### MEAN ###
    mean_colombia = colombia_dataset.mean()
    mean_belgium = belgium_dataset.mean()

    print("Measures of Central Tendency", end="\n\n")
    print("Colombia Mean: ", mean_colombia)
    print('Belgium Mean: ', mean_belgium)

    #### MEDIAN
    median_colombia = colombia_dataset.median()
    median_belgium = belgium_dataset.median()

    print("Colombia Median: ", median_colombia)
    print('Belgium Median: ', median_belgium)

    #### MODE
    mode_colombia = colombia_dataset.mode()
    mode_belgium = belgium_dataset.mode()

    print("Colombia Mode: ", int(mode_colombia))
    print('Belgium Mode: ', int(mode_belgium), end="\n\n\n")

    variance_colombia = statistics.variance(list(colombia_dataset))
    variance_belgium = statistics.variance(list(belgium_dataset))
    
    print("Measures of Spread", end="\n\n")

    print("Colombia Variance: ", variance_colombia)
    print("Belgium Variance: ", variance_belgium)

    pvariance_colombia = statistics.pvariance(list(colombia_dataset))
    pvariance_belgium = statistics.pvariance(list(belgium_dataset))

    print("Colombia Population Variance: ", pvariance_colombia)
    print("Belgium Population Variance: ", pvariance_belgium)

    stdev_colombia = statistics.stdev(list(colombia_dataset))
    stdev_belgium = statistics.stdev(list(belgium_dataset))

    print("Colombia Standard Deviation: ", stdev_colombia)
    print("Belgium Standard Deviation: ", stdev_belgium)

    pstdev_colombia = statistics.pstdev(list(colombia_dataset))
    pstdev_belgium = statistics.pstdev(list(belgium_dataset))

    print("Colombia Population Standard Deviation: ", pstdev_colombia)
    print("Belgium Population Standard Deviation: ", pstdev_belgium)
    # Write Results into XSLX File

    data1 = {
        'Measures of Central Tendency' : ['Mean', 'Median', 'Mode'],
        'Belgium': [int(mean_belgium), int(median_belgium), int(mode_belgium)],
        'Colombia' : [int(mean_colombia), int(median_colombia), int(mode_colombia)],
    }

    data2 = {
        'Measures of Spread' : ['Variance', 'Population Variance', 'Standard Deviation', 'Population Standard Deviation'],
        'Belgium' : [int(variance_belgium), int(pvariance_belgium), int(stdev_belgium), int(pstdev_belgium)],
        'Colombia' : [int(variance_colombia), int(pvariance_colombia), int(stdev_colombia), int(pstdev_colombia)]
    }

    df1 = pd.DataFrame(data1, columns = ['Measures of Central Tendency', 'Belgium', 'Colombia'])
    df2 = pd.DataFrame(data2, columns = ['Measures of Spread', 'Belgium', 'Colombia'])

    # Save Results to Excel Files
    df1.to_excel('cent_tend.xlsx', index=False, header=True)
    df2.to_excel('spread.xlsx', index=False, header=True)


### FREQUENCIES
def frequencies():
    freq_colombia = collections.Counter(list_colombia)
    freq_belgium = collections.Counter(list_belgium)

    print("Colombia Frequencies: ", freq_colombia, end="\n\n")
    print("Belgium Frequencies: ", freq_belgium)

    # PLOT HISTOGRAM
    plt.style.use('ggplot')
    # Belgium
    plt.hist(list_belgium, bins=10, label="Belgium")

    # Colombia
    plt.hist(list_colombia, bins=10, label="Colombia")

    # Plot
    plt.legend()
    plt.show()


# MOVING AVERAGES & VOLATILITY
window_size = 10
# convert list to series
belgium_series = pd.Series(list_belgium)
belgium_windows = belgium_series.rolling(window_size)

colombia_series = pd.Series(list_colombia)
colombia_windows = colombia_series.rolling(window_size)

# remove NaN
belgium_moving_averages = belgium_windows.mean().tolist()[window_size - 1:]
colombia_moving_averages = colombia_windows.mean().tolist()[window_size - 1:]

def moving_averages():
    print(belgium_moving_averages)
    print(colombia_moving_averages)
    # Plot Moving Averages
    plt.plot(belgium_moving_averages, label="Belgium")
    plt.plot(colombia_moving_averages, label="Colombia")
    plt.legend()
    plt.show()

# Volatility
belgium_volatility = belgium_windows.std(ddof=0).tolist()[window_size - 1:]
colombia_volatility = colombia_windows.std(ddof=0).tolist()[window_size - 1:]

def volatility():
    print(belgium_volatility)
    print(colombia_volatility)
    # Plot Volatility
    plt.plot(belgium_volatility, label="Belgium")
    plt.plot(colombia_volatility, label="Colombia")
    plt.legend()
    plt.show()


def write_avg_vol_to_csv():
    # Write Measures of Volatility and Average to .csv file
    data = {
        'Belgium Volatility': belgium_volatility,
        'Belgium Average' : belgium_moving_averages,
        'Colombia Volatility' : colombia_volatility,
        'Colombia Average' : colombia_moving_averages
    }

    df = pd.DataFrame(data, columns = ['Belgium Volatility', 'Belgium Average', 'Colombia Volatility', 'Colombia Average'])

    # Save Results to Excel Files
    df.to_csv('vol_avg.csv', index=False, header=True)

# Linear Regression
def linear_regression():
    x = np.array(list_belgium).reshape(-1, 1)
    y = np.array(list_colombia)

    model = LinearRegression()
    model.fit(x, y)

    result = model.score(x, y)
    print(result)

    # Plot scatter plot to determine linear correlation
    x2 = np.array(list_belgium) # create 1D array for plot

    plt.plot(x2, y, 'o')
    m, b = np.polyfit(x2, y, 1)
    plt.plot(x2, m*x2, + b)
    plt.show()