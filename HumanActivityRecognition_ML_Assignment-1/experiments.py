import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

np.random.seed(42)
num_average_time = 10  # Number of iterations for quicker results
N_values = [100, 500, 1000, 2000, 5000]  # Different sample sizes
M_values = [5, 10, 20, 50, 100]  # Different numbers of features

# Function to create fake data
def create_fake_data(N, M, case):
    if case == 'Discrete_Discrete':
        X = np.random.randint(0, 2, size=(N, M))  # Binary features
        y = np.random.randint(0, 2, size=N)       # Binary target variable
    elif case == 'Discrete_Real':
        X = np.random.randint(0, 2, size=(N, M))  # Binary features
        y = np.random.uniform(0, 10, size=N)      # Continuous target variable
    elif case == 'Real_Discrete':
        X = np.random.uniform(0, 10, size=(N, M)) # Continuous features
        y = np.random.randint(0, 2, size=N)       # Binary target variable
    elif case == 'Real_Real':
        X = np.random.uniform(0, 10, size=(N, M)) # Continuous features
        y = np.random.uniform(0, 10, size=N)      # Continuous target variable
    return X, y

# Function to calculate average time
def measure_time(N, M, case, num_average_time):
    fit_times = []
    predict_times = []
    
    for _ in range(num_average_time):
        X, y = create_fake_data(N, M, case)
        X_train, X_test = X[:int(0.8*N)], X[int(0.8*N):]
        y_train, y_test = y[:int(0.8*N)], y[int(0.8*N):]

        # Initialize the correct model
        if case in ['Discrete_Discrete', 'Real_Discrete']:  # Classification
            dt = DecisionTreeClassifier()
        else:  # Regression
            dt = DecisionTreeRegressor()
        
        # Measure time for fitting
        start_time = time.time()
        dt.fit(X_train, y_train)
        fit_times.append(time.time() - start_time)
        
        # Measure time for prediction
        start_time = time.time()
        dt.predict(X_test)
        predict_times.append(time.time() - start_time)
    
    avg_fit_time = np.mean(fit_times)
    avg_predict_time = np.mean(predict_times)
    return avg_fit_time, avg_predict_time

# Function to plot results
def plot_results(results):
    fig, axes = plt.subplots(4, 2, figsize=(18, 24), sharex=True)

    cases = ['Discrete_Discrete', 'Discrete_Real', 'Real_Discrete', 'Real_Real']
    for i, case in enumerate(cases):
        fit_times_case, predict_times_case = results[case]
        
        # Plot fitting times
        for j, M in enumerate(M_values):
            axes[i, 0].plot(N_values, fit_times_case[j], marker='o', linestyle='-', label=f'M={M}')
        axes[i, 0].set_title(f'{case} - Fitting Time')
        axes[i, 0].set_ylabel('Time (seconds)')
        axes[i, 0].legend(title='Number of Features (M)')
        axes[i, 0].grid(True)
        
        # Plot predicting times
        for j, M in enumerate(M_values):
            axes[i, 1].plot(N_values, predict_times_case[j], marker='o', linestyle='--', label=f'M={M}')
        axes[i, 1].set_title(f'{case} - Predicting Time')
        axes[i, 1].set_ylabel('Time (seconds)')
        axes[i, 1].legend(title='Number of Features (M)')
        axes[i, 1].grid(True)

    # Set common labels
    for ax in axes.flat:
        ax.set_xlabel('Number of Samples (N)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
    plt.suptitle('Decision Tree Runtime Complexity', fontsize=16)
    plt.show()

# Main code to run experiments and plot results
results = {
    'Discrete_Discrete': ([], []),
    'Discrete_Real': ([], []),
    'Real_Discrete': ([], []),
    'Real_Real': ([], [])
}

for case in results.keys():
    print(f'Running experiments for {case}')
    fit_times_case = []
    predict_times_case = []
    for M in M_values:
        fit_times = []
        predict_times = []
        for N in N_values:
            avg_fit_time, avg_predict_time = measure_time(N, M, case, num_average_time)
            fit_times.append(avg_fit_time)
            predict_times.append(avg_predict_time)
        
        fit_times_case.append(fit_times)
        predict_times_case.append(predict_times)
    
    results[case] = (fit_times_case, predict_times_case)

# Plot results
plot_results(results)


