import pandas as pd
import numpy as np

#Function to generate car matrix
def generate_car_matrix(df)->pd.DataFrame:
    df.set_index("id_1",inplace=True)
    df = df.pivot(columns="id_2",values="car")
    #Function to replace NaN values with '0'
    df.fillna(0, inplace=True)

    return df

#Function to count each car type values 
def get_type_count(df)->dict:
    #Condition to check whether car_type column is present in DataFrame or not.
    #If above condition satisfy then drop function drops the car car_type column from DataFrame
    if 'car_type' in df.columns:
        df.drop('car_type', axis=1, inplace=True)
    df['car_type'] = np.where(df['car'] <= 15, 'Low', np.where(df['car'] > 25, 'High', 'Medium'))
    car_type_counts = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    sorted_car_type_counts = dict(sorted(car_type_counts.items()))

    return sorted_car_type_counts

#Function to return bus indexes as a list
def get_bus_indexes(df)->list:
    bus_mean = df['bus'].mean()
    
    # Identify indices where 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()
    
    # Sort the indices in ascending order
    bus_indexes_list = sorted(bus_indexes)

    return bus_indexes_list

#Function to return route values based on the condition given in question 4
def filter_routes(df)->list:
 
    average_truck_per_route = df.groupby('route')['truck'].mean()

    # Filter routes where the average truck value is greater than 7
    selected_routes = average_truck_per_route[average_truck_per_route > 7]

    # Return the sorted list of selected routes
    routes = selected_routes.index.tolist()
    routes.sort()  # Sorting the routes

    return routes
    
#Function to multiply values with a given values in question 5
def custom_multiplier(value):
    if value > 20:
        return value * 0.75
    else:
        return value * 1.25

#Function For question 5 which is using custom_multiplier Function
def multiply_matrix(matrix)->pd.DataFrame:
    mul_matrix= matrix.applymap(custom_multiplier)

    return mul_matrix


#Function to check each id pair have valid timestamp or not, It returns output values as boolean values
def time_check(df)->pd.Series:
    # Write your logic here
    start_timestamp = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], errors='coerce')

    # Combine 'endDay' and 'endTime' to create an end timestamp
    end_timestamp = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], errors='coerce')

    # Create a boolean series indicating if each (id, id_2) pair has incorrect timestamps
    completeness_check = (
        (start_timestamp.dt.hour == 0) & (start_timestamp.dt.minute == 0) & (start_timestamp.dt.second == 0) &
        (end_timestamp.dt.hour == 23) & (end_timestamp.dt.minute == 59) & (end_timestamp.dt.second == 59) &
        (pd.to_timedelta(end_timestamp - start_timestamp).dt.total_seconds() == 24 * 3600 * 7) &
        (start_timestamp.dt.dayofweek == 0) & (end_timestamp.dt.dayofweek == 6)
    )

    # Group by 'id' and 'id_2' and check if any pair has incorrect timestamps
    result = completeness_check.groupby([df['id'], df['id_2']]).any()

    return result

#reading dataset-1.csv and appending to df
df = pd.read_csv(r'datasets\dataset-1.csv')

#Here we are calling all the functions for execution
df_car_matrix = generate_car_matrix(df)
print(df_car_matrix)


df_type_count = get_type_count(df)
print(df_type_count)


df_bus_indexes = get_bus_indexes(df)
print(df_bus_indexes)


df_routes = filter_routes(df)
print(df_routes)


df_mul_matrx = multiply_matrix(df_car_matrix)
print(df_mul_matrx)

#reading dataset-2.csv and appending to df
df = pd.read_csv(r'datasets\dataset-2.csv')
df_time_bool = time_check(df)
print(df_time_bool)
