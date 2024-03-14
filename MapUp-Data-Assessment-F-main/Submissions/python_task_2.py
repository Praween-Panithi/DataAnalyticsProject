import pandas as pd
import math

df = pd.read_csv('dataset-3.csv')

def calculate_distance_matrix(input_df):
	ids_start = df["id_start"].values
	ids_end = df["id_end"].values
	ids_start = ids_start.tolist()
	ids_end = ids_end.tolist()
	ids1 = list(set(ids_start + ids_end))
	ids1.sort()
	unique_ids = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))
	unique_ids = pd.to_numeric(unique_ids, errors='coerce')
	ids = pd.DataFrame(unique_ids, columns=['ID'])
	# Create a pivot table for the distance matrix
	distance_matrix = pd.pivot_table(df, values='distance', index='id_start', columns='id_end', fill_value=0)

	# Reindex the distance_matrix to include all unique IDs
	distance_matrix = distance_matrix.reindex(index=unique_ids, columns=unique_ids, fill_value=0.0)
	# print(distance_matrix)
	distance_matrix = distance_matrix + distance_matrix.T

            
	for row in range(len(ids1)):
		for col in range(len(ids1)):
			#print(row)
			if ids1[row] != ids1[col] and col > row and ids1[row] != ids1[col-1]:
				distance_matrix[ids1[row]][ids1[col]] = distance_matrix[ids1[row]][ids1[col-1]] + distance_matrix[ids1[col-1]][ids1[col]]
	lower_triangle_mask = np.tril(np.ones(distance_matrix.shape), k=-1).astype(bool)
	transpose = distance_matrix.T
	transpose[lower_triangle_mask] = 0
	ans = transpose + distance_matrix
	return ans

def unroll_distance_matrix(input_df):
	input_df[lower_triangle_mask] = 0
	# unroll_distance = unroll_distance_matrix(ans) 
	unroll_df = pd.DataFrame({'id_start': [], 'id_end' : [], 'distance' : []})
	for row in range(len(ids1)):
		for col in range(len(ids1)):
			if row != col and col > row:
#           	print(ans[ids1[row]][ids1[col]])
				unroll_df.loc[len(unroll_df)] = {'id_start' : ids1[row], 'id_end': ids1[col], 'distance' : ans[ids1[col]][ids1[row]]}
	return unroll_df

def find_ids_within_ten_percentage_threshold(df, reference_value):
    # Calculate average distance for each id_start
    average_per_group = df.groupby('id_start')['distance'].mean().reset_index()

    # Find the average distance for the given reference value
    ref_value = average_per_group[average_per_group['id_start'] == reference_value]['distance'].values[0]

    # Calculate the range within 10% of the reference value
    ans_max = math.floor(ref_value + (10 * ref_value) / 100)
    ans_min = math.ceil(ref_value - (10 * ref_value) / 100)

    # Filter the DataFrame based on the condition
    condition = (average_per_group['distance'] >= ans_min) & (average_per_group['distance'] <= ans_max)

    # Return the sorted list of values from id_start column
    sorted_ids = average_per_group.loc[condition].sort_values(by='id_start')['id_start']

    return sorted_ids

def calculate_toll_rate(df):
    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Create new columns for each vehicle type and calculate toll rates
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        column_name = f'{vehicle_type}'
        df[column_name] = df['distance'] * rate_coefficient
    df = df.drop(columns='distance')
    return df

distance_matrix = calculate_distance_matrix(df)
unrolled_distance_matrix = unroll_distance_matrix(distance_matrix)
threshold_range = find_ids_within_ten_percentage_threshold(unrolled_distance_matrix, '')
toll_rates = calculate_toll_rate(unrolled_distance_matrix)