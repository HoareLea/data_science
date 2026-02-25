import pandas as pd

# DataFrame

# Creating a DataFrame from a dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [24, 27, 22, 32],
    'City': ['New York', 'New York', 'Chicago', 'Houston']
}
df = pd.DataFrame(data)
print("DataFrame created from dictionary:\n", df)

# Viewing the first few rows of the DataFrame
print("\nFirst few rows of the DataFrame:\n", df.head())

# Viewing the last few rows of the DataFrame
print("\nLast few rows of the DataFrame:\n", df.tail())

# Getting basic information about the DataFrame
print("\nInfo about the DataFrame:")
df.info()

# Descriptive statistics for the DataFrame
print("\nDescriptive statistics for the DataFrame:\n", df.describe())

# Selecting a single column
print("\nSelecting the 'Name' column:\n", df['Name'])

# Selecting multiple columns
print("\nSelecting 'Name' and 'Age' columns:\n", df[['Name', 'Age']])

# Selecting rows by index
print("\nSelecting the first row:\n", df.iloc[0])

# Selecting rows by label
print("\nSelecting rows where Age is greater than 25:\n", df[df['Age'] > 25])

# Selecting multiple rows by label
print("\nSelecting rows where Age is greater than 25 and Name is not Bob:\n", df[(df['Age'] > 25) & (df['Name'] != 'Bob')])

# Selecting rows with values in a list
print("\nSelecting rows where Name is Bob or Alice\n", df[df['Name'].isin(['Bob', 'Alice'])])

# Adding a new column
df['Salary'] = [50000, 60000, 45000, 70000]
print("\nDataFrame after adding a new column:\n", df)

# Removing a column
df.drop('City', axis=1, inplace=True)  # inplace=True overwrites the original df rather than returning the updates one separately
print("\nDataFrame after removing the 'City' column:\n", df)
df['City'] = ['New York', 'New York', 'Chicago', 'Houston'] # add the column back

# Renaming columns
df.rename(columns={'Name': 'Employee Name', 'Age': 'Employee Age'}, inplace=True)
print("\nDataFrame after renaming columns:\n", df)

# Inserting missing data
df.loc[2, 'Employee Age'] = None
print("\nDataFrame with a missing value:\n", df)

# Filling missing data
df['Employee Age'].fillna(df['Employee Age'].mean(), inplace=True)
print("\nDataFrame after filling missing values:\n", df)

# Grouping data
grouped_df = df.groupby('City')['Employee Age'].mean()
print("\nGrouped DataFrame:\n", grouped_df)

# Merging DataFrames
df1 = pd.DataFrame({
    'ID': [1, 2, 3],
    'Name': ['Alice', 'Bob', 'Charlie']
})
df2 = pd.DataFrame({
    'ID': [3, 4, 5],
    'City': ['Chicago', 'Houston', 'Phoenix']
})
merged_df = pd.merge(df1, df2, on='ID', how='inner')
print("\nMerged DataFrame:\n", merged_df)

# Concatenating DataFrames
concat_df = pd.concat([df1, df2], axis=0, ignore_index=True)
print("\nConcatenated DataFrame:\n", concat_df)

# Pivoting DataFrame
pivot_data = {
    'Date': ['2023-07-01', '2023-07-01', '2023-07-02', '2023-07-02'],
    'City': ['New York', 'Los Angeles', 'New York', 'Los Angeles'],
    'Sales': [100, 150, 200, 250]
}
pivot_df = pd.DataFrame(pivot_data)
pivot_table = pivot_df.pivot_table(values='Sales', index='Date', columns='City', aggfunc='sum')
print("\nPivot Table:\n", pivot_table)

# Reshaping with melt
melted_df = pd.melt(df, id_vars=['Employee Name'], value_vars=['Employee Age', 'Salary'])
print("\nMelted DataFrame:\n", melted_df)

# Using apply to transform data
df['Salary in K'] = df['Salary'].apply(lambda x: x / 1000)
print("\nDataFrame after using apply:\n", df)

# Reading data from a CSV file (uncomment the following lines if you have a CSV file to read)
df_from_csv = pd.read_csv('saved_examples/example.csv')
print("\nDataFrame read from CSV file:\n", df_from_csv)

# Writing DataFrame to a CSV file
df.to_csv('saved_examples/example.csv', index=False)
print("\nDataFrame written to 'example.csv'")

# For demonstration purposes, print out the final DataFrame
print("\nFinal DataFrame:\n", df)

# Series

# Creating a Series from a list
s1 = pd.Series([10, 20, 30, 40, 50])
print("Series created from a list:\n", s1)

# Creating a Series from a dictionary
s2 = pd.Series({'a': 1, 'b': 2, 'c': 3, 'd': 4})
print("\nSeries created from a dictionary:\n", s2)

# Creating a Series with custom index
s3 = pd.Series([100, 200, 300], index=['x', 'y', 'z'])
print("\nSeries with custom index:\n", s3)

# Accessing elements by position
print("\nElement at position 1 in s1:\n", s1[1])

# Accessing elements by index label
print("\nElement with index 'b' in s2:\n", s2['b'])

# Slicing a Series
print("\nSlicing s1 from index 1 to 3:\n", s1[1:4])

# Checking for null values
s4 = pd.Series([1, 2, None, 4, 5])
print("\nSeries with a missing value:\n", s4)
print("\nChecking for null values in s4:\n", s4.isnull())

# Filling missing values
s4_filled = s4.fillna(0)
print("\nSeries after filling missing values in s4:\n", s4_filled)

# Performing operations on Series
s5 = pd.Series([10, 20, 30])
s6 = pd.Series([1, 2, 3])
print("\nAdding s5 and s6:\n", s5 + s6)
print("\nMultiplying s5 by 2:\n", s5 * 2)

# Applying functions to Series
print("\nApplying square function to s5:\n", s5.apply(lambda x: x**2))

# Descriptive statistics for a Series
print("\nDescriptive statistics for s5:\n", s5.describe())

# Series alignment in operations
s7 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s8 = pd.Series([4, 5, 6], index=['b', 'c', 'd'])
print("\nAdding s7 and s8 (with alignment):\n", s7 + s8)

# Checking for existence of an index
print("\nChecking if 'a' is in s7:\n", 'a' in s7)
print("Checking if 'd' is in s7:\n", 'd' in s7)

# Converting Series to DataFrame
s9 = pd.Series([100, 200, 300], index=['A', 'B', 'C'])
df = s9.to_frame(name='Value')
print("\nSeries converted to DataFrame:\n", df)

# Creating a Series from a DataFrame column
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [24, 27, 22]
}
df2 = pd.DataFrame(data)
age_series = df2['Age']
print("\nSeries created from DataFrame column:\n", age_series)

# Concatenating Series
s10 = pd.Series([1, 2, 3])
s11 = pd.Series([4, 5, 6])
concat_series = pd.concat([s10, s11])
print("\nConcatenated Series:\n", concat_series)
