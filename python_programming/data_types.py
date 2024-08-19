# Data Types

# Integer
# Integers are whole numbers. We can perform basic mathematical operations on them
a = 2
b = 3
int_sum = a + b
int_diff = a - b
int_prod = a * b
int_div = a // b  # Integer division
int_power = a ** b  # a ^ b
print("Integer operations:")
print(f"{a} + {b} = {int_sum}")
print(f"{a} - {b} = {int_diff}")
print(f"{a} * {b} = {int_prod}")
print(f"{a} // {b} = {int_div}")
print(f"{a} ** {b} = {int_power}")
print()

# Float
# Floats are decimal numbers. Again, we can perform basic mathematical operations on them
x = 5.5
y = 2.5
float_sum = x + y
float_diff = x - y
float_prod = x * y
float_div = x / y
print("Float operations:")
print(f"{x} + {y} = {float_sum}")
print(f"{x} - {y} = {float_diff}")
print(f"{x} * {y} = {float_prod}")
print(f"{x} / {y} = {float_div}")
print()

# String
# Strings are text. We can perform operations like combining them or changing their case.
str1 = "Hello"
str2 = "World"
str_concat = str1 + " " + str2
str_upper = str1.upper()
str_lower = str2.lower()
print("String operations:")
print(f"Concatenation: {str_concat}")
print(f"Uppercase: {str_upper}")
print(f"Lowercase: {str_lower}")
print()

# List
# Lists are ordered collections of items. We can concatenate lists, append items, and find their length. Elements are accessed via their
# index starting from 0.
list1 = [1, 2, 3, 4]
list2 = [5, 6, 7, 8]
list_concat = list1 + list2
list1.append(9)
list_length = len(list1)
first_element_list1 = list1[0]
last_element_list2 = list2[-1]
print("List operations:")
print(f"Concatenation: {list_concat}")
print(f"First element of list1 {first_element_list1}")
print(f"Last element of list2 {last_element_list2}")
print(f"Append 9 to list1: {list1}")
print(f"Length of list1: {list_length}")
print()

# Tuple
# Tuples are similar to lists, again we can concatenate tuples and find their length. However lists are immutable meaning once created
# they cannot be modified.
tuple1 = (1, 2, 3, 4)
tuple2 = (5, 6, 7, 8)
tuple_concat = tuple1 + tuple2
tuple_length = len(tuple1)
try:
    tuple1[0] = 4  # This will raise a TypeError
except TypeError as e:
    print(f"Error: {e}")
print("Tuple operations:")
print(f"Concatenation: {tuple_concat}")
print(f"Length of tuple1: {tuple_length}")
print()

# Set
# Sets are unordered collections of unique items. We can perform union, intersection, and difference operations.
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}
set_union = set1.union(set2)
set_intersection = set1.intersection(set2)
set1_difference = set1.difference(set2)
set2_difference = set2.difference(set1)
print("Set operations:")
print(f"Union: {set_union}")
print(f"Intersection: {set_intersection}")
print(f"Difference (set1 - set2): {set1_difference}")
print(f"Difference (set2 - set1): {set2_difference}")
print()

# Dictionary
# Dictionaries store key-value pairs. We can add new key-value pairs, and retrieve keys and values.
dict1 = {"name": "Alice", "age": 25}
dict2 = {"name": "Bob", "age": 30}
dict1["city"] = "New York"
dict_keys = dict1.keys()
dict_values = dict1.values()
print("Dictionary operations:")
print(f"Dictionary 1: {dict1}")
print(f"Dictionary 2: {dict2}")
print(f"Keys in dict1: {list(dict_keys)}")
print(f"Values in dict1: {list(dict_values)}")
print()

# Boolean
# Booleans represent truth values. They can be True or False and we can combine them using logic
bool1 = True
bool2 = False
bool_and = bool1 and bool2
bool_or = bool1 or bool2
bool_not = not bool1
print(f"AND: {bool_and}")
print(f"OR: {bool_or}")
print(f"NOT: {bool_not}")
print()

# None Type
# None is a special constant in Python that represents the absence of a value. We can check if a value is None using is None
none_var = None
print(f"Value of none_var: {none_var}")
print(f"Is none_var None?: {none_var is None}")
print()

# Range
# Range represents a sequence of numbers and is immutable.
my_range = range(5)  # Creates a range from 0 to 4
range_list = list(my_range)
print(f"Range: {my_range}")
print(f"Converted to list: {range_list}")
print()

# Complex numbers
# Complex numbers have a real and imaginary part. We can perform arithmetic operations on them.
complex1 = 3 + 4j
complex2 = 1 - 2j
complex_sum = complex1 + complex2
complex_prod = complex1 * complex2
print(f"Complex number 1: {complex1}")
print(f"Complex number 2: {complex2}")
print(f"Sum: {complex_sum}")
print(f"Product: {complex_prod}")
print()

# Bytes and Bytearray
# Bytes and Bytearray are used to store binary data. Bytes are immutable, while Bytearray is mutable.
bytes_data = b"hello"
bytearray_data = bytearray(b"world")
bytes_concat = bytes_data + bytearray_data
print(f"Bytes data: {bytes_data}")
print(f"Bytearray data: {bytearray_data}")
print(f"Concatenation: {bytes_concat}")
print()
