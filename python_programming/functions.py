# Python Functions Explanation

# Function without parameters
def greet():
    """This function prints a greeting."""
    print("Hello, welcome to Python functions!")


# Calling the function
print("### Function without parameters ###")
greet()
print()


# Function with parameters and return value
def add_numbers(a, b):
    """This function adds two numbers and returns the result."""
    return a + b


# Calling the function with arguments
print("### Function with parameters and return value ###")
result = add_numbers(3, 5)
print(f"Result of adding 3 and 5: {result}")
print()


# Function with default arguments
def greet_person(name, greeting="Hello"):
    """This function greets a person with a specified greeting."""
    print(f"{greeting}, {name}!")


# Calling the function with different arguments
print("### Function with default arguments ###")
greet_person("Alice")
greet_person("Bob", "Hi")
print()


# Function with variable-length arguments
def calculate_sum(*args):
    """This function calculates the sum of variable-length arguments."""
    total = 0
    for num in args:
        total += num
    return total


# Calling the function with different number of arguments
print("### Function with variable-length arguments ###")
result1 = calculate_sum(1, 2, 3)
result2 = calculate_sum(10, 20, 30, 40, 50)
print(f"Sum of 1, 2, and 3: {result1}")
print(f"Sum of 10, 20, 30, 40, and 50: {result2}")
print()


# Function with keyword arguments
def print_info(name, age, **kwargs):
    """This function prints information about a person with additional keyword arguments."""
    print(f"Name: {name}, Age: {age}")
    for key, value in kwargs.items():
        print(f"{key}: {value}")


# Calling the function with keyword arguments
print("### Function with keyword arguments ###")
print_info("Alice", 30, city="New York", occupation="Engineer")
print_info("Bob", 25, city="San Francisco", hobby="Reading", pets="Cat")
print()


# Function with docstring
def function_with_docstring():
    """
    This is a function with a docstring.
    It serves as documentation for the function.
    """
    pass


# Accessing docstring using __doc__ attribute
print("### Function with docstring ###")
print(function_with_docstring.__doc__)
