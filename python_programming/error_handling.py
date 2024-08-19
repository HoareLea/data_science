# Error Handling in Python

# SyntaxError example
print("### SyntaxError Example ###")
try:
    # Uncommenting the following line will cause a SyntaxError due to missing closing parenthesis
    # eval('print("Hello)')
    pass
except SyntaxError as e:
    print(f"SyntaxError: {e}")
print()

# NameError example
print("### NameError Example ###")
try:
    # Attempting to use an undefined variable
    print(undefined_variable)
except NameError as e:
    print(f"NameError: {e}")
print()

# TypeError example
print("### TypeError Example ###")
try:
    # Trying to concatenate a string with an integer
    result = "Hello" + 5
except TypeError as e:
    print(f"TypeError: {e}")
print()

# IndexError example
print("### IndexError Example ###")
try:
    # Accessing an index that does not exist in the list
    my_list = [1, 2, 3]
    print(my_list[5])
except IndexError as e:
    print(f"IndexError: {e}")
print()

# ValueError example
print("### ValueError Example ###")
try:
    # Converting an invalid string to an integer
    num = int("invalid")
except ValueError as e:
    print(f"ValueError: {e}")
print()

# KeyError example
print("### KeyError Example ###")
try:
    # Accessing a non-existent key in a dictionary
    my_dict = {"name": "Alice"}
    print(my_dict["age"])
except KeyError as e:
    print(f"KeyError: {e}")
print()

# FileNotFoundError example
print("### FileNotFoundError Example ###")
try:
    # Trying to open a file that does not exist
    with open("non_existent_file.txt", "r") as file:
        content = file.read()
except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}")
print()

# Handling multiple exceptions
print("### Handling Multiple Exceptions ###")
try:
    result = 10 / 0
except (ZeroDivisionError, TypeError) as e:
    print(f"Error: {e}")
print()

# Using finally clause
print("### Using Finally Clause ###")
try:
    file = open("example_file.txt", "w")
    file.write("Hello, world!")
finally:
    # The finally block is always executed, regardless of whether an exception is raised or not.
    file.close()
    print("File closed")
print()

# Using else clause
print("### Using Else Clause ###")
try:
    num = int("123")
except ValueError as e:
    print(f"ValueError: {e}")
else:
    print("Conversion successful, number:", num)
print()


# Custom exception
class CustomError(Exception):
    """Custom exception class."""
    pass


print("### Custom Exception ###")
try:
    raise CustomError("This is a custom error message.")
except CustomError as e:
    print(f"CustomError: {e}")
print()
