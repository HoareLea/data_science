# Define a decorator function which takes a function as an argument and returns a function (the wrapper)
def uppercase_decorator(func):
    def wrapper():
        original_result = func()  # Call the original function
        modified_result = original_result.upper()  # Modify its result
        return modified_result
    return wrapper


# Define a function and add the decorator
@uppercase_decorator
def greet():
    return "hello world!"


# Call the decorated function
output_with_decorator = greet()
print(output_with_decorator)  # Output: HELLO WORLD!
