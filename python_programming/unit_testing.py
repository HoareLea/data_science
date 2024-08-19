from unittest.mock import patch

import pytest

# Unit testing with Pytest
# Typically you would write a separate script for unit testing to separate it from the code you want to test but in this example it is just
# all one script.

# Script to test


def divide(a, b):
    """Return the division of two numbers. Raises ZeroDivisionError if b is zero."""
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b


def divide_then_square(a, b):
    a_divided_by_b = divide(a, b)
    return a_divided_by_b ** 2


# Tests
# In pytest, tests are written as functions and must start with the word test. Assertions are used to check actual values match
# expected values


def test_divide():
    assert divide(6, 2) == 3
    assert divide(-6, -2) == 3
    assert divide(5, 2) == 2.5

    with pytest.raises(ZeroDivisionError):
        divide(1, 0)


# You can parametrise tests using the pytest.mark.parametrize decorator. This loops through different cases.

@pytest.mark.parametrize("a, b, expected", [
    (6, 2, 3),
    (-6, -2, 3),
    (5, 2, 2.5),
    (9, 3, 3)
])
def test_divide(a, b, expected):
    assert divide(a, b) == expected


@pytest.mark.parametrize("a, b", [
    (1, 0),
    (10, 0),
    (-5, 0)
])
def test_divide_zero_division(a, b):
    with pytest.raises(ZeroDivisionError):
        divide(a, b)


# Suppose you want to test a function that depends on another function. For example divide_then_square which depends on divide. We need to
# isolate the functionality of divide_then_square from divide so that a problem with divide does not affect the test for divide_then_square.
# To do this we use the patch decorator to mock/overwrite the value of a variable or function call inside the test.

# In this example we use the patch decorator to mock the return value of the divide function. This is also included as the first argument.
# We pass the value we would like to mock with as an argument in the parametrize call. Inside the test, we overwrite the return value of the
# divide function by setting its .return_value to the value we supplied as an argument.

@pytest.mark.parametrize("a, b, mock_divide_value, expected", [
    (2, 1, 2, 4),
    (10, 2, 5, 25),
])
@patch(__name__ + ".divide")
def test_divide_then_square(mock_divide, a, b, mock_divide_value, expected):
    mock_divide.return_value = mock_divide_value
    result = divide_then_square(a, b)
    assert result == expected


# Note also that you can mock multiple functions in the event that the function you want to test depends on multiple other functions.
# Additionall you can mock variables as well as functions
