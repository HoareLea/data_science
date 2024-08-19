# Define a simple base class
class Car:
    # Class attribute
    wheels = 4

    # Initializer / Instance attributes
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year

    # Instance method make use of the instance attributes
    def display_info(self):
        return f"{self.year} {self.make} {self.model}"

    # Static methods are independent of the instance attributes
    @staticmethod
    def about():
        return "This is a car class."

    # Properties allow for the method to be treated like an attribute (i.e. the parenthesis are not needed when calling it). These are
    # called 'getters'
    @property
    def warranty_expiration(self):
        return self.year + 5

    # You can define a setter method for the property using @property_name.setter. This allows you to validate or modify the value assigned
    # to the property.
    @warranty_expiration.setter
    def warranty_expiration(self, value):
        if value < self.year:
            raise ValueError("warranty_expiration can't be before car's year")
        self._radius = value

    # You can define a deleter method using @property_name.deleter, which defines behavior when the property is deleted using the del
    # statement.
    @warranty_expiration.deleter
    def warranty_expiration(self):
        del self._radius


# Classes can inherit from other classes
class ElectricCar(Car):
    def __init__(self, make, model, year, battery_capacity):
        super().__init__(make, model, year)  # This sets the make, model and year as in the base Car class
        self.battery_capacity = battery_capacity  # We can also add additional attributes when we initialise a sub-class

    # We can override the functionality of methods in the base class by defining methods of the same name in the subclass
    def display_info(self):
        return f"{self.year} {self.make} {self.model} (Electric)"


# Initialising objects of the classes
car1 = Car("Toyota", "Camry", 2023)
car2 = ElectricCar("Tesla", "Model S", 2024, "100 kWh")

# Accessing attributes
print(car1.make)  # Output: Toyota
print(car2.battery_capacity)  # Output: 100 kWh

# Calling methods
print(car1.display_info())  # Output: 2023 Toyota Camry
print(car2.display_info())  # Output: 2024 Tesla Model S (Electric)

# Calling static method
print(Car.about())  # Output: This is a car class.

# Inheritance example
print(isinstance(car1, ElectricCar))  # Output: False, car1 is not an instance of ElectricCar
print(isinstance(car2, Car))  # Output: True, car2 is an instance of Car since it inherits from the Car class

# Accessing inherited methods, notice how this is utilises the method from the sub-class rather than the base class
print(car2.display_info())  # Output: 2024 Tesla Model S (Electric)

# Accessing class attribute
print(car1.wheels)  # Output: 4, wheels is a class attribute shared by all instances of Car

# Modifying instance attributes
car1.year = 2025
print(car1.display_info())  # Output: 2025 Toyota Camry

# Notice how since warranty_expiration is a property, we don't need parenthesis to access it
print(car1.warranty_expiration)  # Output: 2023
