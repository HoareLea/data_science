
# The factory design pattern aims to centralise the code for object creation. It is particularly useful in the case where there are many
# variants of an object which share some functionality but may have some differences in other aspects. It works as follows:

# 1. Define an abstract base class. This encapsulates all the default behaviour that will be shared across all variants.
# 2. Define concrete implementations of the abstract base class. These inherit from the abstract base class but can overwrite or extend its
#    functionality
# 3. Define an abstract factory class. This should contain a method for creating objects which calls the concrete classes' init methods.

# There are several benefits of utilising the factory pattern:

# 1. All the logic which decides which type of object should be created is housed in one central place (namely the create method within the
#    abstract factory class). Without the factory pattern, this logic would need to be included everywhere that objects are initialised in
#    the client code
# 2. The factory pattern allows us to call methods on objects without explicitly needing to know their class. The specific implementation is
#    automatically handled by the concrete class implementations.
# 3. The functionality that makes object variants different from one another is housed in a single place (namely the concrete classes).
#    Without the factory pattern, every class method would need logic which switches the behaviour depending on the variant. This could get
#    very verbose if there were many variants or if the logic was complex.

# Step 1: Define an abstract base class for Button
class Button:
    def render(self):
        pass


# Concrete implementation of Button for iOS. Notice how this overwrites the default implementation in the base class for Button.
class IOSButton(Button):
    def render(self):
        return "Render an iOS style button"


# Concrete implementation of Button for Android. Notice how this overwrites the default implementation in the base class for Button.
class AndroidButton(Button):
    def render(self):
        return "Render an Android style button"


# Factory class to create buttons
class ButtonFactory:
    def create_button(self, platform):
        if platform == "iOS":
            return IOSButton()
        elif platform == "Android":
            return AndroidButton()
        else:
            raise ValueError(f"Unknown platform {platform}. Cannot create button.")


# Client code that uses the factory
if __name__ == "__main__":
    factory = ButtonFactory()

    button_dict1 = {"platform": "iOS"}
    button_dict2 = {"platform": "Android"}

    # Create the two different button variants. Note how we can use the same .create_button() method to initialise the classes because the
    # abstract factory implements the method from the concrete classes.
    buttons = []
    for button_dict in [button_dict1, button_dict2]:
        button = factory.create_button(platform=button_dict["platform"])
        buttons.append(button)

    # Call the render method for the two different button variants. Note how we can use the .render() method without the client code needing
    # to explicitly state which type of button we are using.
    for button in buttons:
        print(button.render())

    # Without the factory design pattern, the render method would need the logic from the create_button method in ButtonFactory which
    # switches the behaviour based on the type of button. This wouldn't be too bad in this specific example since we only have two variants
    # and one method. However, this logic would be needed to be included in every single method. Additionally this logic could become highly
    # verbose as the number of variants grows.
