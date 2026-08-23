## FIRAS EXTRA: Object Oriented Programming

# You can create classes on the fly and use them aslogical units to store complex data types.
# class Employee:
#     '''
#     Empty Class
#     '''
#     pass

# employee = Employee()
# employee.salary = 122000
# employee.firstname = "alice"
# employee.lastname = "wonderland"

# print(f'Employee salary is {employee.salary}')

# print(employee.firstname + " "
# + employee.lastname + " "
# + str(employee.salary) + "$" )


# You can call a method (function) from inside the __init__

# class MyClass:

#     def __init__(self, my_arg):
#         self.my_attribute = my_arg
#         self.my_method()

#     def my_method(self):
#         print("Hello, world!")

# my_object = MyClass("*my argument*")

# print(my_object.my_attribute)


# You can removr __init__ constructor and let it be passed and created in the back through another class
# @ is called a decorator and can be used to inject properties and function to classes and other functions as well
from dataclasses import dataclass
@dataclass
class Calculator:
    number:int
    
    def __str__(self):
        return f"Number = {self.number}"
    def __repr__(self):
        return f"rep:Number = {self.number}"
    # @property # this will allow you to :c.double instead of c.double()
    def half(self):
        self.number /= 2
        return self

    # @property # this will allow you to :c.double instead of c.double()
    def double(self):
        self.number *= 2
        return self


c = Calculator(100)
print(c)
# c.double()
# print(c)

