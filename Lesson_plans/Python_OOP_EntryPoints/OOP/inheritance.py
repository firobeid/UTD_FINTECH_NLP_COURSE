class calculator():
    def __init__(self, var1, var2):
        self.var1 = var1
        self.var2 = var2

    def addition(self):
        return self.var1+self.var2

    def subtraction(self):
        return self.var1-self.var2

    def division(self):
        return self.var1/self.var2

    def multiplication(self):
        return self.var1*self.var2

class calculator_printer(calculator):
    def __init__(self, var1, var2):
        super().__init__(var1, var2)
    
    def print_all(self):
        print('Variable 1: ' + str(self.var1) + ', Variable2 : ' + str(self.var2))
        print('Addition = ' + str(self.addition()))
        print('subtraction = ' + str(self.subtraction()))
        print('division = ' + str(self.division()))
        print('multiplication = ' + str(self.multiplication()))


printer = calculator_printer(10,5)
printer.print_all()
