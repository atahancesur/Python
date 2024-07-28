#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class Parser:
    #This class implements a recursive descent parser for a simple arithmetic expression grammar.
    def __init__(self, file_name):
        # Open file
        self.file = open(file_name, 'r')
        # Initialize the variable that will hold the next token
        self.next_token = None
        # Sets a flag for error condition
        self.error = False
        # Variable to hold unconsumed input
        self.unconsumed_input = ""

    def lex(self): #The job of lex() is to read the next character from the file and place it innext_token each time it is called
        # Read input file at token level
        while True:
            c = self.file.read(1)
             # If we are at the end of the file, terminate using the '$' symbol
            if not c:
                self.next_token = '$'
                return
            
            # Skip blanks
            if c.isspace():
                continue
                
            # Set the next token
            self.next_token = c
            return

    def unconsumed_inpute(self):
         # Accumulate unconsumed input
        self.unconsumed_input += self.file.read()
        return

    def G(self):
        # Start the parsing process
        self.lex()
        print("G -> E")
        self.E()
        
        # Report success if all input has been processed and there are no errors
        if self.next_token == '$' and not self.error:
            print("Success")
        else:
             # In case of error, show unconsumed input
            self.file.seek(0)
            a = self.file.read()
            print(f"Error: Input not consumed= {a}")

    def E(self):
        # End the process if there is an error
        if self.error:
            return

        print("E -> T R")
         # Apply E -> T R rules
        self.T()
        self.R()

    def R(self):
        # End the process if there is an error
        if self.error:
            return

        # If it starts with one of the '+' or '-' symbols
        if self.next_token == '+':
            print("R -> + T R")
            self.lex() # Get the next token
            self.T() # Process the next T expression
            self.R() # Process the remaining R expressions
        elif self.next_token == '-':
            print("R -> - T R")
            self.lex()
            self.T()
            self.R()
        else:
            print("R -> ε")

    def T(self):
        # End the process if there is an error
        if self.error:
            return

        print("T -> F S")
         # Apply the rules T -> F S
        self.F()
        self.S()

    def S(self):
        # End the process if there is an error
        if self.error:
            return
        
        # If it starts with one of the symbols '*' or '/'
        if self.next_token == '*':
            print("S -> * F S")
            self.lex()
            self.F()
            self.S()
        elif self.next_token == '/':
            print("S -> / F S")
            self.lex()
            self.F()
            self.S()
        else:
            print("S -> ε")

    def F(self):
        # End the process if there is an error
        if self.error:
            return
        
        # If it starts with '('
        if self.next_token == '(':
            print("F -> ( E )")
            self.lex()# Get the next token
            self.E()# Process nested expression

            # Bracket closing control
            if self.next_token == ')':
                self.lex()# Get the next token
            else:
                # Flag error condition if parenthesis closing symbol is missing
                self.error = True
                print(f"Error: Unexpected icon = {self.next_token}")
                self.unconsumed_inpute()# Receive unconsumed input
                print(f"Input not consumed = {self.unconsumed_input}")
                return
            
        # If it starts with a number
        elif self.next_token.isdigit():
            print(f"F -> N")
            self.N()# Process number
        else:
            # Terminate on error and unconsumed 
            self.error = True
            print(f"Error: Unexpected icon = {self.next_token}")
            self.unconsumed_inpute()
            print(f"Input not consumed = {self.unconsumed_input}")

    def N(self):
        # End the process if there is an error
        if self.error:
            return
        
        # If it is a number, print it to the screen and get the next token
        if self.next_token.isdigit():
            print(f"N -> {self.next_token}")
            self.lex()# Get the next token
        else:
             # End operation on error and show unconsumed input
            self.error = True
            print(f"Error: Unexpected icon = {self.next_token}")
            self.unconsumed_inpute()# Receive unconsumed input
            print(f"Input not consumed = {self.unconsumed_input}")

# Get file name from user
file_name = input("Please enter the file name: ")

# Create the parser object and start the process
parser  = Parser(file_name)
parser.G()


# In[ ]:




