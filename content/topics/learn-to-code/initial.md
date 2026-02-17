# Getting Started with Programming

Welcome to learning to code! This interactive tutorial will guide you through the basics of programming using Python-like syntax. Each section includes examples you can run and exercises to practice.

## Hello World!

Let's start with the traditional first program. The `print` function outputs text to the console.

> Try running the code below by clicking the "Run" button.

[[code-editor print("Hello world!")|Hello world!]]

## Variables

Variables let us remember values. Variable assignment is done with the `=` sign.

> Assign numbers to two variables and print their sum.

[[code-editor | |x = 5
y = 10
print(x + y)]]

## Operators

Python can do math with standard operators:
- `+` addition
- `-` subtraction  
- `*` multiplication
- `/` division
- `**` exponentiation
- `//` integer division
- `%` modulo

> Define two numbers in variables, and use each operator on them.

[[code-editor | |a = 10
b = 3
print("Addition:", a + b)
print("Subtraction:", a - b)
print("Multiplication:", a * b)
print("Division:", a / b)
print("Exponentiation:", a ** b)
print("Integer division:", a // b)
print("Modulo:", a % b)]]

## Conditionals

Conditional statements allow us to make decisions in code using `if`, `elif`, and `else`.

> Write a program that checks if a number is positive, negative, or zero.

[[code-editor | |num = 5
if num > 0:
    print("Positive")
elif num < 0:
    print("Negative")
else:
    print("Zero")]]

## Loops

Python can do things many times with loops! The `for` loop iterates over a sequence.

> Assign your name to a variable and loop over the letters, printing each one.

[[code-editor | |name = "Python"
for letter in name:
    print(letter)]]

## Functions

Functions are reusable blocks of code. They are defined with `def` and called by name.

> Write a function that takes a number and prints its square.

[[code-editor | |def square(x):
    print(x * x)

square(5)]]

## Exercise: Simple Calculator

> Create a function that takes two numbers and an operator (+, -, *, /) and returns the result.

[[code-editor |15|def calculate(a, b, op):
    if op == "+":
        return a + b
    elif op == "-":
        return a - b
    elif op == "*":
        return a * b
    elif op == "/":
        return a / b
    else:
        return "Invalid operator"

print(calculate(10, 5, "+"))]]

## While Loops

While loops continue as long as a condition is true.

> Write a program that prints numbers from 1 to 5 using a while loop.

[[code-editor | |i = 1
while i <= 5:
    print(i)
    i = i + 1]]

## Lists

Lists are ordered collections of items.

> Create a list of numbers and print each one.

[[code-editor | |numbers = [1, 2, 3, 4, 5]
for num in numbers:
    print(num)]]

## Exercise: Sum of List

> Write a function that takes a list of numbers and returns their sum.

[[code-editor |15|def sum_list(nums):
    total = 0
    for num in nums:
        total = total + num
    return total

print(sum_list([1, 2, 3, 4, 5]))]]

## Strings

Strings are sequences of characters.

> Write a program that counts the vowels in a string.

[[code-editor | |def count_vowels(text):
    vowels = "aeiouAEIOU"
    count = 0
    for char in text:
        if char in vowels:
            count = count + 1
    return count

print(count_vowels("Hello World"))]]

## Final Exercise: FizzBuzz

> Write a program that prints numbers from 1 to 20, but for multiples of 3 print "Fizz", for multiples of 5 print "Buzz", and for multiples of both print "FizzBuzz".

[[code-editor | |for i in range(1, 21):
    if i % 3 == 0 and i % 5 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)]]

