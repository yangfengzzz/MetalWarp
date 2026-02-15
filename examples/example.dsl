# Numeric example - compatible with both interpreter and C codegen

# Variables and arithmetic
x = 10
y = 3
result = x * y + 5
print(result)

# Integer division and modulo
print(x // y)
print(x % y)

# Conditional
if result > 30:
    print(1)
else:
    print(0)

# While loop
n = result
while n > 1:
    n = n // 2
print(n)

# For loop with range
total = 0
for i in range(1, 11):
    total = total + i
print(total)

# Countdown (negative step)
for i in range(5, 0, -1):
    print(i)

# Function definition and call
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(10))

# Power operator
base = 2
print(base ** 10)
