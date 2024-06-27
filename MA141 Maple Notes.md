# Maple Notes

This document provides a comprehensive overview of basic Maple commands and operations across various mathematical topics, including expressions, functions, equations, graphs, differentiation, numerical integration, differential equations, and vectors.

## Basics

```maple
## Basics

```maple
# Assigning Variables
A := 5;                # Simple assignment
assign(A=10);          # Using assign function
Digits := 30;          # Setting precision
evalf(Pi, 30);         # Evaluating Pi to 30 digits

# Lists and Sets
L := [1, 2, 3];           # Define a list (ordered)
S := {1, 2, 3};           # Define a set (unordered)

# Unassigning Variables
A := 'A';              # Unassign variable A

# Built-in Functions
exp(1);                # Exponential function
ln(2);                 # Natural logarithm
sin(Pi/2);             # Sine function
cos(Pi);               # Cosine function
tan(Pi/4);             # Tangent function
arcsin(1);             # Inverse sine
arccos(0);             # Inverse cosine
arctan(1);             # Inverse tangent
abs(-5);               # Absolute value
sqrt(16);              # Square root
surd(27, 3);           # Cube root
signum(-10);           # Sign function: returns -1 for negative, 0 for zero, 1 for positive
```
- Use `;` at the end of each command to execute it.
- The `:=` operator is used for assignment, not `=` which is used for equality checks.
- Some variable names are reserved: I (square root of -1), Pi, gamma (Euler's constant)

## Expressions, Functions, Equations and Algebra

```maple
# Expression Manipulation
subs(x=3, expr);              # Substitute x with 3 in expr
convert(x^(2/3), surd);       # Convert fractional power to surd

# Defining Functions
f := x -> x^2;          # Function f(x) = x^2
g := (x, y) -> x + y;   # Function g(x, y) = x + y

# Calling Functions
f(3);       # Call function f with argument 3
g(2, 4);    # Call function g with arguments 2 and 4

# Simplifying and Expanding Expressions
simplify(expr);               # Simplify the expression
expand((x + 1)^2);            # Expand the expression
factor(x^2 + 2*x + 1);        # Factor the expression

# Equation Manipulation
lhs(x^2 - 4 = 0);             # Left-hand side of the equation
rhs(x^2 - 4 = 0);             # Right-hand side of the equation

# Solving Equations
solve(x^2 - 4 = 0, x);        # Solve the equation for x

# Solving RootOf
allvalues(RootOf(x^3 - x + 1));        # Find all values of the RootOf expression

# Solving Sets of Equations
solve({x + y = 2, x - y = 0}, {x, y}); # Solve system of equations for x and y
```
- Use `->` to define anonymous functions (lambda expressions).
- Use `solve` to find solutions to equations. For systems of equations, use curly braces {} to enclose the set of equations and variables.
- Use `subs(x=#, expr)` to substitute a value into an expression or function.
- To convert a function to an expression, use `subs` with the function. To convert an expression to a function, use `unapply`.
- Use `solve` to solve equations and `allvalues` to find all values of a `RootOf` expression. For systems of equations, enclose them in curly braces `{}`.

## Graphs and Limits

```maple
# Basic Plotting
plot(x^2, x = -10..10);                                # Plot x^2 from -10 to 10
plot([sin(x), cos(x)], x = -Pi..Pi);                   # Plot sin(x) and cos(x) from -Pi to Pi

# Plotting an Odd Root of a Negative Number Using ^
plot((-x)^(1/3), x = -10..10);                         # Plot (-x)^(1/3) correctly

# Plotting Without Vertical Asymptote
plot(tan(x), x = -Pi..Pi, y = -10..10, discontinuity=true);   # Plot tan(x) without vertical asymptote

# Customizing Plots
plot(x^2, x = -10..10, color=red);                     # Plot with color
plot(x^2, x = -10..10, thickness=2);                   # Plot with thickness
plot([x^2, x^3], x = -10..10, color=[red, blue]);      # Plot multiple functions with different colors

# Calculating Limits
limit(x^2 / (x - 1), x = 1, left);        # Limit as x approaches 1 from the left
limit(x^2 / (x - 1), x = 1, right);       # Limit as x approaches 1 from the right


# Examples: Plotting an Odd Root of a Negative Number Using ^
expr := (x-2)^(2/3)
plot(expr, x = 0..10);          # Plotting incorrectly
newexpr := convert(expr, surd); # Plotting correctly
plot(newpxr, x= 0..10);

# Plotting without vertical asymptote
plot(tan(x), x = -Pi..Pi, y = -10..10, discontinuity=true);   # Plot tan(x) without vertical asymptote
```
- Use `plot` to create plots. For multiple functions, enclose them in square brackets `[]`.
- Use `discontinuity=true` to handle discontinuities and avoid vertical asymptotes.
- Use `limit(expr, x=#, left/right)` to calculate the limit of an expression as x approaches a value from the left or right.

## Differentiation and Implicit Plot

```maple
# Differentiation
# Differentiation of Expressions
diff(sin(x), x);                            # Differentiate sin(x) with respect to x
diff(x^3 + 2*x^2 + x, x);                   # First derivative
diff(x^3 + 2*x^2 + x, x$2);                 # Second derivative

# Differentiation of Functions
f := x -> x^3 + 2*x^2 + x;                  
D(f);                                       # First derivative of function
(D@D@D)(f);                                 # Third derivative of function
(D@@3)(f);                                  # Another way to denote third derivative



# Implicit Plotting
with(plots):
p1 := implicitplot(x^2 + y^2 = 1, x = -1..1, y = -1..1, grid=[50,50], color=red);       # Implicit plot with grid and color
p2 := plots[implicitplot](x^2 + y^2 = 1, x = -1..1, y = -1..1);                         # Another way to call implicitplot

display({p1, p2});                          # Display both plots together
```
- Use `diff` for differentiation. Use `implicitplot` for plotting implicit functions.

## Integrals
```maple
with(student):

# Riemann Sums
leftbox(f(x), x = a..b, n);              # Left Riemann sum boxes
middlebox(f(x), x = a..b, n);            # Middle Riemann sum boxes
rightbox(f(x), x = a..b, n);             # Right Riemann sum boxes

leftsum(f(x), x = a..b, n);              # Left Riemann sum
middlesum(f(x), x = a..b, n);            # Middle Riemann sum
rightsum(f(x), x = a..b, n);             # Right Riemann sum

# Tips
# For a decreasing function, leftsum > rightsum

# Increasing the Number of Boxes
lefts := n -> leftsum(f(x), x = a..b, n);    # Define a function for leftsum
limit(lefts(n), n = infinity);               # Compute the limit as n approaches infinity

# Riemann Integrals
Int(f(x), x = a..b);                        # Define the integral
value(%);                                   # Evaluate the integral

# Fundamental Theorem of Calculus
int(f(x), x);                               # Indefinite integral

# Examples
g := n -> rightsum(sqrt(x) + 2, x = 1..4, n);   # Define a function for right Riemann sums
Int(x^3 - 3*x^2 + 2*x, x = 0..1);               # Define the integral
rightsum(sqrt(x) + 2, x = 1..4, 60);            # Compute the right Riemann sum with 60 intervals
limit(g(n), n = infinity);                      # Compute the limit as n approaches infinity
```
- Last three examples are visualized as the following: ![maple1.png](images%2Fmaple1.png)
## Numerical Integration

```maple
with(student):
# Trapezoid Sum
trapezoid(f(x), x = a..b, n);             # Compute the trapezoid sum

# Simpson's Rule
simpson(f(x), x = a..b, n);               # Compute the integral using Simpson's Rule

# Error Analysis
# The error can be estimated using K/n^2, where K is estimated when n = 1
# Example to estimate K
n := 1;
trap_approx := trapezoid(f(x), x = a..b, n);
exact_value := int(f(x), x = a..b);
K_estimate := abs(trap_approx - exact_value) * n^2;

# Comparison of Area Approximation Methods
leftsum(f(x), x = a..b, n);               # Left Riemann sum
trapezoid(f(x), x = a..b, n);             # Trapezoid sum
simpson(f(x), x = a..b, n);               # Simpson's Rule
evalf(Int(x^2, x = 0..1));   # Numerically integrate x^2 from 0 to 1
evalf(Int(sin(x), x = 0..Pi)); # Numerically integrate sin(x) from 0 to Pi
```
- The trapezoid sum typically underestimates the area, while the middle sum overestimates it.
- Accuracy comparison (from least to most accurate): right/left point < trapezoid < simpson < midpoint
- Use `evalf(Int(...))` for numerical integration to get a floating-point result.
- Use `D(f) `for differentiating functions. For higher derivatives, use `D@D@...` or `D@@#`.

## Differential Equations and Euler’s Method

```maple
# Define the differential equation using diff
dep := diff(y(x), x);

# Define the differential expression
dexp := 2*x + y(x);
deq := diff(y(x), x) = dexp;
dep := diff(y(x), x) = dexp;

# Plot
with(DEtools):
DEplot(dep, y(x), x = -2..2, y = -2..2);    # Plot the direction field for the differential equation

# Euler’s Method
xpt := -1;
ypt := 1.4;
newpoint[0] := [xpt, ypt];
deltax := 0.2;

for a to 10 do
    slope := 2*xpt + ypt;
    ypt := ypt + deltax*slope;
    xpt := xpt + deltax;
    newpoint[a] := [xpt, ypt];
od;

# Collect results into a sequence
listofpoints := [seq(newpoint[i], i = 0..10)];
```
- Use `dsolve` to solve differential equations analytically.
- Euler’s method is a simple numerical approach to solve ODEs by iterating updates.
- Use `DEplot` from `DEtools` to plot direction fields of differential equations.

Example
```maple
f := x -> x^2;
df := D(f);

# Define the differential equation and expressions
deq := diff(y(x), x) = 2*x + y(x);
dexp := 2*x + y(x);
dep := diff(y(x), x) = dexp;

# Solve the differential equation exactly
dsolve(deq);

# Plotting the direction field
with(DEtools):
DEplot(dep, y(x), x = -2..2, y = -2..2);

# Euler's Method Implementation
xpt := -1;
ypt := 1.4;
newpoint[0] := [xpt, ypt];
deltax := 0.2;

for a to 10 do
    slope := 2*xpt + ypt;
    ypt := ypt + deltax*slope;
    xpt := xpt + deltax;
    newpoint[a] := [xpt, ypt];
od;

listofpoints := [seq(newpoint[i], i = 0..10)];
approx2 := plot(listofpoints, color=blue):
dirfield2 := DEplot(deq, y(x), x = -2..2, y = 0..4):
display([approx2, dirfield2]);
```

## Taylor Approximations to Functions

```maple
# Taylor Series Expansion
series(sin(x), x = 0, 5);    # Taylor series expansion of sin(x) around 0 up to 5th order
taylor(exp(x), x = 0, 3);    # Taylor series expansion of exp(x) around 0 up to 3rd order
```
- Use `series` or `taylor` to obtain the Taylor series expansion of a function.

## Vectors

```maple
# Define a Vector
A := [x, y, z];                        # Define vector A
subs({x = 1, y = 2, z = 3}, A);        # Substitute values into vector A
Vec := [2, 3, 4] - [3, 4, 5];           # Vector subtraction

# Linear Algebra Package - linalg
with(linalg):
norm(Vec, 2);                          # Euclidean norm of vector Vec 

# Operations
A := [1, 2, 3];
B := [4, 5, 6];
innerprod(A, B);                       # Inner product of vectors A and B
crossprod(A, B);                       # Cross product of vectors A and B
area := norm(crossprod(A, B), 2);      # Area using cross product
type(crossprod(A, B), list);           # Check type of cross product
convert(crossprod(A, B), list);        # Convert cross product to list if necessary
angle(A, B);                           # Angle between vectors A and B
```
- Use `Vector` to define vectors. 
- Use `dotprod` for the dot product and CrossProduct for the cross product.

```maple
# Example: Finding a Vector Orthogonal to a Plane
P := [1, 2, 3];
Q := [4, 5, 6];
R := [-2, 7, 11];

vec_pq := Q - P;                       # Vector PQ
vec_pr := R - P;                       # Vector PR
orthogonal_vector := convert(crossprod(vec_pq, vec_pr), list);  # Vector orthogonal to plane

# Display results
A, Vec, norm(Vec, 2), innerprod(A, B), crossprod(A, B), area, type(crossprod(A, B), list), angle(A, B), orthogonal_vector;
```

