# Julia
Julia is a high-level, high-performance dynamic programming language for technical computing, with syntax that is familiar to users of other technical computing environments. It provides a sophisticated compiler, distributed parallel execution, numerical accuracy, and an extensive mathematical function library. 

Many Julia syntax is similar to python, so we will focus on the different ones in this doc. 

[Link to Python ACO
Youtube](https://www.youtube.com/watch?v=EJKdmEbGre8)

### Variable

#### Naming

x = 1 + 2

- pi can represent π and can be assigned to other values

- name/value of the variable can be other languages

- name of the variable should be lowercase (separated by \_)

- name of types/modules: MyTypeOrModule

- name of functions/macros: myfunction

#### String
```julia
str = "Hello, World!"
str[0]  ERROR
str[1]  = "H"
str[end]  = "!"
str[end÷2]  = ","
str[end+1]  ERROR
str[4:9]  = "lo, Wo"
str[6:6] = str[6]  = ","
SubString(str, 2, 3)  = "el"

#Concatenation
string(str, "! ", whom, ".\n")  = Hello, World!! Wow.
str *   "!"   *   whom   *   "!\n"

#Interpolation - evaluate the value
"$str $whom.\n"  = "Hello, World! Wow."
"1 + 2 = $(1 + 2)"  = "1 + 2 = 3"

#Find first/last index
findfirst(isequal('o'), "xylophone")  = 4
julia> findlast(isequal('o'), "xylophone")  = 7
#Find next index
findnext(isequal('o'), "xylophone", 1)  = 4
findnext(isequal('o'), "xylophone", 5)  = 7 #find next index of "o" from 5
findprev(isequal('o'), "xylophone", 5)  = 4 

#Contains?
occursin("world", "Hello, world.")   = true

s2 = "1 + 1 = $(1 + 1)"  # Expressions in strings
```
- replace, nextind, match, captures, offsets,

#### Function

> f(x,y) = x + y

OR
```julia 
function f(x,y)
    x + y
end
```
- Default returned value = last expression evaluated

Tuple: (2, "how", 0.4)
```julia 
function foo(a,b)
    a+b, a*b
end
x, y = foo(2,3)  = (5, 6) # x = 5, y = 6
```

Array: [2, 3]

### Number

<table>
<colgroup>
<col style="width: 15%" />
<col style="width: 15%" />
<col style="width: 23%" />
<col style="width: 23%" />
<col style="width: 21%" />
</colgroup>
<tbody>
<tr class="odd">
<td>Type</td>
<td>Signed?</td>
<td>Number of bits</td>
<td>Smallest value</td>
<td>Largest value</td>
</tr>
<tr class="even">
<td>Int8</td>
<td>✓</td>
<td>8</td>
<td>-2^7</td>
<td>2^7 - 1</td>
</tr>
<tr class="odd">
<td>UInt8</td>
<td></td>
<td>8</td>
<td>0</td>
<td>2^8 - 1</td>
</tr>
<tr class="even">
<td>Int16</td>
<td>✓</td>
<td>16</td>
<td>-2^15</td>
<td>2^15 - 1</td>
</tr>
<tr class="odd">
<td>UInt16</td>
<td></td>
<td>16</td>
<td>0</td>
<td>2^16 - 1</td>
</tr>
<tr class="even">
<td>Int32</td>
<td>✓</td>
<td>32</td>
<td>-2^31</td>
<td>2^31 - 1</td>
</tr>
<tr class="odd">
<td>UInt32</td>
<td></td>
<td>32</td>
<td>0</td>
<td>2^32 - 1</td>
</tr>
<tr class="even">
<td>Int64</td>
<td>✓</td>
<td>64</td>
<td>-2^63</td>
<td>2^63 - 1</td>
</tr>
<tr class="odd">
<td>UInt64</td>
<td></td>
<td>64</td>
<td>0</td>
<td>2^64 - 1</td>
</tr>
<tr class="even">
<td>Int128</td>
<td>✓</td>
<td>128</td>
<td>-2^127</td>
<td>2^127 - 1</td>
</tr>
<tr class="odd">
<td>UInt128</td>
<td></td>
<td>128</td>
<td>0</td>
<td>2^128 - 1</td>
</tr>
<tr class="even">
<td>Bool</td>
<td>N/A</td>
<td>8</td>
<td>false (0)</td>
<td>true (1)</td>
</tr>
</tbody>
</table>

```julia 
typemax(Int64)   = 9223372036854775807
2.5e-4  =0.00025
2.5f-4  = 0.00025f0(float indicated)
Float32(-1.5)  = -1.5f0
```

#### Floating number

<table>
<colgroup>
<col style="width: 27%" />
<col style="width: 30%" />
<col style="width: 42%" />
</colgroup>
<tbody>
<tr class="odd">
<td>Type</td>
<td>Precision</td>
<td>Number of bits</td>
</tr>
<tr class="even">
<td><a href="https://docs.julialang.org/en/v1/base/numbers/#Core.Float16"><u>Float16</u></a></td>
<td><a href="https://en.wikipedia.org/wiki/Half-precision_floating-point_format"><u>half</u></a></td>
<td>16</td>
</tr>
<tr class="odd">
<td><a href="https://docs.julialang.org/en/v1/base/numbers/#Core.Float32"><u>Float32</u></a></td>
<td><a href="https://en.wikipedia.org/wiki/Single_precision_floating-point_format"><u>single</u></a></td>
<td>32</td>
</tr>
<tr class="even">
<td><a href="https://docs.julialang.org/en/v1/base/numbers/#Core.Float64"><u>Float64</u></a></td>
<td><a href="https://en.wikipedia.org/wiki/Double_precision_floating-point_format"><u>double</u></a></td>
<td>64</td>
</tr>
</tbody>
</table>

#### Complex & Rational Number

```julia 
# Complex Number - functions
real(1 + 2im)  = 1
imag(1 + 2im)  = 2
conj(1 + 2im)  = 1 - 2im
abs(1 + 2im)  = 2.23606797749979
abs2(1 + 2im)  = 5
angle(1 + 2im)  = 1.1071487177940904
# operators
(1 + 2im)*(2 - 3im)  = 8 + 1im
2im^2  = -2 + 0im

#Rational Number
2//3  = 2/3
```

### Math Expression

<table>
<colgroup>
<col style="width: 21%" />
<col style="width: 29%" />
<col style="width: 48%" />
</colgroup>
<tbody>
<tr class="odd">
<td>Expression</td>
<td>Name</td>
<td>Description</td>
</tr>
<tr class="even">
<td>+-x</td>
<td>unary plus/minus</td>
<td>the identity operation</td>
</tr>
<tr class="odd">
<td>-x</td>
<td>unary minus</td>
<td>maps values to their additive inverses</td>
</tr>
<tr class="even">
<td>x +-*/ y</td>
<td>binary + - * /</td>
<td></td>
</tr>
<tr class="odd">
<td>x ÷ y</td>
<td>integer divide</td>
<td>x / y, truncated to an integer</td>
</tr>
<tr class="even">
<td>x \ y</td>
<td>inverse divide</td>
<td>equivalent to y / x</td>
</tr>
<tr class="odd">
<td>x ^ y</td>
<td>power</td>
<td>raises x to the yth power</td>
</tr>
<tr class="even">
<td>x % y</td>
<td>remainder</td>
<td>equivalent to rem(x,y)</td>
</tr>
<tr class="odd">
<td>!x</td>
<td>negation</td>
<td>true &lt;=&gt; false</td>
</tr>
</tbody>
</table>

### Bitwise Operators

<table>
<colgroup>
<col style="width: 39%" />
<col style="width: 60%" />
</colgroup>
<tbody>
<tr class="odd">
<td>Expression</td>
<td>Name</td>
</tr>
<tr class="even">
<td>~x</td>
<td>bitwise not</td>
</tr>
<tr class="odd">
<td>x &amp; y</td>
<td>bitwise and</td>
</tr>
<tr class="even">
<td>x | y</td>
<td>bitwise or</td>
</tr>
<tr class="odd">
<td>x ⊻ y</td>
<td>bitwise xor (exclusive or)</td>
</tr>
<tr class="even">
<td>x &gt;&gt;&gt; y</td>
<td><a href="https://en.wikipedia.org/wiki/Logical_shift"><u>logical
shift</u></a> right</td>
</tr>
<tr class="odd">
<td>x &gt;&gt; y</td>
<td><a href="https://en.wikipedia.org/wiki/Arithmetic_shift"><u>arithmetic
shift</u></a> right</td>
</tr>
<tr class="even">
<td>x &lt;&lt; y</td>
<td>logical/arithmetic shift left</td>
</tr>
</tbody>
</table>

```julia 
~123  = -124
123 & 234  = 106
123 | 234  = 251
123 ⊻ 234  = 145
xor(123, 234)  = 145
~UInt32(123)  = 0xffffff84
~UInt8(123)  = 0x84
```

- Math Functions: `rounding(ceil, floor…)`, `division(div, rem, mod,
  divrem…)`, `sign` & `absolute(abs, sign, copysign…)`, `power` & `root`, `sqrt,
  cbrt`, `exp`, `log`...
