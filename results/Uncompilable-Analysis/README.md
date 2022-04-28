# Analysis of 1883 Defects4J Noncompilable Patches from Top-30 Generated Results


* Please see the original data in **one-error-per-noncompilable-patch.txt**.
* The result is computed based on the **compilable-log.out** with **getError.py**.


The first column lists the compiling failure reason.

The second column shows the number of generated patched programs that failed for the corresponding reason.

|Compiling Error|No.|
|---|---|
|cannot find symbol |606|
|illegal start of expression |329|
|no suitable method/constructor found for ... |132|
|incompatible types |123|
|not a statement |102|
|';' expected |76|
|unreachable statement |63|
|case, default, or '}' expected |59|
|incomparable types |56|
|method X in class Y cannot be applied to given types |53|
|')' expected |45|
|missing return statement |44|
|cannot return a value from method whose result type is void |20|
|illegal start of type |15|
|bad operand types for binary operator |12|
|cannot assign a value to final variable |10|
|x has private access |9|
|variable x is already defined |8|
|inconvertible types |8|
|int/double/boolean cannot be dereferenced |5 |
|unclosed string literal |5|
|'catch' without 'try' |3|
|'else' without 'if'| 4|
|enum types may not be instantiated |1|
|orphaned default |1|


