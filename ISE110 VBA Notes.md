
# ISE110 VBA Notes
## Table of Contents
1. [Setting](#setting)
2. [Range](#range)
3. [Control Flow](#control-flow)
4. [Sheets](#sheets)
5. [Workbook](#workbook)
6. [Text File](#text-file)
7. [Array](#array)
8. [Function](#function)
9. [Form](#form)
10. [Others](#others)
11. [Additional Examples](#additional-examples)

## Setting

| Setting                          | Steps                              |
|----------------------------------|------------------------------------|
| View the name collection         | Formula > Manage Names             |
| Use relative reference <ul><li>Relative reference uses +/- to indicate the cell change  </li></ul>          |Developer > User Relative Reference |


## Range

| Range Command                   | Description/Code                                                                 |
|---------------------------------|----------------------------------------------------------------------------------|
| Select the whole table          | `Ctrl + Shift + 8`<br>`Ctrl + Shift + Down + Right`                               |
| Select the column               | `Ctrl + Shift + Down` (和直接选择不一样)                                           |
| Select cell/s                   | `Range("B6(:C3)").Select`<br>`Range(Ragne("B6") Range("C3"))`                     |
| Select rows/columns             | `Range("sales").Rows/Columns(3/"C").Select`<br>`Range("A1" Range("A1").End(xlDown)).Select` |
| Clear the format                | `Range("sales").ClearFormats`                                                    |
| Select a AE11 Cell              | `Cells(1131) = Range("AE11")`<br>`Range("AE10").Offset(1, 0)`                     |
| Get the address                 | `Range().Cells().Address`                                                        |
| Cells() function                | `Range("A2:B11").Cells(21)` // A3<br>`Range("A2:A11").Cells(2)` // A3             |
| Find the cell with value        | `Range("A1:D1").Find("Patient")`                                                 |
| Count rows/columns              | `Range().Rows.Count`                                                             |
| Set the color                   | `Range().Font.Color = vbMagenta`                                                 |

## Control flow

- **If Else**
```vba
If ... Then
    ...
[Elseif ... Then]
    ...
[Else]
    ...
End If
```
- **Select**
```vba
Select Case ...
    Case 3
        ...
    Case <= 3
        ...
    Case 3 to 4
        ...
    Case Else
        ...
End Select
```
- **For Loop**
```vba
For i = 1 To 10
    ...
    Exit For
Next i
```
- **For Each Loop**
```vba
For Each cell In Selection
    ...
Next cell
```
- **Do Until Loop**
```vba
Do
    ...
Loop Until Cells(Row 1).Value = ""
```
- **Do While Loop**
```vba
Do While ...
    Row = Row + 1
Loop
```
- **Interrupt Infinite Loop** `Ctrl + Break`

## Sheets

| Sheets Command                 | Description/Code                                            |
|--------------------------------|-------------------------------------------------------------|
| Copy a sheet                   | `Sheets(name).Copy Before:=Sheets(number)`                  |
| Move a sheet                   | `Sheets(name).Move After:=Sheets(number)`                   |
| Add a sheet to the end         | `Sheets.Add(After:=Worksheets(Worksheets.Count))`           |
| Delete sheet by name           | `Sheets(name).Delete`                                       |
| Copy and insert rows in a sheet| `Sheets(name).Rows(cell.row).EntireRow.Copy Range("A1").Offset(row, 0).Insert` |

## Workbook

| Workbook Command               | Description/Code                                            |
|--------------------------------|-------------------------------------------------------------|
| Open a workbook                | `String thisPath = ActiveWorkbook.Path Workbooks.Open filename:=thisPath & "\" & "name.xlsx"`<br>`function (Workbook.Open Filename := ".xlsx") is better than sub (Workbook.Open(Filename:=".xslx"))` |
| Save the copy as               | `ActiveWorkbook.SaveCopyAs thisPath & "\" & "name.xlsm"`    |
| Close and save changes         | `ActiveWorkbook.Close savechanges:=True`                    |
| Close the workbook             | `Workbooks(day & ".xlsx").Close`                            |
| Get the file name in the dir   | `Dir(ActiveWorkbook.Path & "\")`                            |
| Get the next file name         | `filename = Dir()`                                          |

## Text File

| Text File Command              | Description/Code                                            |
|--------------------------------|-------------------------------------------------------------|
| Open a text file               | `String thisPath = ActiveWorkbook.Path Workbooks.Open filename:=thisPath & "\" & "name.xlsx"`<br>`function (Workbook.Open Filename := ".xlsx") is better than sub (Workbook.Open(Filename:=".xslx"))` |
| Save the copy as               | `ActiveWorkbook.SaveCopyAs thisPath & "\" & "name.xlsm"`    |
| Close and save changes         | `ActiveWorkbook.Close savechanges:=True`                    |

## Array

| Array Command                  | Description/Code                                            |
|--------------------------------|-------------------------------------------------------------|
| Declare an array               | `Dim employee(100) as String` // idx = 0...99<br>`Dim employee(1 To 100) as String` // idx = 1...100 |
| Re-declare an array            | `ReDim employee(50)` // lost data<br>`ReDim Preserve employee(20)` // keep data only resize |

## Function

| Function Scope         | Description/Code                                              |
|------------------------|---------------------------------------------------------------|
| `public sub_name()`    | can be called in all modules                                  |
| `sub sub_name()`       | can be called in this module only                             |
| `private/dim var_name` | only used in current scope(procedure/module)                  |
| `public var_name`      | accessible by subroutines in outside modules                  |
| `call procedure1(arg1)`  | `procedure1` is called                                        |
| `function1(arg1)`        | `function1` is used(function can return values)               |
| `procedure1(ByRef arg1)` | Default. Passing the address of arg1                          |
| `procedure1(ByVal arg1)` | Passing the copy of arg1. The original value cannot be changed |

## Form

- **Multi-column Listbox**
    ```vba
    ReDim students(length, 3) As String
    For i = 1 To length
        students(i, 0) = Range("A1").Offset(i, 0)
        students(i, 1) = Range("B1").Offset(i, 0)
        students(i, 2) = Range("C1").Offset(i, 0)
    Next i
    Me.listBox.List = students
    ```

- **Populate the List**
    ```vba
    ReDim employee(50) ' lost data
    ReDim Preserve employee(20) ' keep data only resize
    ```
  
## Others

| Others Command                 | Description/Code                                            |
|--------------------------------|-------------------------------------------------------------|
| First/last name                | `first_name = Left(full_name, InStr(full_name, " ") - 1)`<br>`last_name = Right(full_name, Len(full_name) - InStr(full_name, " "))` |
| No alert                       | `Application.DisplayAlerts = False`                         |
| Average/min/max                | `Application.WorksheetFunction.Average(Range(start), Range(end))` |
| String format                  | `string = Format(avg, "$##.00")`                            |
| Random number                  | `WorksheetFunction.RandBetween(Arg1:=10, Arg2:=15)` // 10-15<br>`Int((1000 + 1 - 500) * Rnd + 500)` // integer between 500 to 1000 |
| Weekday function               | `Weekday(date, [return_value])` returns number              |
| Multiply matrix                | `mmult(matrix1, matrix2)` with `Ctrl + Shift + Enter`       |
| Line break                     | `& char(10) &`                                              |



## Additional Examples

```vba
'TSP(Lab 16

'ISE Lab
form_name.show
double click to add the control command
in initialize() sub:
    'Empty CityListBox
    CityListBox.Clear
    
    'Fill CityListBox
    With CityListBox
        .AddItem "San Francisco"
        .AddItem "Oakland"
        .AddItem "Richmond"
    End With

Me.CityListBox.RowSource = “customers” ' customers is the range name
answer = MsgBox("Do you really want to quit?" vbYesNo, “Terminate form”)
if answer = vbYes then Unload Me

Dim cell as range
for each cell in range(“customers”)
    call me.lstcustomer.addItem(cell.value)
next cell
```
```vba
' Read and Write Files
Open MyFile for Input as #2
with Range
do until EOF(2)
    Line Input #2 dataline.offset(i, 0) = dataline
    i = i + 1
loop
end with
Close #1

n = len(myString) - len(Replace(myString, “$”, “”))  ' n is the number of the dollar signs
```
solver:
- use the function(absolute reference) to calculate the goal value (to maximize)
- use the solver to select the goal value and constraints

on error Goto 0
on error Resume Next
on error Goto <label_code>
