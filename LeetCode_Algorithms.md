# LeetCode Algorithms

This file contains implementations and explanations of commonly used algorithms to solve LeetCode problems. Click on each algorithm below to view its foldable section.

| Algorithm                   | Description                              |
|-----------------------------|------------------------------------------|
| [Knuth-Morris-Pratt (KMP)](#knuth-morris-pratt-kmp-algorithm) | String searching algorithm using LPS array to avoid redundant comparisons. |


---
<details>
  <summary>Knuth-Morris-Pratt (KMP) Algorithm</summary>

### Knuth-Morris-Pratt (KMP) Algorithm

The **Knuth-Morris-Pratt (KMP)** algorithm is used for **string pattern matching**. It searches for occurrences of a word (or pattern) within a main text efficiently by preprocessing the pattern to avoid unnecessary comparisons.

#### Steps:
1. **Preprocess the Pattern**: Compute the Longest Prefix Suffix (LPS) array.
2. **Search the Pattern** in the text using the LPS array to skip unnecessary comparisons.

#### 1. Preprocessing: LPS Array
The **LPS array** (also known as the prefix table) is used to skip unnecessary character comparisons when a mismatch occurs. It represents the length of the longest proper prefix of the pattern that is also a suffix.

For example, for pattern `P = "ABABCABAB"`, the LPS array is:

| Index | Pattern Prefix | LPS Value |
|-------|----------------|-----------|
| 0     | A              | 0         |
| 1     | AB             | 0         |
| 2     | ABA            | 1         |
| 3     | ABAB           | 2         |
| 4     | ABABC          | 0         |
| 5     | ABABCA         | 1         |
| 6     | ABABCAB        | 2         |
| 7     | ABABCABA       | 3         |
| 8     | ABABCABAB      | 4         |

#### 2. Search Algorithm: Pseudocode
```python
def computeLPSArray(pattern):
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1
    
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps

def KMPSearch(text, pattern):
    n = len(text)
    m = len(pattern)
    
    lps = computeLPSArray(pattern)
    
    i = 0  # index for text
    j = 0  # index for pattern
    
    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1
        
        if j == m:
            print(f"Pattern found at index {i - j}")
            j = lps[j - 1]
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
```
#### Time and Space Complexity:
- Time Complexity: O(n + m)
- Space Complexity: O(m)

</details> 
