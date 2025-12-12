# Sample Naming Guide

Each sample is named according to the following convention:

**`patN_TISSUE(L or R)_LETTER`**

where:

- **N** → the encrypted patient number  
- **TISSUE** → a three-letter abbreviation of the sampled tissue  
- **L or R** → optional information, included when available, indicating the side of the sample (Left or Right)  
- **LETTER** → a letter used when multiple samples from the same tissue are collected for the same patient  

### Example
- `pat0_AdnL_A`  
- `pat0_AdnL_N`  

These indicate that for patient 0 we have two slides from the **left side of the Adnexa**.

### Important Note
The patient number (**N**) is unique **only within the same cohort type**.  
For example:  
- `pat0` in the **PDS cohort** is **not** the same patient as `pat0` in the **NACT cohort**.