# imageproc

All code related to CS260: Image Processing course, *American University of Armenia*, Fall 2023 and in general to the field

Author: Gevorg Nersesian, gevorg_nersesian@edu.aua.am    

## Usage

In a terminal window of a directory with your image file, type
```bash

python3 main.py [inputFileName] --[mode] [parameter_OPTIONAL]

```
both *.png* and *.jpg* files are supported

In your directory you will find an inverse of the inputted image


## Modes

```inverse```: finds the color inverse

```boxblur```: applies box blur

```gaussianblur```: applies gaussian blur, takes in standard deviation as its parameter (1 by default)
