# Instruction for code implementation

## Environment
+ python 3.8
+ pymoo 0.6.0.1
+ speechbrain 0.5.13
+ torch 1.13.0
+ soundfile 0.11.0
+ pystoi 0.3.3
+ numpy 1.22.4
+ librosa 0.8.1
+ tqdm 4.64.1

## Datasets
We train and evaluate MicPro with 4 datasets: LibriSpeech, AISHELL, VCTK, and VoxCeleb1. Download these datasets and make sure the file paths follow the folder tree below:
### Folder tree
```
dataset
├── librispeech
│   └── train-clean-100
│       ├── 103
│           ├── 1240
│               ├── 103-1240-0057.wav
|               ├── ...
|           ├── ...
|       ├── ...
|
├── VCTK-Corpus
│   └── wav48
│       ├── p225
│           ├── p225_001.wav
|           └── ...
│       ├── ...
|
├── voxceleb1
|   └── vox1_test_wav
|       ├── id10270
|           ├── 5r0dWxy17C8
│               ├── 00001.wav
|               ├── ...
|           ├── ...
|       ├── ...
|               
└── data_aishell
    └── wav
        ├── test
            ├── S0764
                ├── BAC009S0764W0495.wav
                ├── ...
            ├── ...
                 
```
For each dataset, we make a copy of 8k sample rate version. You can use `resample.py` to resample them and create new folders with a `-8k` suffix.
```bash
python resample.py
```

## Modify the G.729 codec
G.729 is developed by ITU-T based on CS-ACELP (conjugate-structure algebraic-code-excited linear prediction). The source code and technical documents can be found at: [G.729](https://www.itu.int/rec/T-REC-G.729) . The basic G.729 code is written in the fixed-point ANSI C. Here we modify the G.729 code for MicPro implementation. 

### 1. Add a source file `lsftrans.c` to the code directory `g729/c_code`
We define the LSF transformation function in lsftrans.c:
```C
#include <stdio.h>
#include <math.h>
#include "typedef.h"
#include "ld8k.h"

void lsp_trans(
    Word16 lsp[]   /* (i/o) Q15 : line spectral frequencies  */
)
{
    int i,j;
    double lsf[10];

    // covert lsfs to normalized digital frequencies
    for(i=0; i<M; i++)
    {   
        lsf[i] = lsp[i]/32768.0;
    }

    // Formants transformation fucntion
    double xi1 = 1;
    for(i=0; i<M; i++)
    {   
        lsf[i] = lsf[i] + lsf[i]*(xi1-1)*(1-lsf[i]);
    }

    // Formants separation function
    double xi2 = 1;
    for(i=0; i<M; i++)
    {
        lsf[i] = lsf[i] + (xi2-1)*sin(2*3.1415926*lsf[i])/10.0;
    }

    // Bandwidth adjustment function
    double xi3 = 1;
    double delta[11] = {0};
    double delta_new[11] = {0};
    double lsf_new[10] = {0};
    for(i=1;i<10;i++){
        delta[i] = lsf[i] - lsf[i-1];
    }
    delta[0] = lsf[0];
    delta[10] =  1 - lsf[9];
    for(i=0;i<11;i++){
        delta_new[i] = delta[i] + (xi3-1)*(1/11.0-delta[i]);
    }
    for(i=0;i<10;i++){
        for(j=0;j<i+1;j++){
            lsf_new[i] += delta_new[j];
        }
    }


    for(i=0; i<M; i++)
    {   
        lsf[i] = lsf_new[i];
        lsp[i] = lsf[i]*32768;
    }
```

### 2. Modify `qua_lsp.c`
This file defines the functions used to quantify LSFs. Add our transformation function between `Lsp_lsf2(lsp, lsf, M)`; and `Lsp_qua_cs(lsf, lsf_q, ana )`;, which are defined in `Qua_lsp()`:
```C
    Lsp_lsf2(lsp, lsf, M);
    lsp_trans(lsf);  // add transformation function here
    Lsp_qua_cs(lsf, lsf_q, ana );
```

### 3. Modify `ld8k.h`
This file defines the funciton prototypes and constants used in G.729.
First add the definition of the constant SYNC in the begining of the file. The definition of SYNC means to force the input and output to be time-aligned.
```C
#define SYNC 1
```
Then add the prototype of the transformation function:
```C
void lsp_trans(
    Word16 lsp[]      /* (i/o) Q15 : line spectral frequencies(range: -1<=val<1)*/
);
```

### 4. Modify `coder.mak`
This file configures the compiling information.
Uncomment the code to reflect the particular system. For example, we use GCC compiler in a Linux system, then we uncomment this option:
```C
# Options for GCC C compiler
CC= gcc
CFLAGS = -O2 -Wall
```
Edit the objects needed for the encoder. Add `lsftrans.o` at the end:
```c
OBJECTS= \
    acelp_co.o\
    basic_op.o\
    bits.o\
    cod_ld8k.o\
    coder.o\
    dspfunc.o\
    filter.o\
    gainpred.o\
    lpc.o\
    lpcfunc.o\
    lspgetq.o\
    oper_32b.o\
    p_parity.o\
    pitch.o\
    pre_proc.o\
    pred_lt3.o\
    pwf.o\
    qua_gain.o\
    qua_lsp.o\
    tab_ld8k.o\
    util.o\
    lsftrans.o

coder :	$(OBJECTS)
    $(CC) -g -o coder $(OBJECTS) -lm
```

Note that the extra option `-lm` is needed since we additionally include `math.h` in `lsftrans.c`.
Finally add the dependency for `lsftrans.o`:
```C
lsptrans.o: lsptrans.c typedef.h ld8k.h
    $(CC) $(CFLAGS) -c  lsptrans.c -lm
```
### 5. Modify `decoder.mak`
Also uncomment the options according to our system:
```C
# Options for GCC C compiler
CC= gcc
CFLAGS = -O2 -Wall
```

### 6. An example for encoding and decoding an audio file
Note that the G.729 codec only support 8k sample rate. The input and output audio should both be 8k. We can use ffmpeg to convert the output to 16k if needed.

Create a bash script:
```bash
orig_path=''  # add your original audio path here
wave_path=''  # add your 8k audio save path here
wave_path_16k=''  # add your 16k audio save path here

cd g729/c_code/
rm *.o
make -f coder.mak
make -f decoder.mak

g729/c_code_float/coder "$orig_path".wav "$wave_path".bin
g729/c_code_float/decoder "$wave_path".bin "$wave_path".pcm
ffmpeg -y -ar 8000 -ac 1 -f s16le -i "$wave_path".pcm -ar 16000 -ac 1 -f s16le "$wave_path_16k".pcm
ffmpeg -f s16le -v 8 -y -ar 16000 -ac 1 -i "$wave_path_16k".pcm "$wave_path_16k".wav
```

**Note that we have provided the modified G.729 source code in our [file](code\g729).**


## Run NSGA algorithm
Make sure you have already prepared the datasets and their 8k versions and check the directory of files used in our code. Then run the NSGA algorithm and the results can be found at the save file defined in our code.
```bash
python moga.py
```
