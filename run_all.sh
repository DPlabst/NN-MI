#!/bin/bash
# Run all examples.

# ----- Example 1: -----
python ex1a_nnmi.py -m 4-ASK -S 4 -d cpu  & 
python ex1b_nnmi.py -m 4-ASK -S 4 -d cpu  &

# ----- Example 2: -----
# a)
# Memoryless
python ex2a_nnmi.py -m 2-PAM -S 1 -d cpu  &
python ex2a_nnmi.py -m 2-ASK -S 1 -d cpu  &
python ex2a_nnmi.py -m 4-ASK -S 1 -d cpu  &
python ex2a_nnmi.py -m 8-ASK -S 1 -d cpu  &
python ex2a_nnmi.py -m 16-ASK -S 1 -d cpu  &
python ex2a_nnmi.py -m 32-ASK -S 1 -d cpu  &
python ex2a_nnmi.py -m 64-ASK -S 1 -d cpu  &

# Complex modulation Nyquist
python ex2a_nnmi.py -m 4-QAM -S 1 -d cpu  &
python ex2a_nnmi.py -m 16-QAM -S 1 -d cpu  &
python ex2a_nnmi.py -m 64-QAM -S 1 -d cpu  &

# b)
# CH6 Channel 
python ex2b_nnmi.py -m 2-ASK -S 4 -d cuda:0 & # Run on GPU

# Example 3: 
# Note: Must vary PA gain Pmax by hand!
# MF and symbol rate sampling
# Pmax is set to 7dB
python ex3a_nnmi.py -m 4-ASK -S 1 -d cpu  &

# Oversampling with Nsim=4  
# Pmax is set to 7dB
python ex3b_nnmi.py -m 4-ASK -S 2 -d cpu  &

wait 
