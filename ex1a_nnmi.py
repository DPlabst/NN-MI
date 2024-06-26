#!/usr/bin/env python
# coding: utf-8

# **************************************************
# * NN-MI: Neural Network Achievable Information Rate Computation for Channels with Memory
# * Daniel Plabst
# * Institute for Communications Engineering (LNT)
# * Technical University of Munich, Germany
# * http://ice.cit.tum.de/
# * Public version: v1.1 2024-03-07
# **************************************************

vstr = "v1.1"  # Version

import time
import numpy as np
import torch
from torch.multiprocessing import Pool
import scipy.signal

# * ------------ Package nnmi  ----------------
from nnmi import hlp  # Helper functions
from nnmi import comsys  # Communication system functions and SIC receiver

# * ------------   PyTorch settings ----------------
torch.set_num_threads(2)  # Two seems enough for small NNs

# * ---------  Plotting/Saving flag -----------
f_showplot = 1
f_save = 1

# * ------------ Parse CLI args ----------------
parser = hlp.init_cliparser()  # Initialize parser
cli_args = parser.parse_args()  # Parse arguments from CLI into cli_args
# [overwrite CLI args manually (for debugging)]
# cli_args = parser.parse_args(
#     [
#         "-m",  # Modulation: M-{PAM/ASK/SQAM/QAM} where M is the constellation size
#         "4-ASK",
#         "-S",  # Number of SIC stages
#         "4",
#         "-s",  # Simulate only individual stage
#         "2",
#         "-d",  # Choose CPU for training
#         "cpu",
#     ]
# )
# * ---------- Processes CLI arguments  -------------------
# Modulation format string: modf_str
# Number of SIC stages: S_SIC
# Individual stages to simulate: S_SIC_vec_1idx [vector]
# Device for simulation: dev (either CPU or GPU)
modf_str, S_SIC, S_SIC_vec_1idx, dev = hlp.process_cliargs(cli_args)

# * ------------------------------------------------ *
# * ----------------- Modify ----------------------- *
# * ------------------------------------------------ *

# * ---------- Average transmit power -------------------
# Vary TX power; AWGN power fixed in channel() class
Ptx_dB_vec = np.arange(-5, 12, 2)
L_snr = len(Ptx_dB_vec)  # Number of SNR steps

# * -----  Differential phase precoding flag ------
f_diff_precoding = True  # Needed for nonl_f(.) = SLD

# Noise depending on real/complex signalling
f_cplx_AWGN = False  # If 0: real AWGN, if 1: c.s. complex AWGN

# * ---  Standard Single-Mode Fiber (SSMF) Parameters ---
L_SSMF = 0e3  # Length [m]
R_sym = 35 * 1e9  # Symbol rate in [Bd] or [Symb/s]

# * -------  Successive Interference Cancellation  -------
# Select L_SIC "closest" previously decoded symbols for SIC
L_SIC = 16


# * -------  Memoryless Nonlinearity -------------
def nonl_f(x, Ptx):
    # --- Linear System  ---
    # z = x

    # --- RX Square-Law Detector   ---
    z = np.abs(x) ** 2

    # --- TX power amplifier amplifier ---
    # PmaxdB = 6  # Maximum power in dBW per real and imaginary component
    # Pmaxlin = 10 ** (PmaxdB / 10)
    # a_sc = np.sqrt(Pmaxlin)
    # z = a_sc * (np.tanh(np.real(x) / a_sc) + 1j * np.tanh(np.imag(x) / a_sc))

    # --- TX DAC with 1-bit resolution ---
    # z = np.sqrt(Ptx/2)*np.sign(np.real(x)) + 1j * np.sqrt(Ptx/2)*np.sign(np.imag(x))

    return z


# * ----------- Neural Network ------------
# Specify the _effective_ NN Layer sizes.
# - The NN does _composite-real_ processing for complex channel outputs and SIC symbols
# - Therefore some dimensions may change depending on real/complex inputs
# - Define: r = [y,x] Input vector consisting of observations y and SIC symbols x
# - Observations y:
#   a) For real channel outputs: input layer size is sz[0]
#   b) For complex channel outputs: input layer size is 2 x sz[0]
# - SIC symbols x:
#   a) For real SIC the input size sz[0] is automatically extended by L_SIC
#   b) For complex SIC the input size sz[0] is automatically extended by 2 x L_SIC
# Note: The print functions display "effective" layer sizes

# Figure: NN operations and sizes:
# -------------
# FW dir. r -> [sz[0] x sz[1]] -\         /-> [2 sz[1] x sz[2]] ... -\
#                                -Concat->                            -Concat-> (Lin. Layer, Softmax) -> M
# BW dir. r -> [sz[0] x sz[1]] -/         \-> [2 sz[1] x sz[2]] ... -/
# -------------

# Layer input sizes according to the above figure
# Note: The output size is always the modulation alphabet size M
szNNvec = np.array([32, 32])

# NN Training parameters
lr = 0.02  # Learning rate
Nimax = 30000  # Maximum number of iterations for training
T_rnn_raw = 36  # Use approximate T_rnn inputs for training (ceil-ed later)
n_batch = 512  # Batches for SGD: Tensor with Dimensions: nBatch x T_rnn x sz[0]
n = 10000  # Set approximate number of validation symbols (ceil-ed later)
n_frames = 100  # Number of frames for final verification

# Training uses a learning rate (LR) scheduler. Every 'n_sched' training iterations the scheduler estimates the current rate using 'n_frames_sched_ver' frames each with 'n' symbols. If no improvement is seen for 10 subsequent calls, the learning rate is reduced by a factor of 0.3.
n_sched = 100  # Number of training iterations after which LR scheduler is called
n_frames_sched_ver = 1  # Number of frames for LR scheduler verification

# * ------- Physical Channel ----------
# Generate "C-Band" SSMF channel instance
# Other channel classes may be written and passed to comsys.channel()
SSMF_chan = comsys.SSMF("C", L_SSMF, R_sym)
# CH6_chan = comsys.CH6()

N_sim = 2  # Oversampling for simulation
d = 1  # Downsample by interger "d" after filtering with h[k]
N_os = N_sim // d  # Must be integer

# * ----- Symbol-wise TX precoder (optional) -------
g_pre = np.array([1])  # No precoder
g_pre = comsys.norm_ps(N_sim=1, filt=g_pre)  # Normalize precoder to unit energy

# * ---------TX DAC FILTER g_ps[u] -------------
tx_rolloff = 0.0
# Raised-cosine pulse
g_ps, _ = comsys.gen_ps_rcos(N_sim=N_sim, N_span=151, alph_roll=tx_rolloff)
g_ps = comsys.norm_ps(N_sim=N_sim, filt=g_ps)  # Normalize to unit energy

# * --------- RX ADC FILTER h[u] -------------
h = [1]  # delta[k]
rx_cutoff = 0.9999  # Relative to N_sim
# RX front-end filter
# h = scipy.signal.firwin(numtaps=201, cutoff=rx_cutoff, window="boxcar")


# * --------------------------------------------------
# * ------------- Filename for results ---------------
# * --------------------------------------------------

# Create channel instance
mychan = comsys.channel(
    g_pre,
    g_ps,
    tx_rolloff,
    h,
    rx_cutoff,
    nonl_f,
    SSMF_chan,  # Pass fiber channel
    N_sim,
    d,
    f_cplx_AWGN,
    modf_str,
    f_diff_precoding,
)


# For multiprocessing
def simulate_stage(s):
    SICstage_s = comsys.SICstage(
        vstr,  # Version
        s,  # Stage index in [1,...S]
        mychan,  # Pass channel instance
        dev=dev,
        Ptx_dB_vec=Ptx_dB_vec,
        szNNvec=szNNvec,
        lr=lr,
        Nimax=Nimax,
        n=n,
        n_frames=n_frames,
        n_frames_sched_ver=n_frames_sched_ver,
        n_batch=n_batch,
        S_SIC=S_SIC,
        L_SIC=L_SIC,
        L_snr=L_snr,
        N_os=N_os,
        N_sched=n_sched,
        T_rnn_raw=T_rnn_raw,
        S_SIC_vec_1idx=S_SIC_vec_1idx,
    )
    return SICstage_s.simulate()


if __name__ == "__main__":
    # * -------------- INFO -----------------
    hlp.printinfo()
    print("Running code on: " + str(dev) + "\n")
    time.sleep(3)

    n_procs = len(S_SIC_vec_1idx)
    with Pool(n_procs) as pool:  # Parallelize across SIC stages
        processed = pool.map(simulate_stage, S_SIC_vec_1idx)

    # For loop for timing analysis and debugging
    # for jj in S_SIC_vec_1idx:
    #     simulate_stage(jj)

    # Stitch together:
    SER_mat = np.zeros((S_SIC, L_snr))
    I_qXY_mat = np.zeros((S_SIC, L_snr))

    kk = 0
    for i in S_SIC_vec_1idx - 1:
        SER_mat[i, :] = processed[kk][0]
        I_qXY_mat[i, :] = processed[kk][1]
        kk = kk + 1

    filename = "results/ex1a_" + processed[0][2]  # Filename
    c_comp = processed[0][3]  # Complexity in no. of mult per APP estimate

    # Print a summary of results
    I_qXY = np.mean(I_qXY_mat, axis=0)
    SER = np.mean(SER_mat, axis=0)
    hlp.summary(Ptx_dB_vec, S_SIC, filename, I_qXY_mat, I_qXY)

    # Save individual SIC rates and average rate:
    if f_save == 1:
        hlp.saveresults(
            Ptx_dB_vec, L_snr, S_SIC, filename, SER_mat, I_qXY_mat, I_qXY, SER, c_comp
        )

    if f_showplot == 1:
        hlp.plot_results(Ptx_dB_vec, S_SIC, SER_mat, I_qXY_mat, I_qXY, SER)
