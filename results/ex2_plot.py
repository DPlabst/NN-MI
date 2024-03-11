import numpy as np
import matplotlib.pyplot as plt  # for debugging

## Helper for visualization of plots
# Note: SNR is defined differently, so we add 3dB everywhere

plt.rcParams["text.usetex"] = True
plt.rcParams["figure.figsize"] = [8, 4]
plt.rcParams.update({"font.size": 10})
fig, (ax1, ax2) = plt.subplots(1, 2)

#### ---------------- REAL ---------------------------
plt.figure(1)
filename = [""] * 7

filename[0] = (
    "ex2a_v1.1,2-PAM,2_8_8,lr=2E-03,Nt=1,Tr=1,B=1024,S=1,Ls=16,I=7.0E+04,V=5.0E+02,n=2.0E+04,a=0.3,Nsim=2,d=2,r=0.3,L=0.0E+00,Rs=3.5E+10,C=2.9E+02.txt"
)
filename[1] = (
    "ex2a_v1.1,2-ASK,2_8_8,lr=2E-03,Nt=1,Tr=1,B=1024,S=1,Ls=16,I=7.0E+04,V=5.0E+02,n=2.0E+04,a=0.3,Nsim=2,d=2,r=0.3,L=0.0E+00,Rs=3.5E+10,C=2.9E+02.txt"
)
filename[2] = (
    "ex2a_v1.1,4-ASK,2_8_8,lr=2E-03,Nt=1,Tr=1,B=1024,S=1,Ls=16,I=7.0E+04,V=5.0E+02,n=2.0E+04,a=0.3,Nsim=2,d=2,r=0.3,L=0.0E+00,Rs=3.5E+10,C=3.0E+02.txt"
)
filename[3] = (
    "ex2a_v1.1,8-ASK,2_8_8,lr=2E-03,Nt=1,Tr=1,B=1024,S=1,Ls=16,I=7.0E+04,V=5.0E+02,n=2.0E+04,a=0.3,Nsim=2,d=2,r=0.3,L=0.0E+00,Rs=3.5E+10,C=3.4E+02.txt"
)
filename[4] = (
    "ex2a_v1.1,16-ASK,2_8_8,lr=2E-03,Nt=1,Tr=1,B=1024,S=1,Ls=16,I=7.0E+04,V=5.0E+02,n=2.0E+04,a=0.3,Nsim=2,d=2,r=0.3,L=0.0E+00,Rs=3.5E+10,C=4.0E+02.txt"
)
filename[5] = (
    "ex2a_v1.1,32-ASK,2_8_8,lr=2E-03,Nt=1,Tr=1,B=1024,S=1,Ls=16,I=7.0E+04,V=5.0E+02,n=2.0E+04,a=0.3,Nsim=2,d=2,r=0.3,L=0.0E+00,Rs=3.5E+10,C=5.3E+02.txt"
)
filename[6] = (
    "ex2a_v1.1,64-ASK,2_8_8,lr=2E-03,Nt=1,Tr=1,B=1024,S=1,Ls=16,I=7.0E+04,V=5.0E+02,n=2.0E+04,a=0.3,Nsim=2,d=2,r=0.3,L=0.0E+00,Rs=3.5E+10,C=7.8E+02.txt"  # Looks good!
)


foldname = "results/"

for j in np.arange(len(filename)):
    MATj = np.loadtxt(foldname + filename[j], delimiter=",", skiprows=1)
    ax1.plot(MATj[:, 0] + 3, MATj[:, 1], "b-", linewidth="0.5")


SNRdB = MATj[:, 0] + 3  # Add 3dB because of noise power definition

# Gaussian bounds:
SNRdBlin = 10 ** (SNRdB / 10)
ax1.plot(SNRdB, 1 / 2 * np.log2(1 + SNRdBlin), "k", linewidth="0.5")

SNRdBlin = 10 ** ((SNRdB - 1.53) / 10)
ax1.plot(SNRdB, 1 / 2 * np.log2(SNRdBlin), "k-.", linewidth="0.5")


xval = 26.7
yval = 1 / 2 * np.log2(1 + 10 ** (xval / 10))
ax1.plot(
    np.array([xval, xval + 1.53]),
    yval * np.ones(2),
    "r.-",
    linewidth="1",
    markersize="1.5",
)
ax1.text(xval + 7, yval - 0.1, "1.53dB", color="red", fontsize=15)


ax1.legend(("OOK", "BPSK", "4-ASK", "8-ASK", "16-ASK", "32-ASK", "64-ASK", "Gaussian"))
ax1.set_ylim((0, 6.0))
ax1.set_ylabel("Rate [bpcu]")
ax1.set_xlabel("SNR [dB]")
ax1.grid(1)
ax1.set_title(r"Real Modulation")


# #### ---------------- COMPLEX ---------------------------

filename = [""] * 3

filename[0] = (
    "ex2a_v1.1,4-QAM,2_8_8,lr=2E-03,Nt=1,Tr=1,B=1024,S=1,Ls=16,I=7.0E+04,V=5.0E+02,n=2.0E+04,a=0.3,Nsim=2,d=2,r=0.3,L=0.0E+00,Rs=3.5E+10,C=4.3E+02.txt"
)
filename[1] = (
    "ex2a_v1.1,16-QAM,2_8_8,lr=2E-03,Nt=1,Tr=1,B=1024,S=1,Ls=16,I=7.0E+04,V=5.0E+02,n=2.0E+04,a=0.3,Nsim=2,d=2,r=0.3,L=0.0E+00,Rs=3.5E+10,C=5.3E+02.txt"
)
filename[2] = (
    "ex2a_v1.1,64-QAM,2_8_8,lr=2E-03,Nt=1,Tr=1,B=1024,S=1,Ls=16,I=7.0E+04,V=5.0E+02,n=2.0E+04,a=0.3,Nsim=2,d=2,r=0.3,L=0.0E+00,Rs=3.5E+10,C=9.1E+02.txt"
)


for j in np.arange(len(filename)):
    MATj = np.loadtxt(foldname + filename[j], delimiter=",", skiprows=1)
    ax2.plot(MATj[:, 0], MATj[:, 1], "g-", linewidth="0.5")

SNRdB = MATj[:, 0]

# Gaussian bounds:
SNRdBlin = 10 ** (SNRdB / 10)
ax2.plot(SNRdB, np.log2(1 + SNRdBlin), "k", linewidth="0.5")

SNRdBlin = 10 ** ((SNRdB - 1.53) / 10)
ax2.plot(SNRdB, np.log2(SNRdBlin), "k-.", linewidth="0.5")

xval = 16.2
yval = np.log2(1 + 10 ** (xval / 10))
ax2.plot(
    np.array([xval, xval + 1.53]),
    yval * np.ones(2),
    "r.-",
    linewidth="1",
    markersize="1.5",
)
ax2.text(xval - 13, yval - 0.1, "1.53dB", color="red", fontsize=15)

ax2.legend(("QPSK", "16-QAM", "64-QAM", "Gaussian"))
ax2.set_ylim((0, 6.0))
ax2.set_ylabel("Rate [bpcu]")
ax2.set_xlabel("SNR [dB]")
ax2.grid(1)
ax2.set_title(r"Complex Modulation")

fig.tight_layout()
fig.subplots_adjust(top=0.84)
fig.suptitle(
    r"Linear: SDD: $S=1$,TX DAC: RRC, roll-off $\alpha=0.3$, RX: Matched filter",
    fontsize=14,
)

plt.savefig("png/ex2a.png", dpi=500)


# --------------
# --------------

plt.figure(2)

filename = [""] * 5

filename[0] = "FBA_CH6_JDD_SDD_rates.csv"

# NN:
filename[1] = (
    "ex2b_v1.1,2-ASK,8_140_140,lr=5E-03,Nt=1031,Tr=1024,B=128,S=1,Ls=32,I=3.0E+04,V=5.0E+02,n=1.9E+04,a=0.0,Nsim=1,d=1,r=0.0,L=0.0E+00,Rs=0.0E+00,C=4.5E+04.txt"
)
filename[2] = (
    "ex2b_v1.1,2-ASK,8_140_140,lr=5E-03,Nt=1033,Tr=1026,B=128,S=4,s=2,Ls=32,I=3.0E+04,V=5.0E+02,n=2.5E+04,a=0.0,Nsim=1,d=1,r=0.0,L=0.0E+00,Rs=0.0E+00,C=4.5E+04.txt"
)
filename[3] = (
    "ex2b_v1.1,2-ASK,8_140_140,lr=5E-03,Nt=1031,Tr=1024,B=128,S=4,s=3,Ls=32,I=3.0E+04,V=5.0E+02,n=3.7E+04,a=0.0,Nsim=1,d=1,r=0.0,L=0.0E+00,Rs=0.0E+00,C=4.5E+04.txt"
)
filename[4] = (
    "ex2b_v1.1,2-ASK,8_140_140,lr=5E-03,Nt=1031,Tr=1024,B=128,S=4,s=4,Ls=32,I=3.0E+04,V=5.0E+02,n=7.8E+04,a=0.0,Nsim=1,d=1,r=0.0,L=0.0E+00,Rs=0.0E+00,C=4.5E+04.txt"
)


# FBA: SDD
MATj = np.loadtxt(foldname + filename[0], delimiter=",", skiprows=1)
plt.plot(MATj[:, 0] + 3, MATj[:, 1], "b-", linewidth="0.5")

# FBA: JDD
MATj = np.loadtxt(foldname + filename[0], delimiter=",", skiprows=1)
plt.plot(MATj[:, 0] + 3, MATj[:, 2], "r-", linewidth="0.5")

# NN: SDD
MATs1 = np.loadtxt(foldname + filename[1], delimiter=",", skiprows=1)
plt.plot(MATs1[:, 0] + 3, MATs1[:, 2], "r--", linewidth="0.5")

# NN: SIC
MATs2 = np.loadtxt(foldname + filename[2], delimiter=",", skiprows=1)
MATs3 = np.loadtxt(foldname + filename[3], delimiter=",", skiprows=1)
MATs4 = np.loadtxt(foldname + filename[4], delimiter=",", skiprows=1)

# S=4
plt.plot(
    MATs1[:, 0] + 3,
    1 / 4 * (MATs1[:, 2] + MATs2[:, 3] + MATs3[:, 4] + MATs4[:, 5]),
    "b--",
    linewidth="0.5",
)

# Gaussian bounds:
SNRdB = MATj[:, 0] + 3  # SNR is defined differently
SNRdBlin = 10 ** ((SNRdB) / 10)
plt.plot(SNRdB, 1 / 2 * np.log2(1 + SNRdBlin), "k-", linewidth="0.5")

# Measure:
xval = 7.96 + 3
yval = 0.8
plt.plot(
    np.array([xval, xval + 0.45]),
    yval * np.ones(2),
    "k.-",
    linewidth="1.0",
    markersize="3",
)
plt.text(xval - 4.4, yval, r"$\approx$ 0.45dB", color="black", fontsize=15)


plt.ylim((0, 1.01))
plt.xlim((-5, 20))

plt.legend(("FBA-JDD", "FBA-SDD", "NN-SDD", "NN-SIC $S=4$", "Gaussian (real)"))

plt.title("CH6 Channel h[k] = [0.19, 0.35, 0.46, 0.5, 0.46, 0.35, 0.19]")
plt.ylabel("Rate [bpcu]")
plt.xlabel("SNR [dB]")
plt.grid(1)
plt.savefig("png/ex2b.png", dpi=500)

plt.show()
plt.pause(1)
plt.pause(2)
