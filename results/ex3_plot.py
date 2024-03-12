import numpy as np
import matplotlib.pyplot as plt  # for debugging

## Helper for visualization of plots

plt.rcParams["text.usetex"] = True
plt.rcParams["figure.figsize"] = [8, 4]
plt.rcParams.update({"font.size": 9})


filename = [""] * 3

filename[0] = (
    "ex2a_v1.1,4-ASK,2_8_8,lr=2E-03,Nt=1,Tr=1,B=1024,S=1,Ls=16,I=7.0E+04,V=5.0E+02,n=2.0E+04,a=0.3,Nsim=2,d=2,r=0.3,L=0.0E+00,Rs=3.5E+10,C=3.0E+02.txt"
)
filename[1] = (
    "ex3a_v1.1,4-ASK,64_64_64,lr=1E-02,Nt=63,Tr=32,B=512,S=1,Ls=16,I=1.0E+04,V=5.0E+02,n=2.0E+04,a=0.3,Nsim=4,d=4,r=0.3,L=0.0E+00,Rs=3.5E+10,C=1.4E+04.txt"
)
filename[2] = (
    "ex3b_v1.1,4-ASK,64_64_64,lr=1E-02,Nt=39,Tr=32,B=512,S=2,Ls=16,I=1.0E+04,V=5.0E+02,n=2.0E+04,a=0.3,Nsim=4,d=1,r=0.99999,L=0.0E+00,Rs=3.5E+10,C=1.4E+04.txt"
)


foldname = "results/"

# Nos = 1 (no nonlinearity)
MATj = np.loadtxt(foldname + filename[0], delimiter=",", skiprows=1)
plt.plot(MATj[:, 0] + 3, MATj[:, 1], "-", linewidth="0.5")

# SDD 1: Nos=1
MATj = np.loadtxt(foldname + filename[1], delimiter=",", skiprows=1)
plt.plot(MATj[:, 0] + 3, MATj[:, 1], "-", linewidth="0.5")

# SDD 1: Nos=4
MATj = np.loadtxt(foldname + filename[2], delimiter=",", skiprows=1)
plt.plot(MATj[:, 0] + 3, MATj[:, 2], "--", linewidth="0.5")

# SIC 2: Nos=4
MATj = np.loadtxt(foldname + filename[2], delimiter=",", skiprows=1)
plt.plot(MATj[:, 0] + 3, MATj[:, 1], "--", linewidth="0.5")

# Gaussian bounds:
SNRdB = MATj[:, 0] + 3
SNRdBlin = 10 ** (SNRdB / 10)
plt.plot(SNRdB, 1 / 2 * np.log2(1 + SNRdBlin), "k-", linewidth="0.5")

plt.legend(
    (
        "Linear",
        "SDD, $N_\mathrm{os}=1$, $P_\mathrm{max}= 7\mathrm{dBW}$",
        "SDD, $N_\mathrm{os}=4$, $P_\mathrm{max} = 7\mathrm{dBW}$",
        "$S=2$, $N_\mathrm{os}=4$, $P_\mathrm{max}= 7\mathrm{dBW}$",
        "Gaussian",
    ),
)

plt.ylabel("Rate [bpcu]")
plt.xlabel("SNR [dB]")
plt.grid(1)
plt.subplots_adjust(left=0.25, right=0.75, top=0.9, bottom=0.15) #Add whitespace
plt.xlim((-5, 30))
plt.ylim((0, 2))

plt.title(
    r"Baseband Communication with Nonlinear Transmit Amplifier",
    fontsize=12,
)

plt.savefig("png/ex3.png", dpi=500)
plt.show()
plt.pause(1)
