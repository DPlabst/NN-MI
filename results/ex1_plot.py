import numpy as np
import matplotlib.pyplot as plt  # for debugging

## Helper for visualization of plots
plt.rcParams["text.usetex"] = True
plt.rcParams["figure.figsize"] = [8, 4]
plt.rcParams.update({"font.size": 9})
fig, (ax1, ax2) = plt.subplots(1, 2)

#### ---------------- Without Fiber ---------------------------

filename = [""] * 1

filename[0] = (
    "ex1a_v1.1,4-ASK,32_64,lr=2E-02,Nt=51,Tr=36,B=512,S=4,Ls=16,I=3.0E+04,V=1.0E+02,n=9.9E+03,a=0.0,Nsim=2,d=1,r=0.9999,L=0.0E+00,Rs=3.5E+10,C=5.4E+03.txt"
)

foldname = "results/"

for j in np.arange(len(filename)):
    MATj = np.loadtxt(foldname + filename[j], delimiter=",", skiprows=1)
    ax1.plot(MATj[:, 0], MATj[:, 1], "r-", linewidth="1.5")

    # Stage rates:
    ax1.plot(MATj[:, 0], MATj[:, 2], "b--", linewidth="0.5")
    ax1.plot(MATj[:, 0], MATj[:, 3:6], "-", linewidth="0.5")


ax1.legend(
    (
        "SIC",
        "SDD",
        "Stage 2",
        "Stage 3",
        "Stage 4",
    )
)
ax1.set_ylim((0, 2.1))
ax1.set_title(r"L=0km")
ax1.set_ylabel("Rate [bpcu]")
ax1.set_xlabel("SNR [dB]")
ax1.grid(1)


# #### ---------------- With Fiber ---------------------------

filename = [""] * 1

filename[0] = (
    "ex1b_v1.1,4-ASK,32_128_64,lr=1E-02,Nt=79,Tr=64,B=1024,S=4,Ls=32,I=2.0E+04,V=1.0E+02,n=1.0E+04,a=0.0,Nsim=2,d=1,r=0.9999,L=3.0E+04,Rs=3.5E+10,C=2.7E+04.txt"
)

for j in np.arange(len(filename)):
    MATj = np.loadtxt(foldname + filename[j], delimiter=",", skiprows=1)
    ax2.plot(MATj[:, 0], MATj[:, 1], "r-", linewidth="1.5")

    # Stage rates:
    ax2.plot(MATj[:, 0], MATj[:, 2], "b--", linewidth="0.5")
    ax2.plot(MATj[:, 0], MATj[:, 3:6], "-", linewidth="0.5")

SNRdB = MATj[:, 0]

ax2.legend(
    (
        "SIC",
        "SDD",
        "Stage 2",
        "Stage 3",
        "Stage 4",
    )
)
ax2.set_ylim((0, 2.1))
ax2.set_title(r"L=30km")
ax2.set_ylabel("Rate [bpcu]")
ax2.set_xlabel("SNR [dB]")
ax2.grid(1)

fig.tight_layout()
fig.subplots_adjust(top=0.84)
fig.suptitle(r"Square-Law Detection: SIC $S=4$, 4-ASK, TX DAC: SINC", fontsize=12)

plt.savefig("png/ex1.png", dpi=500)
plt.show()
plt.pause(1)
