import argparse
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# * --------  Save all results to a CSV file --------- 
def saveresults(
    SNR_dB_vec, L_snr, S_SIC, filename, SER_mat, I_qXY_mat, I_qXY, SER, c_comp
):
    SER_header = ""
    SIC_header = ""
    for i in range(S_SIC):
        SIC_header = SIC_header + "SIC" + str(i + 1) + ","
        SER_header = SER_header + "SER" + str(i + 1) + ","

        # Add other stuff:
        # - [complexity per APP estimate in real multiplications]
        # ...
    V_other = np.zeros((L_snr))
    V_other[0] = c_comp

    MAT = np.vstack(
        (
            SNR_dB_vec,
            I_qXY,  # Average mismatched information rate
            I_qXY_mat,  # Individual SIC stage rates
            SER,  # Average SER
            SER_mat,  # Individual SERs
            V_other,  # Complexity, ...
        )
    )

    np.savetxt(
        filename,
        MAT.T,
        delimiter=",",
        header="Ptxdb,IqXY," + SIC_header + "SER," + SER_header + "Other",
        comments="",
    )


# * -------- Print a summary with the results after finishing the simulation ------- 
def summary(SNR_dB_vec, S_SIC, filename, I_qXY_mat, I_qXY):
    print("\n---------------")
    print("Summary:")
    print(filename)
    SNR_list = SNR_dB_vec.tolist()
    SNR_list.insert(0, "SNR")
    IqXY_list = I_qXY.tolist()
    IqXY_list.insert(0, "Avg. IqXY")
    IqXY_per_stage = np.zeros((I_qXY_mat.shape[0], I_qXY_mat.shape[1] + 1))
    IqXY_per_stage[0:, 0] = np.arange(S_SIC)
    IqXY_per_stage[0:, 1:] = I_qXY_mat

    Summary_list = IqXY_per_stage.tolist()
    for jlist in np.arange(S_SIC):
        Summary_list[jlist][0] = "SIC-" + str(jlist + 1)

    Summary_list.insert(0, SNR_list)
    Summary_list.append(IqXY_list)

    print(
        tabulate(
            Summary_list,
            floatfmt=".3f",
            tablefmt="plain",
            numalign="right",
            stralign="right",
        )
    )


# * -------- Process arguments from CLI ------- 
# (may be overwritten in main file for debugging)
def process_cliargs(cli_args):
    if cli_args.mod_format is not None:
        modf_str = cli_args.mod_format

    # * -------------- SIC Setup ------------
    if cli_args.stages is not None:  # If SIC stages are set via CLI
        S_SIC = cli_args.stages

    if cli_args.indiv is not None:
        S_SIC_vec = np.array([cli_args.indiv], dtype=int)
    else:
        # Default: compute all SIC stages
        S_SIC_vec = np.arange(1, S_SIC + 1, dtype=int)
        # S_SIC_vec = np.array([3]) #Set number of stages, if no CLI parameters provided

    # * -------------- CUDA or CPU -----------------
    if cli_args.device is not None:
        if cli_args.device == "cuda":
            dev = "cuda:0"  # RUN on cuda:0
        elif cli_args.device == "cpu":
            dev = "cpu"  # RUN on CPU
    else:
        # Use CPU is not specified
        dev = "cpu"

    return modf_str, S_SIC, S_SIC_vec, dev

# * -------- Plot results after simulation -------
def plot_results(Ptx_dB_vec, S_SIC, SER_mat, I_qXY_mat, I_qXY, SER):
    lab = ["SIC-" + str(x) for x in range(1, S_SIC + 1)]
    lab.append("Average")
    plt.figure(1)
    plt.semilogy(Ptx_dB_vec, SER_mat.T, "x-")
    plt.semilogy(Ptx_dB_vec, SER, "rx-")
    plt.legend(labels=lab)
    plt.ylim((1e-3, 0.5))
    plt.grid()
    plt.title("SER")

    plt.figure(2)
    plt.plot(Ptx_dB_vec, I_qXY_mat.T, "x-")
    plt.plot(Ptx_dB_vec, I_qXY.T, "rx-")
    plt.legend(labels=lab)
    plt.grid()
    plt.title("Rate")

    plt.show()
    plt.pause(1)


# * --------------- CLI parser --------------
def init_cliparser():
    formatter = lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60)
    parser = argparse.ArgumentParser(
        description="NN-MI: Neural Network Mutual Information Computation for Channels with Memory",
        formatter_class=formatter,
    )
    parser.add_argument(
        "--stages",
        "-S",
        type=np.int64,
        help="number of successive interference cancellation stages",
    )
    parser.add_argument(
        "--mod_format",
        "-m",
        type=str,
        help="""M-ASK, M-PAM, M-SQAM (star-QAM), M-QAM (square) modulation with order M""",
    )
    parser.add_argument(
        "--indiv", "-i", type=np.int64, help="simulation of a single individual stage"
    )
    parser.add_argument(
        "--device",
        "-d",
        choices=["cpu", "cuda"],
        type=str,
        help="run code on cpu or cuda",
    )
    return parser


def printinfo():
    print("**************************************************")
    print(
        " * NN-MI: Neural Network Mutual Information Computation for Channels with Memory"
    )
    print(" * Daniel Plabst")
    print(" * Institute for Communications Engineering (LNT)")
    print(" * Technical University of Munich, Germany")
    print(" * http://ice.cit.tum.de/")
    print(" * Public version: v1.0 2024-01-17")
    print("**************************************************")
