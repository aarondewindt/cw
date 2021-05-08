import matplotlib.pyplot as plt

def plot_state(data, state_name):
    fig, (ax1, ax2) = plt.subplots(2)

    plt.sca(ax1)
    for name in [state_name, f"{state_name}_filtered", f"{state_name}_smoothed"]:
        if name in data:
            data[name].plot.line(x="t", label=name)

    plt.sca(ax2)
    ax2.set_yscale('log')
    for name in [f"{state_name}_filtered_std", f"{state_name}_smoothed_std"]:
        if name in data:
            data[name].plot.line(x="t", label=name)

