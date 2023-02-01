from .imports import *


class Plots(object):
    def __init__(self, path: str = f"./results/", fit_from_n: int = 3):
        assert os.path.isdir(path)
        if not os.path.isdir("./figs/"):
            os.mkdir("./figs/")
        self.path = path
        self.results = [x for x in os.listdir(path) if os.path.isdir(f"{path}{x}")]
        self.fit_from_n = fit_from_n
        self.markers = ["o", "o", "s", "x", "d", "4"]
        self.colors = plt.cm.plasma(np.linspace(0, 1, len(self.markers)))
        self.colors[1] = self.colors[0]
        self.colors[-1] = [0, 0, 0, 1.0]
        self.labels = [
            "\\epsilon \\rightarrow 0",
            None,
            "10^{-4}",
            "10^{-3}",
            "10^{-2}",
            "10^{-1}",
        ]

    def plot_all(self, save: bool = True) -> None:
        methods = [
            method_name
            for method_name in dir(self)
            if callable(getattr(self, method_name))
            and method_name.startswith("plot_")
            and method_name != "plot_all"
        ]
        for method in methods:
            method_to_call = getattr(self, method)
            method_to_call(save)

    def alpha_func(self, n, alpha, beta):
        xi = n * self.eps
        return alpha * np.exp(-xi) * xi / ((1 - np.exp(-xi)) * n ** beta)

    def a_b_func(self, x, a, b):
        return a * x ** b

    def save_plot_data(self, data: Dict, settings: Dict, savestr: str):
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                data.update({k: v.tolist()})
        json_dump = json.dumps(data)
        settings_str = self.return_settings_str(settings)
        with open(f"figs/data_{savestr}_{settings_str}.json", "w") as f:
            f.write(json_dump)

    def plot_comnorm_gates(
        self,
        settings: Dict,
        add_fit: bool = False,
        add_alpha_n: bool = False,
        legend: bool = True,
        save: bool = False,
    ):
        data = None
        for folder in self.results:
            with open(f"{self.path}{folder}/parameters.json", "r") as file:
                parameters = json.load(file)

            if not settings.items() <= parameters.items():
                continue

            with open(f"{self.path}{folder}/results.json", "r") as file:
                data = json.load(file)
            break
        if data is None:
            raise ValueError(f"Data with parsed settings not found in {self.path}")

        fig, axes = plt.subplots(1, 1)
        p_noises_str = data["p_noises_str"]
        fidelities = np.array(data["fidelities"])
        commutator_norms = np.array(data["commutator_norms"]) / (1 - fidelities)
        xaxis = np.array(data["n_gates"])
        xaxis_extended = np.append(xaxis[:-1], np.geomspace(xaxis[-1], 10 ** 4))
        colors = plt.cm.plasma(np.linspace(0, 1, len(p_noises_str)))
        dists_mean = np.nanmean(commutator_norms, axis=0).T
        dists_std = np.nanstd(commutator_norms, axis=0).T

        if save:
            self.save_plot_data(
                data={"x": xaxis, "y": dists_mean, "labels": p_noises_str},
                settings=settings,
                savestr="comnorm",
            )

        for i_p, p_str in enumerate(p_noises_str):
            data = dists_mean[i_p]
            self.eps = float(p_str)
            if add_fit and self.eps < 1e-3:
                popt, _ = curve_fit(
                    self.alpha_func, xaxis[self.fit_from_n :], data[self.fit_from_n :]
                )
                fit = self.alpha_func(xaxis_extended, popt[0], popt[1])
                axes.plot(xaxis_extended, fit, color=colors[i_p])
                # ,alpha=0.5
            labelstr = (
                p_str.replace("1e-", "10^{-") + "}"
            )  # f"{p_str:.0E}".replace("1E-0","10^{-")+"}
            axes.plot(
                xaxis,
                data,
                "o",
                color=colors[i_p],
                marker=self.markers[i_p],
                label=rf"${labelstr}$",
            )
            # axes.fill_between(xaxis,data-dists_std[i_p],data+dists_std[i_p],color=colors[i_p],alpha=0.1)

        plt.xlabel(r"$\nu$")
        plt.ylabel(r"$\frac{||[\rho_{id},\rho]||_1}{1-F}$")
        plt.yscale("log")
        plt.xscale("log")
        plt.ylim([10 ** 2, 10 ** (-4)])
        plt.gca().invert_yaxis()
        if legend:
            plt.legend()
            # bbox_to_anchor=(1.1, 1.05)

        if add_alpha_n:
            self.plot_alpha_N(settings, axes)
        if save:
            settings_str = self.return_settings_str(settings)
            fig.savefig(f"figs/n-gates--commutator-norms--{settings_str}.pdf")
        plt.show()

    def plot_l1_gates(
        self,
        settings: Dict,
        add_fit: bool = False,
        add_alpha_n: bool = False,
        legend: bool = True,
        save: bool = False,
    ):

        # Find data
        data = None
        for folder in self.results:
            with open(f"{self.path}{folder}/parameters.json", "r") as file:
                parameters = json.load(file)

            if not settings.items() <= parameters.items():
                continue

            with open(f"{self.path}{folder}/results.json", "r") as file:
                data = json.load(file)
            break
        if data is None:
            raise ValueError(f"Data with parsed settings not found in {self.path}")

        fig, axes = plt.subplots(1, 1)
        p_noises_str = data["p_noises_str"]
        l1s = np.array(data["l1s"])
        xaxis = np.array(data["n_gates"])
        xaxis_extended = np.append(xaxis[:-1], np.geomspace(xaxis[-1], 10 ** 4))
        dists_mean = np.nanmean(l1s / 2, axis=0).T
        dists_std = np.nanstd(l1s / 2, axis=0).T
        self.save_plot_data(
            data={"x": xaxis, "y": dists_mean, "labels": p_noises_str},
            settings=settings,
            savestr="l1s",
        )

        for i_p, p_str in enumerate(p_noises_str):
            data = dists_mean[i_p]
            self.eps = float(p_str)
            if add_fit and self.eps < 1e-3:
                popt, _ = curve_fit(
                    self.alpha_func, xaxis[self.fit_from_n :], data[self.fit_from_n :]
                )
                fit = self.alpha_func(xaxis_extended, popt[0], popt[1])
                axes.plot(xaxis_extended, fit, "--", color=self.colors[i_p])
                # ,alpha=0.5
            labelstr = (
                p_str.replace("1e-", "10^{-") + "}"
            )  # f"{p_str:.0E}".replace("1E-0","10^{-")+"}
            axes.plot(
                xaxis,
                data,
                "o",
                color=self.colors[i_p],
                marker=self.markers[i_p],
                label=rf"${labelstr}$",
            )
            # axes.fill_between(xaxis,data-dists_std[i_p],data+dists_std[i_p],color= self.colors[i_p],alpha=0.1)

        plt.xlabel(r"$\nu$")
        plt.ylabel(r"$\frac{1}{2}||p_{noisy}-p_{wn}||_1$")
        plt.yscale("log")
        plt.xscale("log")
        plt.ylim([10 ** 2, 10 ** (-4)])
        # plt.xlim([10, 10**(4)]);
        plt.gca().invert_yaxis()
        if legend:
            plt.legend()
            # bbox_to_anchor=(1.1, 1.05)

        if add_fit and add_alpha_n:
            self.plot_alpha_N(settings, axes)
        if save:
            settings_str = self.return_settings_str(settings)
            fig.savefig(f"figs/n-gates--l1dist--{settings_str}.pdf")
        plt.show()

    def return_settings_str(self, settings: Dict, remove: list = []):
        return_entries = ["hamiltonian", "vqe", "n_qubits", "ansatz", "add_scrambling"]
        return_entries = [x for x in return_entries if x not in remove]
        settings_ = dict(settings)
        for key in settings.keys():
            if key not in return_entries:
                settings_.pop(key, None)

        str_ = (
            f"{settings_}".replace(" ", "-")
            .replace(":", "-")
            .replace("_", "-")
            .replace("+", "-")
            .replace("{", "-")
            .replace("}", "-")
            .replace("'", "-")
            .replace(",", "")
            .replace("/", "")
            .replace("--", "-")
        )
        return str_

    def load_data(self, settings: Dict):
        data = None
        for folder in self.results:
            with open(f"{self.path}{folder}/parameters.json", "r") as file:
                parameters = json.load(file)

            if not settings.items() <= parameters.items():
                continue

            with open(f"{self.path}{folder}/results.json", "r") as file:
                data = json.load(file)
            break
        if data is None:
            raise ValueError(f"Data with parsed settings not found in {self.path}")
        return data

    def largeplot_sel(self, settings: Dict, save: bool = False, add_fit: bool = True):
        fig = plt.figure(figsize=(16, 6))  # ,constrained_layout=True);

        y_add_caption = -0.22
        y_lims = [10 ** 0, 10 ** (-3)]
        x_lims = [3 * 10 ** 1, 10 ** (4)]

        ### L1 - random
        settings.update({"vqe": False})
        data = self.load_data(settings)
        l1s = np.array(data["l1s"])
        xaxis = np.array(data["n_gates"])
        xaxis_extended = np.append(
            np.geomspace(10 ** (1), xaxis[0]),
            np.append(xaxis[1:-1], np.geomspace(xaxis[-1], 10 ** 4)),
        )
        dists_mean = np.nanmean(l1s / 2, axis=0).T
        p_noises_str = data["p_noises_str"]
        self.save_plot_data(
            data={"x": xaxis, "y": dists_mean, "labels": p_noises_str},
            settings=settings,
            savestr="l1s",
        )
        ax1 = fig.add_subplot(121)
        for i_p, p_str in enumerate(p_noises_str):
            l1 = dists_mean[i_p]
            self.eps = float(p_str)
            if self.eps < 1e-3:
                popt, _ = curve_fit(
                    self.alpha_func, xaxis[self.fit_from_n :], l1[self.fit_from_n :]
                )
                fit = self.alpha_func(xaxis_extended, popt[0], popt[1])
                ax1.plot(xaxis_extended, fit, "--", color=self.colors[i_p])

            if self.labels[i_p] is not None:
                ax1.plot(
                    xaxis,
                    l1,
                    "o",
                    color=self.colors[i_p],
                    marker=self.markers[i_p],
                    label=rf"${self.labels[i_p]}$",
                )
            else:
                ax1.plot(
                    xaxis, l1, "o", color=self.colors[i_p], marker=self.markers[i_p]
                )

        plt.xlabel(r"$\nu$")
        plt.ylabel(r"$W$ (eigenvalue uniformity)")
        plt.yscale("log")
        plt.xscale("log")
        plt.title(r"(a) $L_1$ distance, random parameters", y=y_add_caption)
        plt.ylim(y_lims)
        plt.xlim(x_lims)
        ax1.invert_yaxis()
        # plt.xlim(x_lims);
        # self.plot_alpha_N(settings,ax1,fit_y="l1s",add_fit=add_fit)

        # Commutator norm - random
        ax = fig.add_subplot(122)
        fidelities = np.array(data["fidelities"])
        commutator_norms = np.array(data["commutator_norms"]) / (1 - fidelities)
        comnorm_mean = np.nanmean(commutator_norms, axis=0).T
        self.save_plot_data(
            data={"x": xaxis, "y": dists_mean, "labels": p_noises_str},
            settings=settings,
            savestr="comnorm",
        )
        for i_p, p_str in enumerate(p_noises_str):
            comnorm = comnorm_mean[i_p]
            self.eps = float(p_str)
            if self.eps < 1e-7:
                continue
            if self.eps < 1e-3:
                popt, _ = curve_fit(
                    self.alpha_func,
                    xaxis[self.fit_from_n :],
                    comnorm[self.fit_from_n :],
                )
                fit = self.alpha_func(xaxis_extended, popt[0], popt[1])
                ax.plot(xaxis_extended, fit, "--", color=self.colors[i_p])
                # ,alpha=0.5

            if self.labels[i_p] is not None:
                ax.plot(
                    xaxis,
                    comnorm,
                    "o",
                    color=self.colors[i_p],
                    marker=self.markers[i_p],
                    label=rf"${self.labels[i_p]}$",
                )
            else:
                ax.plot(
                    xaxis,
                    comnorm,
                    "o",
                    color=self.colors[i_p],
                    marker=self.markers[i_p],
                )

        plt.xlabel(r"$\nu$")
        plt.ylabel(r"$C$ (commutator norm)")
        plt.yscale("log")
        plt.xscale("log")
        plt.title(r"(b) Commutator norms, random parameters", y=y_add_caption)
        plt.ylim(y_lims)
        plt.xlim(x_lims)
        ax.invert_yaxis()
        # plt.xlim(x_lims);
        # self.plot_alpha_N(settings,ax,fit_y="commutator_norms",add_fit=add_fit)

        handles, labels = ax1.get_legend_handles_labels()
        plt.tight_layout()
        fig.legend(handles, labels, loc=(0.22, -0.005), ncol=6)
        if save:
            settings_str = self.return_settings_str(settings)
            fig.savefig(f"figs/l1--comnorms--{settings_str}.pdf", bbox_inches="tight")
        plt.show()
        # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    def largeplot_lih(self, settings: Dict, save: bool = False, add_fit: bool = True):
        fig = plt.figure(figsize=(14, 10), constrained_layout=True)

        y_add_caption = -0.3
        y_lims = [10 ** 0, 10 ** (-8)]
        x_lims = [10 ** 2, 10 ** (5)]

        ### L1 - random
        settings.update({"vqe": False})
        data = self.load_data(settings)
        l1s = np.array(data["l1s"])
        xaxis = np.array(data["n_gates"])
        xaxis_extended = np.geomspace(x_lims[0], x_lims[-1])
        dists_mean = np.nanmean(l1s / 2, axis=0).T
        p_noises_str = data["p_noises_str"]
        self.save_plot_data(
            data={"x": xaxis, "y": dists_mean, "labels": p_noises_str},
            settings=settings,
            savestr="l1s",
        )
        ax1 = fig.add_subplot(221)
        for i_p, p_str in enumerate(p_noises_str):
            l1 = dists_mean[i_p]
            self.eps = float(p_str)
            if self.eps < 1e-3:
                popt, _ = curve_fit(
                    self.alpha_func, xaxis[self.fit_from_n :], l1[self.fit_from_n :]
                )
                fit = self.alpha_func(xaxis_extended, popt[0], popt[1])
                ax1.plot(xaxis_extended, fit, "--", color=self.colors[i_p])

            if self.labels[i_p] is not None:
                ax1.plot(
                    xaxis,
                    l1,
                    "o",
                    color=self.colors[i_p],
                    marker=self.markers[i_p],
                    label=rf"${self.labels[i_p]}$",
                )
            else:
                ax1.plot(
                    xaxis, l1, "o", color=self.colors[i_p], marker=self.markers[i_p]
                )

        plt.xlabel(r"$\nu$")
        plt.ylabel(r"$W$ (eigenvalue uniformity)")
        plt.yscale("log")
        plt.xscale("log")
        plt.xlim(x_lims)
        plt.ylim(y_lims)
        ax1.invert_yaxis()

        plt.title(r"(a) $L_1$ distance, random parameters", y=y_add_caption)

        # Commutator norm - random
        ax = fig.add_subplot(223)
        fidelities = np.array(data["fidelities"])
        commutator_norms = np.array(data["commutator_norms"]) / (1 - fidelities)
        comnorm_mean = np.nanmean(commutator_norms, axis=0).T
        self.save_plot_data(
            data={"x": xaxis, "y": dists_mean, "labels": p_noises_str},
            settings=settings,
            savestr="comnorm",
        )
        for i_p, p_str in enumerate(p_noises_str):
            comnorm = comnorm_mean[i_p]
            self.eps = float(p_str)
            if self.eps < 1e-7:
                continue
            if self.eps < 1e-3:
                popt, _ = curve_fit(
                    self.alpha_func,
                    xaxis[self.fit_from_n :],
                    comnorm[self.fit_from_n :],
                )
                fit = self.alpha_func(xaxis_extended, popt[0], popt[1])
                ax.plot(xaxis_extended, fit, "--", color=self.colors[i_p])
                # ,alpha=0.5

            if self.labels[i_p] is not None:
                ax.plot(
                    xaxis,
                    comnorm,
                    "o",
                    color=self.colors[i_p],
                    marker=self.markers[i_p],
                    label=rf"${self.labels[i_p]}$",
                )
            else:
                ax.plot(
                    xaxis,
                    comnorm,
                    "o",
                    color=self.colors[i_p],
                    marker=self.markers[i_p],
                )

        plt.xlabel(r"$\nu$")
        plt.ylabel(r"$C$ (commutator norm)")
        plt.yscale("log")
        plt.xscale("log")
        plt.title(r"(c) Commutator norms, random parameters", y=y_add_caption)
        plt.ylim(y_lims)
        plt.xlim(x_lims)
        ax.invert_yaxis()

        # L1 - VQE
        settings.update({"vqe": True})
        data = self.load_data(settings)
        l1s = np.array(data["l1s"])
        xaxis = np.array(data["n_gates"])
        xaxis_extended = np.geomspace(x_lims[0], x_lims[-1])
        dists_mean = np.nanmean(l1s / 2, axis=0).T
        p_noises_str = data["p_noises_str"]
        self.save_plot_data(
            data={"x": xaxis, "y": dists_mean, "labels": p_noises_str},
            settings=settings,
            savestr="l1s",
        )
        ax = fig.add_subplot(222)
        for i_p, p_str in enumerate(p_noises_str):
            l1 = dists_mean[i_p]
            self.eps = float(p_str)
            if self.eps < 1e-3:
                popt, _ = curve_fit(
                    self.alpha_func, xaxis[self.fit_from_n :], l1[self.fit_from_n :]
                )
                fit = self.alpha_func(xaxis_extended, popt[0], popt[1])
                ax.plot(xaxis_extended, fit, "--", color=self.colors[i_p])

            if self.labels[i_p] is not None:
                ax.plot(
                    xaxis,
                    l1,
                    "o",
                    color=self.colors[i_p],
                    marker=self.markers[i_p],
                    label=rf"${self.labels[i_p]}$",
                )
            else:
                ax.plot(
                    xaxis, l1, "o", color=self.colors[i_p], marker=self.markers[i_p]
                )

        plt.xlabel(r"$\nu$")
        plt.yscale("log")
        plt.xscale("log")
        plt.ylim(y_lims)
        plt.xlim(x_lims)
        ax.invert_yaxis()
        plt.title(r"(b) $L_1$ distance, VQE parameters", y=y_add_caption)

        # Commutator norm - VQE
        ax = fig.add_subplot(224)
        fidelities = np.array(data["fidelities"])
        commutator_norms = np.array(data["commutator_norms"]) / (1 - fidelities)
        comnorm_mean = np.nanmean(commutator_norms, axis=0).T
        self.save_plot_data(
            data={"x": xaxis, "y": dists_mean, "labels": p_noises_str},
            settings=settings,
            savestr="comnorm",
        )
        for i_p, p_str in enumerate(p_noises_str):
            comnorm = comnorm_mean[i_p]
            self.eps = float(p_str)
            if self.eps < 1e-7:
                continue
            # if self.eps < 1e-3:
            #   popt, _ = curve_fit(self.alpha_func, xaxis[ self.fit_from_n:], comnorm[ self.fit_from_n:])
            #   fit = self.alpha_func(xaxis_extended,popt[0],popt[1])
            #   ax.plot(xaxis_extended,fit,color=self.colors[i_p]);#,alpha=0.5

            if self.labels[i_p] is not None:
                ax.plot(
                    xaxis,
                    comnorm,
                    "o",
                    color=self.colors[i_p],
                    marker=self.markers[i_p],
                    label=rf"${self.labels[i_p]}$",
                )
            else:
                ax.plot(
                    xaxis,
                    comnorm,
                    "o",
                    color=self.colors[i_p],
                    marker=self.markers[i_p],
                )

        plt.xlabel(r"$\nu$")
        plt.yscale("log")
        plt.xscale("log")
        plt.ylim(y_lims)
        plt.xlim(x_lims)
        ax.invert_yaxis()
        plt.title(r"(d) Commutator norms, VQE parameters", y=y_add_caption)

        handles, labels = ax1.get_legend_handles_labels()
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.15)
        fig.legend(handles, labels, loc=(0.20, -0.005), ncol=6)
        if save:
            settings_str = self.return_settings_str(settings)
            fig.savefig(f"figs/l1--comnorms--{settings_str}.pdf", bbox_inches="tight")
        plt.show()

    def largeplot(self, settings: Dict, save: bool = False, add_fit: bool = True):
        fig = plt.figure(figsize=(14, 10), constrained_layout=True)

        y_add_caption = -0.3
        y_lims = [10 ** 0, 10 ** (-3)]
        x_lims = [10 ** 1, 10 ** (4)]

        ### L1 - random
        settings.update({"vqe": False})
        data = self.load_data(settings)
        l1s = np.array(data["l1s"])
        xaxis = np.array(data["n_gates"])
        xaxis_extended = np.append(
            np.geomspace(10 ** (1), xaxis[0]),
            np.append(xaxis[1:-1], np.geomspace(xaxis[-1], 10 ** 4)),
        )
        dists_mean = np.nanmean(l1s / 2, axis=0).T
        p_noises_str = data["p_noises_str"]
        self.save_plot_data(
            data={"x": xaxis, "y": dists_mean, "labels": p_noises_str},
            settings=settings,
            savestr="l1s",
        )
        ax1 = fig.add_subplot(221)
        for i_p, p_str in enumerate(p_noises_str):
            l1 = dists_mean[i_p]
            self.eps = float(p_str)
            if self.eps < 1e-3:
                popt, _ = curve_fit(
                    self.alpha_func, xaxis[self.fit_from_n :], l1[self.fit_from_n :]
                )
                fit = self.alpha_func(xaxis_extended, popt[0], popt[1])
                ax1.plot(xaxis_extended, fit, "--", color=self.colors[i_p])

            if self.labels[i_p] is not None:
                ax1.plot(
                    xaxis,
                    l1,
                    "o",
                    color=self.colors[i_p],
                    marker=self.markers[i_p],
                    label=rf"${self.labels[i_p]}$",
                )
            else:
                ax1.plot(
                    xaxis, l1, "o", color=self.colors[i_p], marker=self.markers[i_p]
                )

        if settings["add_scrambling"]:
            ax1.legend(bbox_to_anchor=(1.1, 0.95))
        plt.xlabel(r"$\nu$")
        plt.ylabel(r"$W$ (eigenvalue uniformity)")
        plt.yscale("log")
        plt.xscale("log")
        if settings["hamiltonian"] in ["TFI-u", "XXZ-u", "TFI-uni", "XXZ-uni"]:
            plt.ylim(y_lims)
            plt.xlim(x_lims)
            ax1.invert_yaxis()
            if settings["add_scrambling"]:
                self.plot_alpha_N(settings, ax1, fit_y="l1s", add_fit=add_fit)

        if not settings["add_scrambling"]:
            plt.title(r"(a) $L_1$ distance, random parameters", y=y_add_caption)

            # Commutator norm - random
            ax = fig.add_subplot(223)
            fidelities = np.array(data["fidelities"])
            commutator_norms = np.array(data["commutator_norms"]) / (1 - fidelities)
            comnorm_mean = np.nanmean(commutator_norms, axis=0).T
            self.save_plot_data(
                data={"x": xaxis, "y": dists_mean, "labels": p_noises_str},
                settings=settings,
                savestr="comnorm",
            )
            for i_p, p_str in enumerate(p_noises_str):
                comnorm = comnorm_mean[i_p]
                self.eps = float(p_str)
                if self.eps < 1e-7:
                    continue
                if self.eps < 1e-3:
                    popt, _ = curve_fit(
                        self.alpha_func,
                        xaxis[self.fit_from_n :],
                        comnorm[self.fit_from_n :],
                    )
                    fit = self.alpha_func(xaxis_extended, popt[0], popt[1])
                    ax.plot(xaxis_extended, fit, "--", color=self.colors[i_p])
                    # ,alpha=0.5

                if self.labels[i_p] is not None:
                    ax.plot(
                        xaxis,
                        comnorm,
                        "o",
                        color=self.colors[i_p],
                        marker=self.markers[i_p],
                        label=rf"${self.labels[i_p]}$",
                    )
                else:
                    ax.plot(
                        xaxis,
                        comnorm,
                        "o",
                        color=self.colors[i_p],
                        marker=self.markers[i_p],
                    )

            plt.xlabel(r"$\nu$")
            plt.ylabel(r"$C$ (commutator norm)")
            plt.yscale("log")
            plt.xscale("log")
            plt.title(r"(c) Commutator norms, random parameters", y=y_add_caption)
            if settings["hamiltonian"] in ["TFI-u", "XXZ-u", "TFI-uni", "XXZ-uni"]:
                plt.ylim(y_lims)
                plt.xlim(x_lims)
                ax.invert_yaxis()
                # self.plot_alpha_N(settings,ax,fit_y="commutator_norms",add_fit=add_fit)

            # L1 - VQE
            settings.update({"vqe": True})
            data = self.load_data(settings)
            l1s = np.array(data["l1s"])
            xaxis = np.array(data["n_gates"])
            xaxis_extended = np.append(
                np.geomspace(10 ** (1), xaxis[0]),
                np.append(xaxis[1:-1], np.geomspace(xaxis[-1], 10 ** 4)),
            )
            dists_mean = np.nanmean(l1s / 2, axis=0).T
            p_noises_str = data["p_noises_str"]
            self.save_plot_data(
                data={"x": xaxis, "y": dists_mean, "labels": p_noises_str},
                settings=settings,
                savestr="l1s",
            )
            ax = fig.add_subplot(222)
            for i_p, p_str in enumerate(p_noises_str):
                l1 = dists_mean[i_p]
                self.eps = float(p_str)
                if self.eps < 1e-3:
                    popt, _ = curve_fit(
                        self.alpha_func, xaxis[self.fit_from_n :], l1[self.fit_from_n :]
                    )
                    fit = self.alpha_func(xaxis_extended, popt[0], popt[1])
                    ax.plot(xaxis_extended, fit, "--", color=self.colors[i_p])

                if self.labels[i_p] is not None:
                    ax.plot(
                        xaxis,
                        l1,
                        "o",
                        color=self.colors[i_p],
                        marker=self.markers[i_p],
                        label=rf"${self.labels[i_p]}$",
                    )
                else:
                    ax.plot(
                        xaxis, l1, "o", color=self.colors[i_p], marker=self.markers[i_p]
                    )

            plt.xlabel(r"$\nu$")
            # plt.ylabel(r"$W$ (eigenvalue uniformity)");
            plt.yscale("log")
            plt.xscale("log")
            if settings["hamiltonian"] in ["TFI-u", "XXZ-u", "TFI-uni", "XXZ-uni"]:
                plt.ylim(y_lims)
                plt.xlim(x_lims)
                ax.invert_yaxis()
            plt.title(r"(b) $L_1$ distance, VQE parameters", y=y_add_caption)

            # Commutator norm - VQE
            ax = fig.add_subplot(224)
            fidelities = np.array(data["fidelities"])
            commutator_norms = np.array(data["commutator_norms"]) / (1 - fidelities)
            comnorm_mean = np.nanmean(commutator_norms, axis=0).T
            self.save_plot_data(
                data={"x": xaxis, "y": dists_mean, "labels": p_noises_str},
                settings=settings,
                savestr="comnorm",
            )
            for i_p, p_str in enumerate(p_noises_str):
                comnorm = comnorm_mean[i_p]
                self.eps = float(p_str)
                if self.eps < 1e-7:
                    continue
                # if self.eps < 1e-3:
                #   popt, _ = curve_fit(self.alpha_func, xaxis[ self.fit_from_n:], comnorm[ self.fit_from_n:])
                #   fit = self.alpha_func(xaxis_extended,popt[0],popt[1])
                #   ax.plot(xaxis_extended,fit,color=self.colors[i_p]);#,alpha=0.5

                if self.labels[i_p] is not None:
                    ax.plot(
                        xaxis,
                        comnorm,
                        "o",
                        color=self.colors[i_p],
                        marker=self.markers[i_p],
                        label=rf"${self.labels[i_p]}$",
                    )
                else:
                    ax.plot(
                        xaxis,
                        comnorm,
                        "o",
                        color=self.colors[i_p],
                        marker=self.markers[i_p],
                    )

            plt.xlabel(r"$\nu$")
            # plt.ylabel(r"$C$ (commutator norm)");
            plt.yscale("log")
            plt.xscale("log")
            if settings["hamiltonian"] in ["TFI-u", "XXZ-u", "TFI-uni", "XXZ-uni"]:
                plt.ylim(y_lims)
                plt.xlim(x_lims)
                ax.invert_yaxis()
            plt.title(r"(d) Commutator norms, VQE parameters", y=y_add_caption)

            handles, labels = ax1.get_legend_handles_labels()
            plt.tight_layout()
            fig.subplots_adjust(bottom=0.15)
            fig.legend(handles, labels, loc=(0.20, -0.005), ncol=6)
        if save:
            settings_str = self.return_settings_str(settings)
            fig.savefig(f"figs/l1--comnorms--{settings_str}.pdf", bbox_inches="tight")
        plt.show()

    def plot_alpha_N(
        self,
        settings: Dict,
        axes: object = None,
        fit_y: str = "l1s",
        add_fit: bool = True,
        return_data: bool = False,
    ):
        sub_settings = settings.copy()
        sub_settings.pop("n_qubits", None)

        alphas = {}
        for folder in self.results:
            with open(f"{self.path}{folder}/parameters.json", "r") as file:
                parameters = json.load(file)

            if not sub_settings.items() <= parameters.items():
                continue

            with open(f"{self.path}{folder}/results.json", "r") as file:
                data = json.load(file)

            y = np.array(data[fit_y])
            if fit_y == "commutator_norms":
                fidelities = np.array(data["fidelities"])
                commutator_norms = y / (1 - fidelities)
                dists_mean = np.nanmean(commutator_norms, axis=0).T
            else:
                dists_mean = np.nanmean(y / 2, axis=0).T

            xaxis = np.array(data["n_gates"])
            noises = data["p_noises_str"]
            data = dists_mean[noises.index("1e-8")]
            self.eps = 1e-8
            popt, _ = curve_fit(
                self.alpha_func, xaxis[self.fit_from_n :], data[self.fit_from_n :]
            )
            alpha_fit = popt[0]
            alphas.update({parameters["n_qubits"]: alpha_fit})

        xaxis = np.fromiter(alphas.keys(), dtype=np.float_)
        yaxis = np.fromiter(alphas.values(), dtype=np.float_)
        idx_sort = np.argsort(xaxis)
        xaxis = xaxis[idx_sort]
        yaxis = yaxis[idx_sort]
        popt, _ = curve_fit(self.a_b_func, xaxis, yaxis)
        a_b_fit = self.a_b_func(xaxis, popt[0], popt[1])
        print("a", popt[0], "b", popt[1])
        if return_data:
            return xaxis, yaxis, a_b_fit

        if axes is not None:
            ax = inset_axes(
                axes,
                width="45%",
                height="25%",
                loc="lower left",
                bbox_to_anchor=(1.1, 0.1, 1, 1),
                bbox_transform=axes.transAxes,
            )
        else:
            fig, ax = plt.subplots(1, 1)

        ax.plot(xaxis, yaxis, "s", color="blue")
        if add_fit:
            ax.plot(xaxis, a_b_fit, color="blue")
        ax.set_ylabel(r"$\alpha$")
        ax.set_xlabel(r"$N$")
        ticks = [""] * len(xaxis)
        ticks[0] = int(xaxis[0])
        ticks[-1] = int(xaxis[-1])
        ax.set_xticks(xaxis, ticks)

    def large_alpha_N(self, n_qubits: int, save: bool = False, add_fit: bool = True):
        fig = plt.figure(figsize=(14, 10), constrained_layout=True)

        y_add_caption = -0.42

        ax = plt.subplot(321)
        settings = {
            "ansatz": "HVA",
            "hamiltonian": "XXZ-u",
            "n_qubits": n_qubits,
            "add_scrambling": False,
            "vqe": False,
        }
        xaxis, yaxis, a_b_fit = self.plot_alpha_N(
            settings, fit_y="l1s", add_fit=add_fit, return_data=True
        )
        ax.plot(xaxis, yaxis, "s", color="blue")
        if add_fit:
            ax.plot(xaxis, a_b_fit, color="blue")
        ax.set_ylabel(r"$\alpha$")
        ax.set_xlabel(r"$N$")
        ticks = [""] * len(xaxis)
        ticks[0] = int(xaxis[0])
        ticks[-1] = int(xaxis[-1])
        ax.set_xticks(xaxis, ticks)
        plt.title(r"(a) HVA-XXX, random", y=y_add_caption)

        ax = plt.subplot(322)
        settings = {
            "ansatz": "HVA",
            "hamiltonian": "XXZ-u",
            "n_qubits": n_qubits,
            "add_scrambling": False,
            "vqe": True,
        }
        xaxis, yaxis, a_b_fit = self.plot_alpha_N(
            settings, fit_y="l1s", add_fit=add_fit, return_data=True
        )
        ax.plot(xaxis, yaxis, "s", color="blue")
        if add_fit:
            ax.plot(xaxis, a_b_fit, color="blue")
        ax.set_ylabel(r"$\alpha$")
        ax.set_xlabel(r"$N$")
        ticks = [""] * len(xaxis)
        ticks[0] = int(xaxis[0])
        ticks[-1] = int(xaxis[-1])
        ax.set_xticks(xaxis, ticks)
        plt.title(r"(b) HVA-XXX, VQE", y=y_add_caption)

        ax = plt.subplot(323)
        settings = {
            "ansatz": "HVA",
            "hamiltonian": "TFI-u",
            "n_qubits": n_qubits,
            "add_scrambling": False,
            "vqe": False,
        }
        xaxis, yaxis, a_b_fit = self.plot_alpha_N(
            settings, fit_y="l1s", add_fit=add_fit, return_data=True
        )
        ax.plot(xaxis, yaxis, "s", color="blue")
        if add_fit:
            ax.plot(xaxis, a_b_fit, color="blue")
        ax.set_ylabel(r"$\alpha$")
        ax.set_xlabel(r"$N$")
        ticks = [""] * len(xaxis)
        ticks[0] = int(xaxis[0])
        ticks[-1] = int(xaxis[-1])
        ax.set_xticks(xaxis, ticks)
        plt.title(r"(c) HVA-TFI, random", y=y_add_caption)

        ax = plt.subplot(324)
        settings = {
            "ansatz": "HVA",
            "hamiltonian": "TFI-u",
            "n_qubits": n_qubits,
            "add_scrambling": False,
            "vqe": True,
        }
        xaxis, yaxis, a_b_fit = self.plot_alpha_N(
            settings, fit_y="l1s", add_fit=add_fit, return_data=True
        )
        ax.plot(xaxis, yaxis, "s", color="blue")
        if add_fit:
            ax.plot(xaxis, a_b_fit, color="blue")
        ax.set_ylabel(r"$\alpha$")
        ax.set_xlabel(r"$N$")
        ticks = [""] * len(xaxis)
        ticks[0] = int(xaxis[0])
        ticks[-1] = int(xaxis[-1])
        ax.set_xticks(xaxis, ticks)
        plt.title(r"(d) HVA-TFI, VQE", y=y_add_caption)

        ax = plt.subplot(325)
        settings = {
            "ansatz": "SEL",
            "n_qubits": n_qubits,
            "hamiltonian": "TFI-u",
            "vqe": False,
        }
        xaxis, yaxis, a_b_fit = self.plot_alpha_N(
            settings, fit_y="l1s", add_fit=add_fit, return_data=True
        )
        ax.plot(xaxis, yaxis, "s", color="blue")
        if add_fit:
            ax.plot(xaxis, a_b_fit, color="blue")
        ax.set_ylabel(r"$\alpha$")
        ax.set_xlabel(r"$N$")
        ticks = [""] * len(xaxis)
        ticks[0] = int(xaxis[0])
        ticks[-1] = int(xaxis[-1])
        ax.set_xticks(xaxis, ticks)
        plt.title(r"(e) SEL-TFI, random", y=y_add_caption)

        if save:
            fig.savefig(f"figs/alpha--n.pdf", bbox_inches="tight")
        plt.show()

