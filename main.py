from simulations.experiment import Experiment
from simulations.imports import *
from simulations.parameters import Parameters


def run():
    start = time.time()
    try:
        args = sys.argv[-1].split("|")
    except:
        args = []
    print("------------------------------------")
    print("RUNNING EXPERIMENT...")
    kwargs = {}
    parameters_temp = Parameters(mkdir=False)
    if "=" in args[0]:
        for arg in args:
            var = arg.split("=")[0]
            val = arg.split("=")[1]
            par_val = getattr(parameters_temp, var)

            if isinstance(par_val, bool):
                val = val.lower() == "true"
            elif isinstance(par_val, int):
                val = int(val)
            elif isinstance(par_val, float):
                val = float(val)
            elif isinstance(par_val, str):
                pass
            else:
                var = None
                print("COULD NOT FIND VARIABLE:", var)
            kwargs.update({var: val})

    parameters = Parameters(kwargs)
    print("Running with:", parameters)
    experiment = Experiment(parameters)
    experiment.run(save=True)
    time_it_took = time.time() - start
    print(f"FINISHED EXPERIMENT IN {time_it_took} SECONDS")
    print("------------------------------------")


if __name__ == "__main__":
    run()
