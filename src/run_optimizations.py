import argparse
import os
import shutil

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, default=1, choices=range(1, 7), nargs="*")
parser.add_argument("--all", action="store_true")
args = parser.parse_args()
if args.all:
    args.n = list(range(1, 7))
for i in args.n:
    result_dir = os.path.join(output_dir, f"optimization{i}")
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)
opt_options = [
    "--bicycle-only --steer-with PEDAL_STEER_TORQUE --bicycle-parametrization Browser",
    "--bicycle-only --steer-with PEDAL_STEER_TORQUE --bicycle-parametrization Pista",
    "--bicycle-only --steer-with PEDAL_STEER_TORQUE --bicycle-parametrization Fisher",
    "--bicycle-only --steer-with PEDAL_STEER_TORQUE --bicycle-parametrization Fisher"
    " --front-frame-suspension",
    "--model-upper-body --steer-with PEDAL_STEER_TORQUE --shoulder-type FLEX_ROT"
    " --bicycle-parametrization Browser --rider-parametrization Jason",
    "--model-upper-body --steer-with HUMAN_TORQUE --shoulder-type FLEX_ROT"
    " --bicycle-parametrization Browser --rider-parametrization Jason",
]
for opt_n in args.n:
    result_dir = os.path.join(output_dir, f"optimization{opt_n}")
    os.system(f"python ./src/run.py {opt_options[opt_n - 1]} --output {result_dir}")
