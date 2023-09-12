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
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    else:
        shutil.rmtree(result_dir)
if 1 in args.n:
    result_dir = os.path.join(output_dir, "optimization1")
    os.system(f"python -m src.run --bicycle-only --steer-with PEDAL_STEER_TORQUE "
              f"--bicycle-parametrization Browser --output {result_dir}")
if 2 in args.n:
    result_dir = os.path.join(output_dir, "optimization2")
    os.system(f"python -m src.run --bicycle-only --steer-with PEDAL_STEER_TORQUE "
              f"--bicycle-parametrization Pista --output {result_dir}")
if 3 in args.n:
    result_dir = os.path.join(output_dir, "optimization3")
    os.system(f"python -m src.run --bicycle-only --steer-with PEDAL_STEER_TORQUE "
              f"--bicycle-parametrization Fisher --output {result_dir}")
if 4 in args.n:
    result_dir = os.path.join(output_dir, "optimization4")
    os.system(f"python -m src.run --bicycle-only --front-frame-suspension "
              f"--steer-with PEDAL_STEER_TORQUE --bicycle-parametrization Fisher "
              f"--output {result_dir}")
if 5 in args.n:
    result_dir = os.path.join(output_dir, "optimization5")
    os.system(f"python -m src.run --model-upper-body --shoulder-type FLEX_ROT "
              f"--steer-with PEDAL_STEER_TORQUE --bicycle-parametrization Browser "
              f"--rider-parametrization Jason --output {result_dir}")
if 6 in args.n:
    result_dir = os.path.join(output_dir, "optimization6")
    os.system(f"python -m src.run --model-upper-body --shoulder-type FLEX_ROT "
              f"--steer-with HUMAN_TORQUE --bicycle-parametrization Browser "
              f"--rider-parametrization Jason --output {result_dir}")
