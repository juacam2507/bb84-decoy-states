import numpy as np
import argparse
from plotter import Plotter


def main():
    plotter = Plotter()
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", help="Path to the input file")
    parser.add_argument(
        "--type", choices=["key_rate", "yields"], help="Type of plot to generate"
    )

    args = parser.parse_args()

    print(f"Generating {args.type} plot from file {args.file}...")

    if args.type == "key_rate":
        plotter.R_vs_D_plot(args.file)

    elif args.type == "yields":
        plotter.yield_plot(args.file)

    else:
        raise ValueError("Invalid plot type. Available options: (key_rate/yields)")


if __name__ == "__main__":
    main()
