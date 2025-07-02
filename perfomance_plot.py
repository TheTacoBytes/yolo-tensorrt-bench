import argparse
import pandas as pd
import matplotlib.pyplot as plt

def find_latency_col(df):
    for col in ['ms', 'inference_ms', 'time_ms']:
        if col in df.columns:
            return col
    raise ValueError(f"No known latency column in {df.columns}")

def plot_time_series(csv_paths, labels, skip, log_scale, ylim, output):
    plt.figure()
    for path, label in zip(csv_paths, labels):
        df = pd.read_csv(path)
        lat_col = find_latency_col(df)
        x = df['frame'][skip:]
        y = df[lat_col][skip:]
        plt.plot(x, y, label=label)
    plt.xlabel('Frame')
    plt.ylabel('Inference Time (ms)')
    plt.title('Per-Frame Inference Time Comparison')
    if log_scale:
        plt.yscale('log')
    if ylim:
        plt.ylim(0, ylim)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved time-series plot to {output}")

def plot_box(csv_paths, labels, skip, output):
    data = {}
    for path, label in zip(csv_paths, labels):
        df = pd.read_csv(path)
        lat_col = find_latency_col(df)
        data[label] = df[lat_col][skip:].values
    fig, ax = plt.subplots()
    pd.DataFrame(data).plot.box(ax=ax)
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title('Latency Distribution by Method')
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved box-plot to {output}")

def main():
    parser = argparse.ArgumentParser(
        description="Compare inference performance across CSV logs"
    )
    parser.add_argument(
        'csv_files', nargs='+',
        help="CSV files with columns 'frame' and latency ('ms' or 'inference_ms')"
    )
    parser.add_argument(
        '--labels', nargs='+', required=True,
        help="One label per CSV, in the same order"
    )
    parser.add_argument(
        '--mode', choices=['line','box'], default='line',
        help="Plot type: 'line' (time series) or 'box' (distribution)"
    )
    parser.add_argument(
        '--skip', type=int, default=0,
        help="Skip the first N frames (warm-up)"
    )
    parser.add_argument(
        '--log', action='store_true',
        help="Use log scale on the y-axis (only in line mode)"
    )
    parser.add_argument(
        '--ylim', type=float, default=None,
        help="Clip y-axis to [0, ylim] (only in line mode)"
    )
    parser.add_argument(
        '--output', default='performance_plot.png',
        help="Output image filename"
    )
    args = parser.parse_args()

    if len(args.csv_files) != len(args.labels):
        raise ValueError("Number of CSV files must match number of labels")

    if args.mode == 'line':
        plot_time_series(
            args.csv_files, args.labels,
            skip=args.skip,
            log_scale=args.log,
            ylim=args.ylim,
            output=args.output
        )
    else:
        plot_box(
            args.csv_files, args.labels,
            skip=args.skip,
            output=args.output
        )

if __name__ == "__main__":
    main()
