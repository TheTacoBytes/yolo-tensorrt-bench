#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def find_latency_col(df):
    for c in ('ms', 'inference_ms', 'time_ms'):
        if c in df.columns:
            return c
    raise ValueError(f"No known latency column in {df.columns}")

def print_stats(data, label):
    arr = np.array(data)
    print(f"{label}:  min={arr.min():.3f}  median={np.median(arr):.3f}  mean={arr.mean():.3f}  max={arr.max():.3f}")

def plot_time_series(csv_paths, labels, skip, log_scale, ylim, output):
    plt.figure(figsize=(8,4))
    for path, label in zip(csv_paths, labels):
        df = pd.read_csv(path)
        lat = find_latency_col(df)
        x = df['frame'][skip:]
        y = df[lat][skip:]
        print_stats(y, label)
        plt.plot(x, y, label=label, linewidth=1)
    plt.xlabel('Frame')
    plt.ylabel('Inference Time (ms)')
    plt.title('Per-Frame Inference Time')
    if log_scale:
        plt.yscale('log')
    if ylim:
        plt.ylim(0, ylim)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    print(f"Time-series plot saved to {output}")

def plot_box(csv_paths, labels, skip, output):
    data = {}
    for path, label in zip(csv_paths, labels):
        df = pd.read_csv(path)
        lat = find_latency_col(df)
        arr = df[lat][skip:].values
        print_stats(arr, label)
        data[label] = arr
    df_all = pd.DataFrame(data)
    plt.figure(figsize=(6,4))
    df_all.boxplot(rot=45)
    plt.ylabel('Inference Time (ms)')
    plt.title('Latency Distribution')
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    print(f"Box-plot saved to {output}")

def main():
    p = argparse.ArgumentParser(
        description="Compare inference performance across multiple CSV logs")
    p.add_argument('csv_files', nargs='+',
                   help="Paths to CSVs (must have 'frame' + latency column)")
    p.add_argument('--labels', nargs='+', required=True,
                   help="Labels for each CSV, in order")
    p.add_argument('--mode', choices=('line','box'), default='line',
                   help="Plot type: time-series ('line') or distribution ('box')")
    p.add_argument('--skip', type=int, default=0,
                   help="Skip first N frames (warm-up)")
    p.add_argument('--log', action='store_true',
                   help="Log scale Y axis (only in line mode)")
    p.add_argument('--ylim', type=float, default=None,
                   help="Max Y value (only in line mode)")
    p.add_argument('--output', default='performance_plot.png',
                   help="Output image file")
    args = p.parse_args()

    if len(args.csv_files) != len(args.labels):
        p.error("You must supply exactly one label per CSV file.")

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

if __name__=='__main__':
    main()
