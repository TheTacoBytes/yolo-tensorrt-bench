from ultralytics import YOLO
import cv2, time, csv
from pathlib import Path

def main(source: Path, output: Path, device: str = 'cuda'):
    # --- 1. Load & export to TensorRT engine if needed ---
    seg_weights = 'yolo11n-seg.pt'
    engine_file = Path(seg_weights).stem + '.engine'
    model = YOLO(seg_weights)                      # load segmentation weights
    model.fuse()
    # Export engine if it doesn’t exist
    if not Path(engine_file).exists():
        print(f"Exporting TensorRT engine to {engine_file}…")
        model.export(format='engine')              # yolo11n-seg.engine

    # --- 2. Load the TensorRT engine for inference ---
    trt_model = YOLO(engine_file)                  # load engine (no .to() here!)
    trt_model.overrides['conf'] = 0.75             # 75% confidence

    # --- 3. Open video & prepare writer ---
    cap = cv2.VideoCapture(str(source))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (w, h)
    )

    # --- 4. Inference loop & timing ---
    timings, frame_idx = [], 0
    while True:
        ret, img_bgr = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Ultralytics expects RGB numpy array
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        t0 = time.time()
        # pass device here; specify segmentation task
        results = trt_model.predict(img, device=device, imgsz=640, task='segment')[0]
        elapsed = (time.time() - t0) * 1000
        timings.append(elapsed)

        # draw masks & boxes
        annotated = results.plot()
        out = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        writer.write(out)

        cv2.imshow('YOLO11-SEG (TensorRT)', out)
        if cv2.waitKey(1) == 27:
            break  # ESC

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # --- 5. Summary & save metrics ---
    avg = sum(timings)/len(timings) if timings else 0
    print(f'Frames: {frame_idx}, Avg inference: {avg:.1f} ms')

    metrics = output.parent / 'metrics_yolo11_trt.csv'
    with open(metrics, 'w', newline='') as f:
        wcsv = csv.writer(f)
        wcsv.writerow(['frame','inference_ms'])
        for i, t in enumerate(timings, start=1):
            wcsv.writerow([i, f'{t:.2f}'])
        wcsv.writerow([]); wcsv.writerow(['avg', f'{avg:.2f}'])
    print(f'Saved timings to {metrics}')

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(
        description="Run YOLO11-seg with TensorRT and save performance metrics"
    )
    p.add_argument('--source', type=Path, default=Path('./videos/video.mp4'))
    p.add_argument('--output', type=Path, default=Path('results/y11_trt_out.mp4'))
    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()
    main(args.source, args.output, args.device)
