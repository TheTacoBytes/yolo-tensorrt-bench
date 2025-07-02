from ultralytics import YOLO
import cv2, time, csv
from pathlib import Path

def main(source, output, device='cuda'):
    # Load YOLO 11 nano instance‐segmentation
    model = YOLO('yolo11n-seg.pt')  
    model.fuse()                       # fuse for speed
    model.to(device)
    model.overrides['conf'] = 0.75     # 75% confidence threshold

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

    timings, frame_idx = [], 0
    while True:
        ret, img_bgr = cap.read()
        if not ret: break
        frame_idx += 1

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        t0 = time.time()
        # run in segmentation mode
        results = model.predict(img, imgsz=640, task='segment')[0]
        dt = (time.time() - t0) * 1000
        timings.append(dt)

        # results.orig_img has boxes + masks drawn
        # out = cv2.cvtColor(results.orig_img, cv2.COLOR_RGB2BGR)
        results = model.predict(img, imgsz=640, task='segment')[0]
        # draw boxes + masks onto a copy
        annotated = results.plot()  
        out       = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        writer.write(out)
        cv2.imshow('YOLO11 Segmentation', out)
        if cv2.waitKey(1) == 27:
            break  # ESC

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    avg = sum(timings)/len(timings) if timings else 0
    print(f'Frames: {frame_idx}, Avg inference: {avg:.1f} ms')

    # save per‐frame timings
    metrics = output.parent / 'metrics_yolo11_seg.csv'
    with open(metrics, 'w', newline='') as f:
        wcsv = csv.writer(f)
        wcsv.writerow(['frame','ms'])
        for i, t in enumerate(timings, 1):
            wcsv.writerow([i, f'{t:.2f}'])
        wcsv.writerow([]); wcsv.writerow(['avg', f'{avg:.2f}'])
    print(f'Saved timings to {metrics}')

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=Path('./videos/video.mp4'))
    p.add_argument('--output', type=Path, default=Path('results/y11_seg_out.mp4'))
    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()
    main(args.source, args.output, args.device)
