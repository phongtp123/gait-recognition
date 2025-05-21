import numpy as np
import torch
import pandas
import time
import sys
import utils
import csv

def save_embeddings_to_csv(embeddings, save_path):
    """
    Lưu embeddings vào file CSV.
    """
    with open(save_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Ghi header động theo độ dài sequence
        max_len = max(len(seq) for seq in embeddings.keys())
        header = [f"Field_{i}" for i in range(max_len)] + ["Embedding"]
        csv_writer.writerow(header)

        for sequence, emb in embeddings.items():
            row = list(sequence) + [' '.join([f"{x:.6f}" for x in emb])]
            csv_writer.writerow(row)

    print(f"\n✅ Embeddings saved to {save_path}")

def _evaluate_casia_b(embeddings):
    """
    Test dataset consists of sequences of last 50 ids from CASIA B Dataset.
    Data is divided in the following way:
    Gallery Set:
        NM 1, NM 2, NM 3, NM 4
    Probe Set:
        Subset 1:
            NM 5, NM 6
         Subset 2:
            BG 1, BG 2
         Subset 3:
            CL 1, CL 2
    """

    gallery = {k: v for (k, v) in embeddings.items() if k[1] == 0 and k[2] <= 4}
    gallery_per_angle = {}
    for angle in range(0, 181, 18):
        gallery_per_angle[angle] = {k: v for (k, v) in gallery.items() if k[3] == angle}

    probe_nm = {k: v for (k, v) in embeddings.items() if k[1] == 0 and k[2] >= 5}
    probe_bg = {k: v for (k, v) in embeddings.items() if k[1] == 1}
    probe_cl = {k: v for (k, v) in embeddings.items() if k[1] == 2}

    correct = np.zeros((3, 11, 11))
    total = np.zeros((3, 11, 11))
    for gallery_angle in range(0, 181, 18):
        gallery_embeddings = np.array(list(gallery_per_angle[gallery_angle].values()))
        gallery_targets = list(gallery_per_angle[gallery_angle].keys())
        gallery_pos = int(gallery_angle / 18)

        probe_num = 0
        for probe in [probe_nm, probe_bg, probe_cl]:
            for (target, embedding) in probe.items():
                subject_id, _, _, probe_angle = target
                probe_pos = int(probe_angle / 18)

                distance = np.linalg.norm(gallery_embeddings - embedding, ord=2, axis=1)
                min_pos = np.argmin(distance)
                min_target = gallery_targets[int(min_pos)]

                if min_target[0] == subject_id:
                    correct[probe_num, gallery_pos, probe_pos] += 1
                total[probe_num, gallery_pos, probe_pos] += 1

            probe_num += 1

    accuracy = correct / total

    # Exclude same view
    for i in range(3):
        accuracy[i] -= np.diag(np.diag(accuracy[i]))

    accuracy_flat = np.sum(accuracy, 1) / 10

    header = ["NM#5-6", "BG#1-2", "CL#1-2"]

    accuracy_avg = np.mean(accuracy)
    sub_accuracies_avg = np.mean(accuracy_flat, 1)
    sub_accuracies = dict(zip(header, list(sub_accuracies_avg)))

    dataframe = pandas.DataFrame(
        np.concatenate((accuracy_flat, sub_accuracies_avg[..., np.newaxis]), 1),
        header,
        list(range(0, 181, 18)) + ["mean"],
    )

    return correct, accuracy_avg, sub_accuracies, dataframe


def evaluate(data_loader, model, evaluation_fn, use_flip=False, save_csv_path=None):
    model.eval()
    batch_time = utils.AverageMeter()
    all_features = []

    # Calculate embeddings
    with torch.no_grad():
        end = time.time()
        embeddings = dict()
        for idx, (points, target) in enumerate(data_loader):
            if use_flip:
                bsz = points.shape[0]
                data_flipped = torch.flip(points, dims=[1])
                points = torch.cat([points, data_flipped], dim=0)

            output = model(points)  

            if use_flip:
                f1, f2 = torch.split(output, [bsz, bsz], dim=0)
                output = torch.mean(torch.stack([f1, f2]), dim=0)

            for i in range(output.shape[0]):
                sequence = tuple(
                    int(t[i]) if type(t[i]) is torch.Tensor else t[i] for t in target
                )
                embeddings[sequence] = output[i].cpu().numpy()

            batch_time.update(time.time() - end)
            end = time.time()

            output_str = output.cpu().numpy().tolist()[:3]  # Hiển thị 3 phần tử đầu tiên

            output_first = output_str[0] if output_str else None

            if isinstance(output, torch.Tensor):
                output_shape = list(output.shape)
            else:
                output_shape = [len(output_str), len(output_str[0])] if output_str else [0]

            print(
                #f"Feature: {output_first}\t"
                f"Feature Shape: {output_shape}\t"
                f"Test: [{idx}/{len(data_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
            )
            sys.stdout.flush()

    if save_csv_path:
        save_embeddings_to_csv(embeddings, save_csv_path)

    return evaluation_fn(embeddings)
