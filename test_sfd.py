"""
    Author: Nan Wu
    Time: 10/28/2023
    Base Dataset and Generalized Functions for Sea Fog Dataset
    inputs are: Pred: [b, num_classes, H, W]; Target: [b, 1, H, W]
    We should record the following information:
        for SeaFogDataset, we use pred[:, 1].unsqueeze(1) to get the binary prediction
            - ACC (binary ACC)
            - F1 score (binary F1 score)
            - CSI (binary Jaccard index)
            - Kappa (binary Cohen's kappa)
            - Precision (binary precision)
            - Recall (binary recall)
            - Confusion Matrix
            - Precision-Recall Curve
            - Dice score

        for YBSFDataset, we use multiclass prediction [num_classes, H, W] and
        ** must ignore ** background class 0 to get the multiclass prediction
            - ACC (multiclass ACC)
            - F1 score (multiclass F1 score)
            - CSI (multiclass Jaccard index)
            - Kappa (multiclass Cohen's kappa)
            - Precision (multiclass precision)
            - Recall (multiclass recall)
            - Confusion Matrix
            - Precision-Recall Curve
            - Dice score

"""
import json
import os
import time

import torch
import pandas as pd
from torch import nn
from torchmetrics.classification import (BinaryAccuracy, BinaryPrecision, BinaryRecall,
                                         BinaryF1Score, BinaryConfusionMatrix,
                                         BinaryCohenKappa, BinaryPrecisionRecallCurve)
from torchmetrics.classification import BinaryJaccardIndex as BinaryCSI

from torchmetrics.classification import (MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score,
                                         MulticlassConfusionMatrix, MulticlassCohenKappa,
                                         MulticlassPrecisionRecallCurve)
from torchmetrics.classification import MulticlassJaccardIndex as MulticlassCSI
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def compute_binary_seg_metrics(preds, targets, threshold=0.5):
    """
    :param preds: [b, num_classes, H, W]
    :param targets: [b, num_classes, H, W]
    :return: prec, recall, acc, f1, csi, kappa, conf
    """
    if preds.shape[1] > 2:
        raise ValueError("Binary segmentation metrics only support binary prediction")
    # elif preds.shape[1] == 2:
    #     preds, targets = preds[:, 1].unsqueeze(1), targets[:, 1].unsqueeze(1)
    preds, targets = preds[:, 1].unsqueeze(1), targets[:, 1].unsqueeze(1)
    csi = BinaryCSI(threshold=threshold)(preds, targets).item()
    acc = BinaryAccuracy(multidim_average='global', threshold=threshold)(preds, targets).item()
    recall = BinaryRecall(multidim_average='global', threshold=threshold)(preds, targets).item()
    prec = BinaryPrecision(multidim_average='global', threshold=threshold)(preds, targets).item()
    conf = BinaryConfusionMatrix(threshold=threshold)(preds, targets).tolist()
    kappa = BinaryCohenKappa(threshold=threshold)(preds, targets).item()
    f1 = BinaryF1Score(multidim_average='global', threshold=threshold)(preds, targets).tolist()
    pr_curve = BinaryPrecisionRecallCurve(thresholds=torch.linspace(0, 1, 100))(preds, targets.long())
    pr_curve = [tensor_elem for tensor_elem in pr_curve]
    metric_dict = {'acc': acc, 'prec': prec, 'recall': recall, 'f1': f1, 'csi': csi, 'kappa': kappa, 'conf': conf,
                   'pr_curve': pr_curve}
    return metric_dict


def compute_multiclass_seg_metrics(preds, targets, num_classes=4, ignore_index=0):
    """
    :param preds: [b, num_classes, H, W]
    :param targets: [b, num_classes, H, W]
    :return: prec, recall, acc, f1, csi, kappa, conf
    """
    if preds.shape[1] != num_classes:
        raise ValueError("Multiclass segmentation metrics only support multiclass prediction")
    if len(targets.shape) == 4:
        targets = torch.argmax(targets, dim=1)
    acc_overall = MulticlassAccuracy(num_classes=num_classes, average='micro', multidim_average="global",
                                     ignore_index=0)(preds, targets).item()
    acc_classwise = MulticlassAccuracy(num_classes=num_classes, average='none', multidim_average="global",
                                       ignore_index=0)(preds, targets).tolist()
    prec_overall = MulticlassPrecision(num_classes=num_classes, average='micro', multidim_average="global",
                                       ignore_index=0)(preds, targets).item()
    prec_classwise = MulticlassPrecision(num_classes=num_classes, average='none', multidim_average="global",
                                         ignore_index=0)(preds, targets).tolist()
    recall_overall = MulticlassRecall(num_classes=num_classes, average='micro', multidim_average="global",
                                      ignore_index=0)(preds, targets).item()
    recall_classwise = MulticlassRecall(num_classes=num_classes, average='none', multidim_average="global",
                                        ignore_index=0)(preds, targets).tolist()
    f1_overall = MulticlassF1Score(num_classes=num_classes, average='micro', multidim_average="global",
                                   ignore_index=0)(preds, targets).item()
    f1_classwise = MulticlassF1Score(num_classes=num_classes, average='none', multidim_average="global",
                                     ignore_index=0)(preds, targets).tolist()
    csi_overall = MulticlassCSI(num_classes=num_classes, average='micro', ignore_index=0)(preds, targets).item()
    csi_classwise = MulticlassCSI(num_classes=num_classes, average='none', ignore_index=0)(preds, targets).tolist()
    kappa_overall = MulticlassCohenKappa(num_classes=num_classes, ignore_index=0)(preds, targets).item()
    conf_overall = MulticlassConfusionMatrix(num_classes=num_classes, ignore_index=0)(preds, targets).tolist()
    pr_curve_overall = MulticlassPrecisionRecallCurve(num_classes=num_classes, ignore_index=0, thresholds=torch.linspace(0, 1, 100))(preds, targets)
    pr_curve_overall = [tensor_elem for tensor_elem in pr_curve_overall]
    metric_dict = {'acc_overall': acc_overall, 'acc_classwise': acc_classwise,
                   'prec_overall': prec_overall, 'prec_classwise': prec_classwise,
                   'recall_overall': recall_overall, 'recall_classwise': recall_classwise,
                   'f1_overall': f1_overall, 'f1_classwise': f1_classwise,
                   'csi_overall': csi_overall, 'csi_classwise': csi_classwise,
                   'kappa_overall': kappa_overall, 'conf_overall': conf_overall,
                   'pr_curve_overall': pr_curve_overall}
    return metric_dict


def save_to_excel(retrieval_metrics, save_path):
    metrics_df = pd.DataFrame()

    for metric_name, metric_values in retrieval_metrics.items():
        # Remove the '@' and everything after it in the metric name
        metric = metric_name.split("@")[0]

        # If metric is a compound metric, handle it separately
        if isinstance(metric_values, dict):
            for sub_metric, sub_metric_values in metric_values.items():
                # Create a compound metric name
                compound_metric_name = f"{metric}_{sub_metric}"

                # Check if sub_metric_values is a tensor, if yes, move to CPU
                if isinstance(sub_metric_values, torch.Tensor):
                    sub_metric_values = sub_metric_values.cpu()

                # Convert list to string
                sub_metric_values = str(sub_metric_values)

                metrics_df.loc[0, compound_metric_name] = sub_metric_values

        else:
            # Check if metric_values is a tensor, if yes, move to CPU
            if isinstance(metric_values, torch.Tensor):
                metric_values = metric_values.cpu().item()

            # Convert list to string
            # if isinstance(metric_values, list):
            #     metric_values = str(metric_values)

            if isinstance(metric_values, list):
                metric_values = [item.cpu().tolist() if isinstance(item, torch.Tensor) else item for item in
                                 metric_values]
                metric_values = json.dumps(metric_values)



            metrics_df.loc[0, metric] = metric_values

    # Write the DataFrame to an Excel file
    with pd.ExcelWriter(save_path) as writer:
        metrics_df.to_excel(writer, index=False)


def gen_seg_results(dataset_name, mode, inc_choice, test_seg_loader, trained_seg_model, num_classes=4, ignore_index=0):
    trained_seg_model.eval()
    with torch.no_grad():
        query_preds, query_labels = [], []
        pbarss = tqdm(test_seg_loader, total=len(test_seg_loader), leave=True)
        for data in pbarss:
            if mode == 'test':
                test_images, test_labels = data['original_image'][:, inc_choice, :, :].cuda(), data[
                    'original_label'].cuda()
            elif mode == 'test_degrade':
                test_images, test_labels = data['degraded_image'][:, inc_choice, :, :].cuda(), data[
                    'degraded_label'].cuda()
            # start_time = time.time()
            preds = trained_seg_model(test_images)
            # end_time = time.time()
            # print("运行时间: {:.6f} 秒".format(end_time - start_time), "样本量:", preds.shape[0])  # 打印运行时间
            query_preds.append(preds.cpu()), query_labels.append(test_labels.cpu())

        query_preds, query_labels = torch.cat(query_preds), torch.cat(query_labels)
        query_preds = nn.Softmax(dim=1)(query_preds)
        if dataset_name == 'seafog':
            # query_preds, query_labels = query_preds[:, 1].unsqueeze(1), query_labels[:, 1].unsqueeze(1)
            all_metrics = compute_binary_seg_metrics(query_preds, query_labels)
            return all_metrics
        elif dataset_name == 'ybsf':
            if len(query_labels.shape) == 4:
                query_labels = torch.argmax(query_labels, dim=1)
            all_metrics = compute_multiclass_seg_metrics(query_preds, query_labels)
            return all_metrics


def get_final_seg_cls(dataset_name, mode, inc_choice, block_model, input_model_path, output_excel_path, test_loader):
    block_model.load_state_dict(torch.load(input_model_path))
    block_model = block_model.cuda()
    seg_results = gen_seg_results(dataset_name=dataset_name, mode=mode, inc_choice=inc_choice,
                                  test_seg_loader=test_loader,
                                  trained_seg_model=block_model)
    out_path_cls = output_excel_path.replace('.xlsx', '_seg.xlsx')
    save_to_excel(seg_results, out_path_cls)


if __name__ == "__main__":
    pass
    # preds = torch.tensor([[[[0.2, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.2]],
    #                        [[0.1, 0.2, 0.2], [0.2, 0.6, 0.1], [0.2, 0.2, 0.1]],
    #                        [[0.3, 0.2, 0.1], [0.2, 0.1, 0.1], [0.1, 0.2, 0.3]],
    #                        [[0.4, 0.5, 0.6], [0.5, 0.2, 0.7], [0.6, 0.5, 0.4]]]])
    #
    # # Create a tensor for target labels, shape: [b=1, num_classes=4, H=3, W=3]
    # targets = torch.tensor([[[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    #                          [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    #                          [[0, 0, 1], [0, 0, 0], [0, 0, 0]],
    #                          [[0, 1, 0], [1, 0, 1], [1, 1, 0]]]])
    # acc_all, acc_classwise, prec_all, prec_classwise, recall_all, recall_classwise, f1_all, f1_classwise, csi_all, csi_classwise, kappa_all, kappa_classwise, conf_all, conf_classwise = compute_multiclass_seg_metrics(
    #     preds, targets)
    # print('acc_all: ', acc_all, '\n',
    #       'acc_classwise: ', acc_classwise, '\n',
    #       'prec_all: ', prec_all, '\n',
    #       'prec_classwise: ', prec_classwise, '\n',
    #       'recall_all: ', recall_all, '\n',
    #       'recall_classwise: ', recall_classwise, '\n',
    #       'f1_all: ', f1_all, '\n',
    #       'f1_classwise: ', f1_classwise, '\n',
    #       'csi_all: ', csi_all, '\n',
    #       'csi_classwise: ', csi_classwise, '\n',
    #       'kappa_all: ', kappa_all, '\n',
    #       'kappa_classwise: ', kappa_classwise, '\n',
    #       'conf_all: ', conf_all, '\n',
    #       'conf_classwise: ', conf_classwise, '\n',
    #       )
    # # generate fake sample pairs
    # # pred = torch.tensor([[[[0.2, 0.4, 0.1],
    # #                        [0.7, 0.3, 0.8],
    # #                        [0.6, 0.9, 0.4]],
    # #
    # #                       [[0.8, 0.6, 0.9],
    # #                        [0.3, 0.7, 0.2],
    # #                        [0.4, 0.1, 0.6]]]])
    # #
    # # ground_truth = torch.tensor([[[[0, 1, 1],
    # #                                [1, 0, 0],
    # #                                [0, 1, 1]]]])
    # # # get formatted prediction and ground truth
    # # pred = pred[:, 1, :, :].unsqueeze(1)
    # # # generate fake sample pair
    # # metric_acc = BinaryAccuracy(multidim_average='samplewise', threshold=0.5)
    # # metric_cohenkappa = BinaryCohenKappa(threshold=0.5)
    # # metric_precision = BinaryPrecision(multidim_average='samplewise', threshold=0.5)
    # # metric_recall = BinaryRecall(multidim_average='samplewise', threshold=0.5)
    # # metric_f1score = BinaryF1Score(multidim_average='samplewise', threshold=0.5)
    # # metric_csi = BinaryCSI(threshold=0.5)
    # # metric_confusion = BinaryConfusionMatrix(threshold=0.5)
    # # metric_pr = BinaryPrecisionRecallCurve()
    # # metric_pr.update(pred, ground_truth)
    # # fig_, ax_ = metric_pr.plot(score=True)
    # # fig_.savefig('pr_curve.png')
    # # print('ACC: ', metric_acc(pred, ground_truth), '\n',
    # #       'CohenKappa: ', metric_cohenkappa(pred, ground_truth), '\n',
    # #       'Precision: ', metric_precision(pred, ground_truth), '\n',
    # #       'Recall: ', metric_recall(pred, ground_truth), '\n',
    # #       'F1Score: ', metric_f1score(pred, ground_truth), '\n',
    # #       'CSI: ', metric_csi(pred, ground_truth), '\n',
    # #       'Confusion Matrix: ', metric_confusion(pred, ground_truth), '\n',
    # #       'Precision-Recall Curve: ', metric_pr(pred, ground_truth), '\n',
    # #       'Dice Score: ', F.mse_loss(pred, ground_truth), '\n',
    # #       )
