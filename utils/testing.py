import torch
from adaptation import LMMDLoss
from utils.clip_util import AverageMeter
import os
import utils.clip_util as clu
import torch.nn as nn
import numpy as np
from utils.Plots import plot_reliability_diagram, ECE
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score,confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score
from sklearn.metrics import precision_score, recall_score
from ROC_DCA import calculate_net_benefit_multiclass, net_benefit_none, net_benefit_all, roc
import torch.nn.functional as F
from nets.PromptCLIP import PromptLearner_client, TextEncoder
from nets.CoCoOpCLIP import TextEncoderCocoOp
from torch.distributions import Categorical
from utils.Plots import save_retention_curve_values
from utils.tta import compute_sfea_loss,entropy_minimization_loss


def toeval(model):
    model.model.eval()
    model.fea_attn.eval()

def toevalMLP(mlp):
    mlp.eval()

# def calculate_entropy(prob):
#     distributions = Categorical(probs=prob)
#     entropies = distributions.entropy()
#     average_entropy = entropies.mean()
#     return average_entropy.item()
def calculate_entropy(prob): # Take average for a batch of samples, do not consider single sample
    # prob: (B, C) tensor where B is the batch size and C is the number of classes
    log_prob = F.log_softmax(prob, dim=1)  # Apply log-softmax for numerical stability
    entropy = -torch.sum(prob * log_prob, dim=1)  # Compute entropy for each row
    average_entropy = entropy.mean()  # Compute the average entropy across the batch
    return average_entropy.item()

def calculate_entropy_single(prob):
    """
    prob: (B, C) tensor, where B is batch size and C is number of classes
    """
    log_prob = F.log_softmax(prob, dim=1) 
    entropy = -torch.sum(prob * log_prob, dim=1)
    return entropy 


def test(args, model, data_loader, device, mlp, **kwargs):
    toeval(model)
    toevalMLP(mlp)
    total = 0
    correct = 0
    correct2 = 0
    correct3 = 0
    bacc = 0
    Prediction = []
    Preds = []
    Label = []
    tpr = []
    fpr = []
    texts = model.labels
    text_features = clu.get_text_features_list(texts, model.model).float()
    if args.method == 'ours' or args.method == 'fedclip' or args.method == 'facmic':
        with torch.no_grad():
            for batch in (data_loader):
                image, text, label = batch
                image = image.to(device)
                label = label.to(device)
                image_features = clu.get_image_features(
                    image, model.model, model.preprocess).float()
                image_features_attn = model.fea_attn(image_features)
                image_features = torch.mul(
                    image_features_attn, image_features).detach()


                output = mlp(image_features)
                _, indices = torch.max(output, dim=1)
                total += len(label)
                pred = torch.squeeze(indices)
                res = torch.cat([pred.view(-1, 1), label.view(-1, 1)], dim=1)
                res = res.cpu().numpy()
                Prediction.append(output.cpu().numpy())
                Preds.append(indices.cpu().numpy())
                Label.append(label.cpu().numpy())
                correct += np.sum(np.array(res)[:, 0] == np.array(res)[:, 1])

            all_preds = np.concatenate(Prediction)
            all_prediction = np.concatenate(Preds)
            all_labels = np.concatenate(Label)
            net_benefits = calculate_net_benefit_multiclass(all_labels, all_preds)
            tpr, fpr = roc(all_labels, all_preds)
            net_benefits_all = net_benefit_all(all_labels)
            net_benefits_none = net_benefit_none()
            # cm = confusion_matrix(all_labels, all_prediction)
            # ConfusionMatrixDisplay(cm).plot()
            # plt.show()
            # print(all_labels.shape, all_prediction.shape, total, acc, correct, correct/total, acc/total)
            bacc = balanced_accuracy_score(all_labels, all_prediction)
            f1 = f1_score(all_labels, all_prediction, average='macro')
            precision = precision_score(all_labels, all_prediction, average='macro')
            recall = recall_score(all_labels, all_prediction, average='macro')
        return correct / total, bacc, f1, precision, recall, net_benefits, tpr, fpr, net_benefits_all, net_benefits_none
    elif args.method == 'pfedclip' or args.method == 'attn_mlp':
        with torch.no_grad():
            for batch in (data_loader):
                image, text, label = batch
                image = image.to(device)
                image = clu.add_mixed_noise(image)
                label = label.to(device)
                image_features = clu.get_image_features(
                    image, model.model, model.preprocess).float()
                image_features_attn = model.fea_attn(image_features)
                image_features = torch.mul(
                    image_features_attn, image_features).detach()

                similarity = clu.get_similarity(image_features, text_features)
                _, indices3 = similarity.topk(1)
                output = mlp(image_features)
                _, indices = torch.max(output, dim=1)
                # alpha = float(max(abs(entropy2 - 3), abs(entropy - 3)) / (abs(entropy2 - 3) + abs(entropy - 3)))
                entropy = calculate_entropy_single(similarity)
                entropy2 = calculate_entropy_single(F.softmax(output, dim=-1))
                # args.alpha = float(max(abs(entropy2), abs(entropy)) / (abs(entropy2) + abs(entropy))) # Based on entropy
                # args.alpha = ((entropy)) / ((entropy2) + (entropy))  # Based on entropy
                w1 = 1.0 / (entropy + 1e-8)  # (B,)
                w2 = 1.0 / (entropy2 + 1e-8)  # (B,)
                w2_normalized = w2 / (w1 + w2)  # (B,)
                args.alpha = w2_normalized.unsqueeze(1)
                # args.alpha = (calculate_entropy(similarity) - torch.log(torch.tensor(len(model.labels), dtype=torch.float32)))  # Based on entropy
                # print(similarity, output, (similarity + output) / 2)
                output2 = (1-args.alpha) * similarity + args.alpha * F.softmax(output, dim=-1)
                # output2 = F.softmax(output, dim=1)

                _, indices2 = torch.max(output2, dim=1)

                total += len(label)
                pred = torch.squeeze(indices) # only local
                pred2 = torch.squeeze(indices2) # combined
                pred3 = torch.squeeze(indices3) # glo
                res = torch.cat([pred.view(-1, 1), label.view(-1, 1)], dim=1)
                res = res.cpu().numpy()
                res2 = torch.cat([pred2.view(-1, 1), label.view(-1, 1)], dim=1)
                res2 = res2.cpu().numpy()
                res3 = torch.cat([pred3.view(-1, 1), label.view(-1, 1)], dim=1)
                res3 = res3.cpu().numpy()

                Prediction.append(output2.cpu().numpy())
                # Prediction.append(F.softmax(output,dim=1).cpu().numpy())
                Preds.append(pred2.cpu().numpy())
                Label.append(label.cpu().numpy())
                correct += np.sum(np.array(res)[:, 0] == np.array(res)[:, 1])
                correct2 += np.sum(np.array(res2)[:, 0] == np.array(res2)[:, 1])
                correct3 += np.sum(np.array(res3)[:, 0] == np.array(res3)[:, 1])
            all_preds = np.concatenate(Prediction)
            all_prediction = np.concatenate(Preds)
            all_labels = np.concatenate(Label)

            confs = all_preds[range(all_preds.shape[0]), all_prediction]
            # confs = logits[range(logits.shape[0]), preds]
            avg_conf = confs.mean()
            print('avg batch conf:', avg_conf)
            ece_value = ECE(confs, all_prediction, all_labels)
            print('Overall ece value:', round(ece_value, 2))
            # if os.path.exists('./CalibrationResults/miniDomainNet/CIPQRS/pFedCLIP++') == False:
            #     os.makedirs('./CalibrationResults/miniDomainNet/CIPQRS/pFedCLIP++')
            # plot_reliability_diagram(all_prediction, confs, all_labels, 10, None,
            #                          './CalibrationResults/miniDomainNet/CIPQRS/pFedCLIP++/Client_local_' + str(
            #                              args.index) + '_reliability_diagram.pdf')
            # ####
            # net_benefits = calculate_net_benefit_multiclass(all_labels, all_preds)
            net_benefits = []
            # tpr, fpr = roc(all_labels, all_preds)
            # net_benefits_all = net_benefit_all(all_labels)
            tpr = []
            fpr = []
            net_benefits_all = []
            net_benefits_none = net_benefit_none()
            bacc = balanced_accuracy_score(all_labels, all_prediction)
            f1 = f1_score(all_labels, all_prediction, average='macro')
            precision = precision_score(all_labels, all_prediction, average='macro')
            recall = recall_score(all_labels, all_prediction, average='macro')
            print('Accuracy using FAM only: {:.4f} |Accuracy using MLP only: {:.4f} |'
                  ' Accuracy using FAM and MLP: {:.4f} | alpha: {:.2f} | entropy_FAM: {:.2f} | entropy_MLP: {:.2f}'
                  .format(correct3 / total, correct / total, correct2 / total, args.alpha.mean().item(), entropy.mean().item(), entropy2.mean().item()))
            # print('Accuracy using MLP only (without Softmax): {:.4f}'.format(correct / total))
            # print('Accuracy using MLP and FAM: {:.4f}'.format(correct2 / total))
        return correct2 / total, bacc, f1, precision, recall, net_benefits, tpr, fpr, net_benefits_all, net_benefits_none

    elif args.method == 'attn_mlp_tta':
        # TTA
        clu.freeze_param_norm(mlp)
        for e in range(0, 5):
            for batch in (data_loader):
                optimizer_tta = kwargs['optimizer_tta']
                image, text, label = batch
                image = image.to(device)
                image_features = model.model.encode_image(image).float()
                image_features_attn = model.fea_attn(image_features).float()
                image_features = torch.mul(
                    image_features_attn, image_features).float()

                image_features = image_features / \
                                 image_features.norm(dim=1, keepdim=True)

                similarities = torch.mm(image_features, text_features.T)
                logits_mlp = mlp(image_features)
                # entropy = calculate_entropy_single(similarities)
                # entropy2 = calculate_entropy_single(F.softmax(logits_mlp, dim=-1))
                # w1 = 1.0 / (entropy + 1e-8)  # (B,)
                # w2 = 1.0 / (entropy2 + 1e-8)  # (B,)
                # w2_normalized = w2 / (w1 + w2)  # (B,)
                # args.alpha = w2_normalized.unsqueeze(1)
                # logits_ens = (1 - args.alpha) * similarities + args.alpha * logits_mlp
                loss_tta = compute_sfea_loss(logits_mlp) + entropy_minimization_loss(logits_mlp)
                optimizer_tta.zero_grad()
                print(loss_tta)
                loss_tta.backward()
                optimizer_tta.step()

        with torch.no_grad():
            for batch in (data_loader):
                image, text, label = batch
                image = image.to(device)
                image = clu.add_mixed_noise(image)
                label = label.to(device)
                image_features = clu.get_image_features(
                    image, model.model, model.preprocess).float()
                image_features_attn = model.fea_attn(image_features)
                image_features = torch.mul(
                    image_features_attn, image_features).detach()

                similarity = clu.get_similarity(image_features, text_features)
                _, indices3 = similarity.topk(1)
                output = mlp(image_features)
                _, indices = torch.max(output, dim=1)
                # alpha = float(max(abs(entropy2 - 3), abs(entropy - 3)) / (abs(entropy2 - 3) + abs(entropy - 3)))
                entropy = calculate_entropy_single(similarity)
                entropy2 = calculate_entropy_single(F.softmax(output, dim=-1))
                # args.alpha = float(max(abs(entropy2), abs(entropy)) / (abs(entropy2) + abs(entropy))) # Based on entropy
                # args.alpha = ((entropy)) / ((entropy2) + (entropy))  # Based on entropy
                w1 = 1.0 / (entropy + 1e-8)  # (B,)
                w2 = 1.0 / (entropy2 + 1e-8)  # (B,)
                w2_normalized = w2 / (w1 + w2)  # (B,)
                args.alpha = w2_normalized.unsqueeze(1)
                # args.alpha = (calculate_entropy(similarity) - torch.log(torch.tensor(len(model.labels), dtype=torch.float32)))  # Based on entropy
                # print(similarity, output, (similarity + output) / 2)
                output2 = (1-args.alpha) * similarity + args.alpha * F.softmax(output, dim=-1)
                # output2 = F.softmax(output, dim=1)

                _, indices2 = torch.max(output2, dim=1)

                total += len(label)
                pred = torch.squeeze(indices) # only local
                pred2 = torch.squeeze(indices2) # combined
                pred3 = torch.squeeze(indices3) # glo
                res = torch.cat([pred.view(-1, 1), label.view(-1, 1)], dim=1)
                res = res.cpu().numpy()
                res2 = torch.cat([pred2.view(-1, 1), label.view(-1, 1)], dim=1)
                res2 = res2.cpu().numpy()
                res3 = torch.cat([pred3.view(-1, 1), label.view(-1, 1)], dim=1)
                res3 = res3.cpu().numpy()

                Prediction.append(output2.cpu().numpy())
                # Prediction.append(F.softmax(output,dim=1).cpu().numpy())
                Preds.append(pred2.cpu().numpy())
                Label.append(label.cpu().numpy())
                correct += np.sum(np.array(res)[:, 0] == np.array(res)[:, 1])
                correct2 += np.sum(np.array(res2)[:, 0] == np.array(res2)[:, 1])
                correct3 += np.sum(np.array(res3)[:, 0] == np.array(res3)[:, 1])
            all_preds = np.concatenate(Prediction)
            all_prediction = np.concatenate(Preds)
            all_labels = np.concatenate(Label)

            confs = all_preds[range(all_preds.shape[0]), all_prediction]
            # confs = logits[range(logits.shape[0]), preds]
            avg_conf = confs.mean()
            print('avg batch conf:', avg_conf)
            ece_value = ECE(confs, all_prediction, all_labels)
            print('Overall ece value:', round(ece_value, 2))
            # if os.path.exists('./CalibrationResults/miniDomainNet/CIPQRS/pFedCLIP++') == False:
            #     os.makedirs('./CalibrationResults/miniDomainNet/CIPQRS/pFedCLIP++')
            # plot_reliability_diagram(all_prediction, confs, all_labels, 10, None,
            #                          './CalibrationResults/miniDomainNet/CIPQRS/pFedCLIP++/Client_local_' + str(
            #                              args.index) + '_reliability_diagram.pdf')
            # ####
            # net_benefits = calculate_net_benefit_multiclass(all_labels, all_preds)
            net_benefits = []
            # tpr, fpr = roc(all_labels, all_preds)
            # net_benefits_all = net_benefit_all(all_labels)
            tpr = []
            fpr = []
            net_benefits_all = []
            net_benefits_none = net_benefit_none()
            bacc = balanced_accuracy_score(all_labels, all_prediction)
            f1 = f1_score(all_labels, all_prediction, average='macro')
            precision = precision_score(all_labels, all_prediction, average='macro')
            recall = recall_score(all_labels, all_prediction, average='macro')
            print('Accuracy using FAM only: {:.4f} |Accuracy using MLP only: {:.4f} |'
                  ' Accuracy using FAM and MLP: {:.4f} | alpha: {:.2f} | entropy_FAM: {:.2f} | entropy_MLP: {:.2f}'
                  .format(correct3 / total, correct / total, correct2 / total, args.alpha.mean().item(), entropy.mean().item(), entropy2.mean().item()))
        return correct2 / total, bacc, f1, precision, recall, net_benefits, tpr, fpr, net_benefits_all, net_benefits_none

    elif args.method == 'fedmlp' or args.method == 'CLIPFC':
        with torch.no_grad():
            for batch in (data_loader):
                image, text, label = batch
                image = image.to(device)
                label = label.to(device)
                image_features = clu.get_image_features(
                    image, model.model, model.preprocess).float()

                output = mlp(image_features)
                _, indices = torch.max(output, 1)
                total += len(label)
                pred = torch.squeeze(indices)
                res = torch.cat([pred.view(-1, 1), label.view(-1, 1)], dim=1)
                res = res.cpu().numpy()
                Prediction.append(output.cpu().numpy())
                Preds.append(indices.cpu().numpy())
                Label.append(label.cpu().numpy())
                correct += np.sum(np.array(res)[:, 0] == np.array(res)[:, 1])
            all_preds = np.concatenate(Prediction)
            all_prediction = np.concatenate(Preds)
            all_labels = np.concatenate(Label)
            net_benefits = calculate_net_benefit_multiclass(all_labels, all_preds)
            tpr, fpr = roc(all_labels, all_preds)
            net_benefits_all = net_benefit_all(all_labels)
            net_benefits_none = net_benefit_none()
            bacc = balanced_accuracy_score(all_labels, all_prediction)
            f1 = f1_score(all_labels, all_prediction, average='macro')
            precision = precision_score(all_labels, all_prediction, average='macro')
            recall = recall_score(all_labels, all_prediction, average='macro')
        return correct / total, bacc, f1, precision, recall, net_benefits, tpr, fpr, net_benefits_all, net_benefits_none
    elif args.method == 'promptFL':
        bacc = 0
        tokenized_prompts = mlp.tokenized_prompts  #
        text_encoder = TextEncoder(clip_model=model.model).to('cuda')  # freezed
        Prediction = []
        Label = []
        with torch.no_grad():
            for batch in data_loader:
                image, _, label = batch
                image = image.to(device)
                image = clu.add_mixed_noise(image)
                label = label.to(device)
                image_features = model.model.encode_image(image).float()
                prompts = mlp()  # prompt learner
                text_features = text_encoder(prompts, tokenized_prompts).float()

                image_features = image_features / \
                                 image_features.norm(dim=1, keepdim=True)
                text_features = text_features / \
                                text_features.norm(dim=1, keepdim=True)

                logit_scale = model.model.logit_scale.exp()
                logits = logit_scale * image_features @ text_features.t()
                _, indices = torch.max(logits, dim=-1)  # indices equals pseudo-label
                total += len(label)
                pred = torch.squeeze(indices)
                # print(indices, pred)
                res = torch.cat([pred.view(-1, 1), label.view(-1, 1)], dim=1)
                res = res.cpu().numpy()
                Preds.append(indices.cpu().numpy())
                Prediction.append(F.softmax(logits,dim=-1).cpu().numpy())
                Label.append(label.cpu().numpy())
                correct += np.sum(np.array(res)[:, 0] == np.array(res)[:, 1])
            all_preds = np.concatenate(Prediction)
            all_prediction = np.concatenate(Preds)
            all_labels = np.concatenate(Label)
            confs = all_preds[range(all_preds.shape[0]), all_prediction]
            # confs = logits[range(logits.shape[0]), preds]
            avg_conf = confs.mean()
            print('avg batch conf:', avg_conf)
            ece_value = ECE(confs, all_prediction, all_labels)
            print('Overall ece value:', round(ece_value, 2))
            if os.path.exists('./CalibrationResults/miniDomainNet/CIPQRS/PromptFL/') == False:
                os.makedirs('./CalibrationResults/miniDomainNet/CIPQRS/PromptFL/')
            # plot_reliability_diagram(all_prediction, confs, all_labels, 10, None,
            #                          './CalibrationResults/miniDomainNet/CIPQRS/PromptFL/Client_' + str(
            #                              args.index) + '_reliability_diagram.pdf')
            # net_benefits = calculate_net_benefit_multiclass(all_labels, all_preds)
            net_benefits = []
            # tpr, fpr = roc(all_labels, all_preds)
            tpr = []
            fpr = []
            net_benefits_all = net_benefit_all(all_labels)
            net_benefits_none = net_benefit_none()
            bacc = balanced_accuracy_score(all_labels, all_prediction)
            f1 = f1_score(all_labels, all_prediction, average='macro')
            precision = precision_score(all_labels, all_prediction, average='macro')
            recall = recall_score(all_labels, all_prediction, average='macro')
        return correct / total, bacc, f1, precision, recall, net_benefits, tpr, fpr, net_benefits_all, net_benefits_none
    elif args.method == 'FAA-CLIP':
        with torch.no_grad():
            for batch in tqdm(data_loader):
                image, text, label = batch
                image = image.to(device)
                label = label.to(device)
                image_features = clu.get_image_features(
                    image, model.model, model.preprocess).float()
                image_features_attn = model.fea_attn(image_features)
                image_features = torch.mul(
                    image_features_attn, image_features).detach()

                similarity = clu.get_similarity(image_features, text_features)
                _, indices = similarity.topk(1)
                total += len(label)
                pred = torch.squeeze(indices)
                res = torch.cat([ pred.view(-1, 1), label.view(-1, 1) ], dim=1)
                res = res.cpu().numpy()
                Prediction.append(similarity.cpu().numpy())
                Preds.append(pred.cpu().numpy())
                Label.append(label.cpu().numpy())
                correct += np.sum(np.array(res)[ :, 0 ] == np.array(res)[ :, 1 ])

            all_preds = np.concatenate(Prediction)
            all_prediction = np.concatenate(Preds)
            all_labels = np.concatenate(Label)
            net_benefits = calculate_net_benefit_multiclass(all_labels, all_preds)
            tpr, fpr = roc(all_labels, all_preds)
            net_benefits_all = net_benefit_all(all_labels)
            net_benefits_none = net_benefit_none()
            confs = all_preds[range(all_preds.shape[0]), all_prediction]
            # confs = logits[range(logits.shape[0]), preds]
            avg_conf = confs.mean()
            print('avg batch conf:', avg_conf)
            ece_value = ECE(confs, all_prediction, all_labels)
            print('Overall ece value:', round(ece_value, 2))
            if not os.path.exists('./CalibrationResults/OfficeHome/ACPR/FAA-CLIP/'):
                os.makedirs('./CalibrationResults/OfficeHome/ACPR/FAA-CLIP/')
            # plot_reliability_diagram(all_prediction, confs, all_labels, 10, None,
            #                          './CalibrationResults/OfficeHome/ACPR/FAA-CLIP/Client_local_FAM_' + str(
            #                              args.index) + '_reliability_diagram.pdf')

            bacc = balanced_accuracy_score(all_labels, all_prediction)
            f1 = f1_score(all_labels, all_prediction, average='macro')
            precision = precision_score(all_labels, all_prediction, average='macro')
            recall = recall_score(all_labels, all_prediction, average='macro')
            print('Accuracy: {:.4f}'.format(correct / total))
        return correct / total, bacc, f1, precision, recall, net_benefits, tpr, fpr, net_benefits_all, net_benefits_none

    elif args.method == 'CocoOpCLIP':
        bacc = 0
        tokenized_prompts = mlp.tokenized_prompts  #
        text_encoder = TextEncoderCocoOp(clip_model=model.model).to('cuda')  # freezed
        Prediction = []
        Label = []
        with torch.no_grad():
            for batch in data_loader:
                image, _, label = batch
                image = image.to(device)
                image = clu.add_mixed_noise(image)
                label = label.to(device)
                image_features = model.model.encode_image(image)
                image_features = image_features / \
                                 image_features.norm(dim=1, keepdim=True)

                prompts = mlp(image_features)
                logit_scale = model.model.logit_scale.exp()
                logits = [ ]
                for pts_i, imf_i in zip(prompts, image_features):
                    text_features = text_encoder(pts_i, tokenized_prompts)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    l_i = logit_scale * imf_i @ text_features.t()
                    logits.append(l_i)
                logits = torch.stack(logits)
                _, indices = torch.max(logits, dim=-1)  # indices equals pseudo-label
                total += len(label)
                pred = torch.squeeze(indices)
                # print(indices, pred)
                res = torch.cat([pred.view(-1, 1), label.view(-1, 1)], dim=1)
                res = res.cpu().numpy()
                Preds.append(indices.cpu().numpy())
                Prediction.append(F.softmax(logits,dim=1).cpu().numpy())
                Label.append(label.cpu().numpy())
                correct += np.sum(np.array(res)[:, 0] == np.array(res)[:, 1])
            all_preds = np.concatenate(Prediction)
            all_prediction = np.concatenate(Preds)
            all_labels = np.concatenate(Label)
            net_benefits = calculate_net_benefit_multiclass(all_labels, all_preds)
            # tpr, fpr = roc(all_labels, all_preds)
            confs = all_preds[range(all_preds.shape[0]), all_prediction]
            # confs = logits[range(logits.shape[0]), preds]
            avg_conf = confs.mean()
            print('avg batch conf:', avg_conf)
            ece_value = ECE(confs, all_prediction, all_labels)
            print('Overall ece value:', round(ece_value, 2))
            if os.path.exists('./CalibrationResults/miniDomainNet/CIPQRS/CocoOp/') == False:
                os.makedirs('./CalibrationResults/miniDomainNet/CIPQRS/CocoOp/')
            # plot_reliability_diagram(all_prediction, confs, all_labels, 10, None,
            #                          './CalibrationResults/miniDomainNet/CIPQRS/CocoOp/Client_local_' + str(
            #                              args.index) + '_reliability_diagram.pdf')
            net_benefits_all = net_benefit_all(all_labels)
            net_benefits_none = net_benefit_none()
            bacc = balanced_accuracy_score(all_labels, all_prediction)
            f1 = f1_score(all_labels, all_prediction, average='macro')
            precision = precision_score(all_labels, all_prediction, average='macro')
            recall = recall_score(all_labels, all_prediction, average='macro')
        return correct / total, bacc, f1, precision, recall, net_benefits, tpr, fpr, net_benefits_all, net_benefits_none

    else:
        bacc = 0
        Prediction = []
        Label = []
        with torch.no_grad():
            for batch in data_loader:
                image, _, label = batch
                image = image.to(device)
                label = label.to(device)
                image_features = clu.get_image_features(
                    image, model.model, model.preprocess).float()
                similarity = clu.get_similarity(image_features, text_features)
                _, indices = similarity.topk(1)  # indices equals pseudo-label
                total += len(label)
                pred = torch.squeeze(indices)
                # print(indices, pred)
                res = torch.cat([pred.view(-1, 1), label.view(-1, 1)], dim=1)
                res = res.cpu().numpy()
                Prediction.append(similarity.cpu().numpy())
                Preds.append(pred.cpu().numpy())
                Label.append(label.cpu().numpy())
                correct += np.sum(np.array(res)[:, 0] == np.array(res)[:, 1])
            all_preds = np.concatenate(Prediction)
            all_prediction = np.concatenate(Preds)
            all_labels = np.concatenate(Label)
            confs = all_preds[range(all_preds.shape[0]), all_prediction]
            # confs = logits[range(logits.shape[0]), preds]
            avg_conf = confs.mean()
            print('avg batch conf:', avg_conf)
            ece_value = ECE(confs, all_prediction, all_labels)
            print('Overall ece value:', round(ece_value, 2))
            if os.path.exists('./CalibrationResults/miniDomainNet/CIPQRS/LoRA') == False:
                os.makedirs('./CalibrationResults/miniDomainNet/CIPQRS/LoRA')
            # plot_reliability_diagram(all_prediction, confs, all_labels, 10, None,
            #                          './CalibrationResults/miniDomainNet/CIPQRS/LoRA/Client_local_' + str(
            #                              args.index) + '_reliability_diagram.pdf')
            # net_benefits = calculate_net_benefit_multiclass(all_labels, all_preds)
            net_benefits = []
            # tpr, fpr = roc(all_labels, all_preds)
            tpr = []
            fpr = []
            net_benefits_all = net_benefit_all(all_labels)
            net_benefits_none = net_benefit_none()
            bacc = balanced_accuracy_score(all_labels, all_prediction)
            f1 = f1_score(all_labels, all_prediction, average='macro')
            precision = precision_score(all_labels, all_prediction, average='macro')
            recall = recall_score(all_labels, all_prediction, average='macro')
        return correct / total, bacc, f1, precision, recall, net_benefits, tpr, fpr, net_benefits_all, net_benefits_none


def Glotest(args, model, data_loader, device):
    toeval(model)
    total = 0
    correct = 0
    bacc = 0
    Prediction = []
    Preds = []
    Label = []
    tpr = []
    fpr = []
    texts = model.labels
    text_features = clu.get_text_features_list(texts, model.model).float()
    if (args.method == 'ours' or args.method == 'fedclip' or
            args.method == 'clip' or args.method == 'pfedclip'
            or args.method == 'facmic' or args.method == 'attn_mlp'):
        with torch.no_grad():
            for batch in (data_loader):
                image, text, label = batch
                text = text.to(device)
                image = image.to(device)
                image = clu.add_mixed_noise(image)
                label = label.to(device)
                image_features = clu.get_image_features(
                    image, model.model, model.preprocess).float()
                image_features_attn = model.fea_attn(image_features)
                image_features = torch.mul(
                    image_features_attn, image_features).detach()

                similarity = clu.get_similarity(image_features, text_features)
                _, indices = similarity.topk(1)  # indices equals pseudo-label

                total += len(label)
                pred = torch.squeeze(indices)
                res = torch.cat([pred.view(-1, 1), label.view(-1, 1)], dim=1)
                res = res.cpu().numpy()
                Prediction.append(similarity.cpu().numpy())
                Preds.append(pred.cpu().numpy())
                Label.append(label.cpu().numpy())
                correct += np.sum(np.array(res)[:, 0] == np.array(res)[:, 1])
            all_preds = np.concatenate(Prediction)
            all_prediction = np.concatenate(Preds)
            all_labels = np.concatenate(Label)
            net_benefits = calculate_net_benefit_multiclass(all_labels, all_preds)
            # tpr, fpr = roc(all_labels, all_preds)
            net_benefits_all = net_benefit_all(all_labels)
            net_benefits_none = net_benefit_none()
            bacc = balanced_accuracy_score(all_labels, all_prediction)
            confs = all_preds[range(all_preds.shape[0]), all_prediction]
            # confs = logits[range(logits.shape[0]), preds]
            avg_conf = confs.mean()
            print('avg batch conf:', avg_conf)
            ece_value = ECE(confs, all_prediction, all_labels)
            print('Overall ece value:', round(ece_value, 2))
            # if not os.path.exists('./CalibrationResults/miniDomainNet/CIPQRS/FedCLIP/'):
            #     os.makedirs('./CalibrationResults/miniDomainNet/CIPQRS/FedCLIP/')
            # plot_reliability_diagram(all_prediction, confs, all_labels, 10, None,
            #                          './CalibrationResults/miniDomainNet/CIPQRS/FedCLIP/Global_FAM_' + str(
            #                              args.index) + '_reliability_diagram.pdf')
            f1 = f1_score(all_labels, all_prediction, average='macro')
            precision = precision_score(all_labels, all_prediction, average='macro')
            recall = recall_score(all_labels, all_prediction, average='macro')
        return correct / total, bacc, f1, precision, recall, net_benefits, tpr, fpr, net_benefits_all, net_benefits_none

    elif args.method == 'FAA-CLIP' or args.method == 'FAA-CLIP-MLP':
        with torch.no_grad():
            for batch in tqdm(data_loader):
                image, text, label = batch
                image = image.to(device)
                label = label.to(device)
                image_features = clu.get_image_features(
                    image, model.model, model.preprocess).float()
                image_features_attn = model.fea_attn(image_features)
                image_features = torch.mul(
                    image_features_attn, image_features).detach()

                similarity = clu.get_similarity(image_features, text_features)
                _, indices = similarity.topk(1)
                total += len(label)
                pred = torch.squeeze(indices)
                res = torch.cat([ pred.view(-1, 1), label.view(-1, 1) ], dim=1)
                res = res.cpu().numpy()
                Prediction.append(similarity.cpu().numpy())
                Preds.append(indices.cpu().numpy())
                Label.append(label.cpu().numpy())
                correct += np.sum(np.array(res)[ :, 0 ] == np.array(res)[ :, 1 ])

            all_preds = np.concatenate(Prediction)
            all_prediction = np.concatenate(Preds)
            all_labels = np.concatenate(Label)
            net_benefits = calculate_net_benefit_multiclass(all_labels, all_preds)
            tpr, fpr = roc(all_labels, all_preds)
            net_benefits_all = net_benefit_all(all_labels)
            net_benefits_none = net_benefit_none()
            bacc = balanced_accuracy_score(all_labels, all_prediction)
            f1 = f1_score(all_labels, all_prediction, average='macro')
            precision = precision_score(all_labels, all_prediction, average='macro')
            recall = recall_score(all_labels, all_prediction, average='macro')
            print('Accuracy: {:.4f}'.format(correct / total))
        return correct / total, bacc, f1, precision, recall, net_benefits, tpr, fpr, net_benefits_all, net_benefits_none

    else:
        bacc = 0
        Prediction = []
        Label = []
        with torch.no_grad():
            for batch in data_loader:
                image, _, label = batch
                image = image.to(device)
                label = label.to(device)
                image_features = clu.get_image_features(
                    image, model.model, model.preprocess).float()
                similarity = clu.get_similarity(image_features, text_features)
                _, indices = similarity.topk(1)  # indices equals pseudo-label
                total += len(label)
                pred = torch.squeeze(indices)
                # print(indices, pred)
                res = torch.cat([pred.view(-1, 1), label.view(-1, 1)], dim=1)
                res = res.cpu().numpy()
                Prediction.append(similarity.cpu().numpy())
                Preds.append(pred.cpu().numpy())
                Label.append(label.cpu().numpy())
                correct += np.sum(np.array(res)[:, 0] == np.array(res)[:, 1])
            all_preds = np.concatenate(Prediction)
            all_prediction = np.concatenate(Preds)
            all_labels = np.concatenate(Label)
            confs = all_preds[range(all_preds.shape[0]), all_prediction]
            # confs = logits[range(logits.shape[0]), preds]
            avg_conf = confs.mean()
            print('avg batch conf:', avg_conf)
            ece_value = ECE(confs, all_prediction, all_labels)
            print('Overall ece value:', round(ece_value, 2))
            # if os.path.exists('./CalibrationResults/miniDomainNet/CIPQRS/FedAVG/') == False:
            #     os.makedirs('./CalibrationResults/miniDomainNet/CIPQRS/FedAVG/')
            # plot_reliability_diagram(all_prediction, confs, all_labels, 10, None,
            #                          './CalibrationResults/miniDomainNet/CIPQRS/FedAVG/Client_' + str(
            #                              args.index) + '_reliability_diagram.pdf')
            # net_benefits = calculate_net_benefit_multiclass(all_labels, all_preds)
            net_benefits = []
            # tpr, fpr = roc(all_labels, all_preds)
            tpr = []
            fpr = []
            net_benefits_all = net_benefit_all(all_labels)
            net_benefits_none = net_benefit_none()
            bacc = balanced_accuracy_score(all_labels, all_prediction)
            f1 = f1_score(all_labels, all_prediction, average='macro')
            precision = precision_score(all_labels, all_prediction, average='macro')
            recall = recall_score(all_labels, all_prediction, average='macro')
        return correct / total, bacc, f1, precision, recall, net_benefits, tpr, fpr, net_benefits_all, net_benefits_none

def testlp(args, model, data_loader, device, mlp, text_weights, alpha_vec):
    toeval(model)
    toevalMLP(mlp)
    features, labels = [ ], [ ]
    with torch.no_grad():
        for batch in (data_loader):
            image, text, label = batch
            images = image.to(device)
            images = clu.add_mixed_noise(images)
            label = label.to(device)
            image_features = model.model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features)
            labels.append(label)
    features, labels = torch.cat(features), torch.cat(labels)

    mlp.eval()
    vision_logits_val = mlp(features)
    text_logits_val = features.detach() @ text_weights
    logits_val = vision_logits_val + torch.ones(features.shape[ 0 ], 1).to(
        model.model.dtype).cuda() @ alpha_vec * text_logits_val
    acc_val = np.mean(logits_val.argmax(dim=1).cpu().numpy() == labels.cpu().numpy()) * 100.0
    precision = precision_score(labels.cpu().numpy(), logits_val.argmax(dim=1).cpu().numpy(), average='macro')
    recall = recall_score(labels.cpu().numpy(), logits_val.argmax(dim=1).cpu().numpy(), average='macro')
    f1 = f1_score(labels.cpu().numpy(), logits_val.argmax(dim=1).cpu().numpy(), average='macro')
    bacc = balanced_accuracy_score(labels.cpu().numpy(), logits_val.argmax(dim=1).cpu().numpy())
    # net_benefits = calculate_net_benefit_multiclass(logits_val.argmax(dim=1).cpu().numpy(), labels.cpu().numpy())
    # tpr, fpr = roc(labels.cpu().numpy(), logits_val.argmax(dim=1).cpu().numpy())
    net_benefits = [ ]
    tpr = [ ]
    fpr = [ ]
    net_benefits_all = net_benefit_all(labels.cpu().numpy())
    net_benefits_none = net_benefit_none()
    logits_val = F.softmax(logits_val, dim=1)
    confs = logits_val[range(logits_val.cpu().detach().numpy().shape[0]), logits_val.argmax(dim=1).cpu().detach().numpy()] # all preds = probs, all_prediction = preds.
    # confs = logits[range(logits.shape[0]), preds]
    avg_conf = confs.mean()
    print('avg batch conf:', avg_conf)
    ece_value = ECE(confs.cpu().detach().numpy(), logits_val.argmax(dim=1).cpu().detach().numpy(), logits_val.cpu().detach().numpy())
    print('Overall ece value:', round(ece_value, 2))
    if not os.path.exists('./CalibrationResults/miniDomainNet/CIPQRS/LP++/'):
        os.makedirs('./CalibrationResults/miniDomainNet/CIPQRS/LP++/')
    # plot_reliability_diagram(logits_val.argmax(dim=1).cpu().detach().numpy(), confs.cpu().detach().numpy(), labels.cpu().detach().numpy(), 10, None,
    #                          './CalibrationResults/miniDomainNet/CIPQRS/LP++/Global_FAM_' + str(
    #                              args.index) + '_reliability_diagram.pdf')
    print('The accuracy for val data is ', acc_val)
    return acc_val / 100., bacc, f1, precision, recall, net_benefits, tpr, fpr, net_benefits_all, net_benefits_none


def test_fedclip_client(args, model, data_loader, device, mlps, client_classifier, client_index):
    toeval(model)
    for mlp in mlps:
        toevalMLP(mlp)
    total = 0
    correct = 0
    correct2 = 0
    correct3 = 0
    correct_localmlp = 0
    bacc = 0
    Prediction = []
    Preds = []
    Label = []
    texts = model.labels
    text_features = clu.get_text_features_list(texts, model.model).float()
    if args.method == 'fedclip_client':
        with torch.no_grad():
            for batch in (data_loader):
                image, text, label = batch
                image = image.to(device)
                image = clu.add_mixed_noise(image)
                label = label.to(device)
                image_features = clu.get_image_features(
                    image, model.model, model.preprocess).float()
                image_features_attn = model.fea_attn(image_features)
                image_features = torch.mul(
                    image_features_attn, image_features).detach()
                similarity = clu.get_similarity(image_features, text_features)

                client_weights = F.softmax(client_classifier(image_features),dim=-1) # [B, N]
                # print(client_weights)
                output = torch.zeros_like(mlps[0](image_features))  # [B, C]
                for client in range(len(mlps)):
                    logits = F.softmax(mlps[client](image_features),dim=1)  # [B, C] For each b in B,
                    expanded_weights = client_weights[:, client].unsqueeze(1)  # [B, 1] read the i-th mlp weights
                    # print(expanded_weights, expanded_weights.shape)
                    # i-th MLP outputs multiply i-th client weights
                    output += expanded_weights * logits
                print(client_index)
                if client_index == args.n_clients:
                    output_localmlp = mlps[client_index-1](image_features)
                else:
                    output_localmlp = mlps[client_index](image_features)
                # logits_list = [mlp(image_features) for mlp in mlps] # [B,C]
                # logit = torch.stack(logits_list, dim=1) # [B,N,C]
                # expanded_weights = client_weights.unsqueeze(-1).expand_as(logit) # [B,N,C]
                # output = (expanded_weights * logit).sum(dim=1) # [B,C]
                # output = mlps[client_index](image_features)
                _, indices3 = similarity.topk(1)
                _, indices = torch.max(output, dim=1)
                _, indices_localmlp = torch.max(output_localmlp, dim=1)
                # alpha = float(max(abs(entropy2 - 3), abs(entropy - 3)) / (abs(entropy2 - 3) + abs(entropy - 3)))
                entropy = calculate_entropy_single(similarity)
                entropy2 = calculate_entropy_single(F.softmax(output, dim=-1))
                # args.alpha = float(max(abs(entropy2), abs(entropy)) / (abs(entropy2) + abs(entropy))) # Based on entropy
                # args.alpha = ((entropy)) / ((entropy2) + (entropy))  # Based on entropy
                w1 = 1.0 / (entropy + 1e-8)  # (B,)
                w2 = 1.0 / (entropy2 + 1e-8)  # (B,)
                w2_normalized = w2 / (w1 + w2)  # (B,)
                args.alpha = w2_normalized.unsqueeze(1)
                # args.alpha = (calculate_entropy(similarity) - torch.log(torch.tensor(len(model.labels), dtype=torch.float32)))  # Based on entropy
                # print(similarity, output, (similarity + output) / 2)
                output2 = (1-args.alpha) * similarity + args.alpha * F.softmax(output, dim=-1)
                # output2 = F.softmax(output, dim=1)

                _, indices2 = torch.max(output2, dim=1)

                total += len(label)
                pred = torch.squeeze(indices) # only local
                pred_localmlp = torch.squeeze(indices_localmlp)  # only local
                pred2 = torch.squeeze(indices2) # combined
                pred3 = torch.squeeze(indices3) # glo
                res = torch.cat([pred.view(-1, 1), label.view(-1, 1)], dim=1)
                res = res.cpu().numpy()
                res_localmlp = torch.cat([pred_localmlp.view(-1, 1), label.view(-1, 1)], dim=1)
                res_localmlp = res_localmlp.cpu().numpy()
                res2 = torch.cat([pred2.view(-1, 1), label.view(-1, 1)], dim=1)
                res2 = res2.cpu().numpy()
                res3 = torch.cat([pred3.view(-1, 1), label.view(-1, 1)], dim=1)
                res3 = res3.cpu().numpy()

                Prediction.append(output2.cpu().numpy())
                # Prediction.append(F.softmax(output,dim=1).cpu().numpy())
                Preds.append(pred2.cpu().numpy())
                Label.append(label.cpu().numpy())
                correct += np.sum(np.array(res)[:, 0] == np.array(res)[:, 1])
                correct_localmlp += np.sum(np.array(res_localmlp)[:, 0] == np.array(res_localmlp)[:, 1])
                correct2 += np.sum(np.array(res2)[:, 0] == np.array(res2)[:, 1])
                correct3 += np.sum(np.array(res3)[:, 0] == np.array(res3)[:, 1])
            all_preds = np.concatenate(Prediction)
            all_prediction = np.concatenate(Preds)
            all_labels = np.concatenate(Label)

            confs = all_preds[range(all_preds.shape[0]), all_prediction]
            # confs = logits[range(logits.shape[0]), preds]
            avg_conf = confs.mean()
            print('avg batch conf:', avg_conf)
            ece_value = ECE(confs, all_prediction, all_labels)
            print('Overall ece value:', round(ece_value, 2))
            # if os.path.exists('./CalibrationResults/miniDomainNet/CIPQRS/pFedCLIP++') == False:
            #     os.makedirs('./CalibrationResults/miniDomainNet/CIPQRS/pFedCLIP++')
            # plot_reliability_diagram(all_prediction, confs, all_labels, 10, None,
            #                          './CalibrationResults/miniDomainNet/CIPQRS/pFedCLIP++/Client_local_' + str(
            #                              args.index) + '_reliability_diagram.pdf')
            # ####
            # net_benefits = calculate_net_benefit_multiclass(all_labels, all_preds)
            net_benefits = []
            # tpr, fpr = roc(all_labels, all_preds)
            # net_benefits_all = net_benefit_all(all_labels)
            tpr = []
            fpr = []
            net_benefits_all = []
            net_benefits_none = net_benefit_none()
            bacc = balanced_accuracy_score(all_labels, all_prediction)
            f1 = f1_score(all_labels, all_prediction, average='macro')
            precision = precision_score(all_labels, all_prediction, average='macro')
            recall = recall_score(all_labels, all_prediction, average='macro')
            print('Accuracy using FAM only: {:.4f} | Accuracy using local MLP only: {:.4f} | Accuracy using Mixture of MLPs: {:.4f} |'
                  ' Accuracy using FAM and MLP: {:.4f} | alpha: {:.2f} | entropy_FAM: {:.2f} | entropy_MLP: {:.2f}'
                  .format(correct3 / total, correct_localmlp / total, correct / total, correct2 / total, args.alpha.mean().item(), entropy.mean().item(), entropy2.mean().item()))
            # print('Accuracy using MLP only (without Softmax): {:.4f}'.format(correct / total))
            # print('Accuracy using MLP and FAM: {:.4f}'.format(correct2 / total))
        return correct2 / total, bacc, f1, precision, recall, net_benefits, tpr, fpr, net_benefits_all, net_benefits_none
