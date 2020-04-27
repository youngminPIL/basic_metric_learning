from __future__ import absolute_import

import torch
import torch.nn as nn
import numpy as np

vlr = 1
vlr2 = 1
scale = 1000000
mom = 0.9
n_ = 3
def get_size_scalar(torch_tensor):
    a,b,c,d = torch_tensor.size()
    return a*b*c*d

def get_size_scalarFC(torch_tensor):
    a,b = torch_tensor.size()
    return a*b


def compute_weight_variation_FC(modelA, modelB):

    L1_varation = []
    nweight = get_size_scalar(modelA.model.conv1.weight)
    variation = torch.pow(torch.norm(modelA.model.conv1.weight - modelB.model.conv1.weight, 2),2)
    L1_varation.append(variation.detach().numpy()/nweight*scale)

    for i in range(len(modelA.model.layer1)):
        variation = 0
        nweight = 0
        nweight += get_size_scalar(modelA.model.layer1[i].conv1.weight)
        variation += torch.pow(torch.norm(modelA.model.layer1[i].conv1.weight - modelB.model.layer1[i].conv1.weight, 2),2)
        nweight += get_size_scalar(modelA.model.layer1[i].conv2.weight)
        variation += torch.pow(torch.norm(modelA.model.layer1[i].conv2.weight - modelB.model.layer1[i].conv2.weight, 2),2)
        nweight += get_size_scalar(modelA.model.layer1[i].conv3.weight)
        variation += torch.pow(torch.norm(modelA.model.layer1[i].conv3.weight - modelB.model.layer1[i].conv3.weight, 2),2)
        L1_varation.append(variation.detach().numpy()/nweight*scale)

    #L2_varation = []
    for i in range(len(modelA.model.layer2)):
        variation = 0
        nweight = 0
        nweight += get_size_scalar(modelA.model.layer2[i].conv1.weight)
        variation += torch.pow(torch.norm(modelA.model.layer2[i].conv1.weight - modelB.model.layer2[i].conv1.weight, 2),2)
        nweight += get_size_scalar(modelA.model.layer2[i].conv2.weight)
        variation += torch.pow(torch.norm(modelA.model.layer2[i].conv2.weight - modelB.model.layer2[i].conv2.weight, 2),2)
        nweight += get_size_scalar(modelA.model.layer2[i].conv3.weight)
        variation += torch.pow(torch.norm(modelA.model.layer2[i].conv3.weight - modelB.model.layer2[i].conv3.weight, 2),2)
        L1_varation.append(variation.detach().numpy()/nweight*scale)

    #L3_varation = []
    for i in range(len(modelA.model.layer3)):
        variation = 0
        nweight = 0
        nweight += get_size_scalar(modelA.model.layer3[i].conv1.weight)
        variation += torch.pow(torch.norm(modelA.model.layer3[i].conv1.weight - modelB.model.layer3[i].conv1.weight, 2),2)
        nweight += get_size_scalar(modelA.model.layer3[i].conv2.weight)
        variation += torch.pow(torch.norm(modelA.model.layer3[i].conv2.weight - modelB.model.layer3[i].conv2.weight, 2),2)
        nweight += get_size_scalar(modelA.model.layer3[i].conv3.weight)
        variation += torch.pow(torch.norm(modelA.model.layer3[i].conv3.weight - modelB.model.layer3[i].conv3.weight, 2),2)
        L1_varation.append(variation.detach().numpy()/nweight*scale)

    #L4_varation = []

    for i in range(len(modelA.model.layer4)):
        variation = 0
        nweight = 0
        nweight += get_size_scalar(modelA.model.layer4[i].conv1.weight)
        variation += torch.pow(torch.norm(modelA.model.layer4[i].conv1.weight - modelB.model.layer4[i].conv1.weight, 2),2)
        nweight += get_size_scalar(modelA.model.layer4[i].conv2.weight)
        variation += torch.pow(torch.norm(modelA.model.layer4[i].conv2.weight - modelB.model.layer4[i].conv2.weight, 2),2)
        nweight += get_size_scalar(modelA.model.layer4[i].conv3.weight)
        variation += torch.pow(torch.norm(modelA.model.layer4[i].conv3.weight - modelB.model.layer4[i].conv3.weight, 2),2)
        L1_varation.append(variation.detach().numpy()/nweight*scale)

    variation = 0
    nweight = 0
    nweight += get_size_scalarFC(modelA.classifier.add_block[0].weight)
    variation += torch.pow(torch.norm(modelA.classifier.add_block[0].weight - modelB.classifier.add_block[0].weight, 2),2)

    nweight += get_size_scalarFC(modelA.classifier.classifier[0].weight)
    variation += torch.pow(torch.norm(modelA.classifier.classifier[0].weight - modelB.classifier.classifier[0].weight, 2),2)
    L1_varation.append(variation.detach().numpy()/nweight* scale)

    return L1_varation


def compute_weight_variation_final(modelA, modelB):

    L1_varation = []
    nweight = get_size_scalar(modelA.model.conv1.weight)
    variation = torch.pow(torch.norm(modelA.model.conv1.weight.cpu() - modelB.model.conv1.weight.cpu(), 2),2)
    L1_varation.append(variation.detach().numpy()/nweight*scale)

    for i in range(len(modelA.model.layer1)):
        variation = 0
        nweight = 0
        nweight += get_size_scalar(modelA.model.layer1[i].conv1.weight)
        variation += torch.pow(torch.norm(modelA.model.layer1[i].conv1.weight.cpu() - modelB.model.layer1[i].conv1.weight.cpu(), 2),2)
        nweight += get_size_scalar(modelA.model.layer1[i].conv2.weight)
        variation += torch.pow(torch.norm(modelA.model.layer1[i].conv2.weight.cpu() - modelB.model.layer1[i].conv2.weight.cpu(), 2),2)
        nweight += get_size_scalar(modelA.model.layer1[i].conv3.weight)
        variation += torch.pow(torch.norm(modelA.model.layer1[i].conv3.weight.cpu() - modelB.model.layer1[i].conv3.weight.cpu(), 2),2)
        L1_varation.append(variation.detach().numpy()/nweight*scale)

    #L2_varation = []
    for i in range(len(modelA.model.layer2)):
        variation = 0
        nweight = 0
        nweight += get_size_scalar(modelA.model.layer2[i].conv1.weight)
        variation += torch.pow(torch.norm(modelA.model.layer2[i].conv1.weight.cpu() - modelB.model.layer2[i].conv1.weight.cpu(), 2),2)
        nweight += get_size_scalar(modelA.model.layer2[i].conv2.weight)
        variation += torch.pow(torch.norm(modelA.model.layer2[i].conv2.weight.cpu() - modelB.model.layer2[i].conv2.weight.cpu(), 2),2)
        nweight += get_size_scalar(modelA.model.layer2[i].conv3.weight)
        variation += torch.pow(torch.norm(modelA.model.layer2[i].conv3.weight.cpu() - modelB.model.layer2[i].conv3.weight.cpu(), 2),2)
        L1_varation.append(variation.detach().numpy()/nweight*scale)

    #L3_varation = []
    for i in range(len(modelA.model.layer3)):
        variation = 0
        nweight = 0
        nweight += get_size_scalar(modelA.model.layer3[i].conv1.weight)
        variation += torch.pow(torch.norm(modelA.model.layer3[i].conv1.weight.cpu() - modelB.model.layer3[i].conv1.weight.cpu(), 2),2)
        nweight += get_size_scalar(modelA.model.layer3[i].conv2.weight)
        variation += torch.pow(torch.norm(modelA.model.layer3[i].conv2.weight.cpu() - modelB.model.layer3[i].conv2.weight.cpu(), 2),2)
        nweight += get_size_scalar(modelA.model.layer3[i].conv3.weight)
        variation += torch.pow(torch.norm(modelA.model.layer3[i].conv3.weight.cpu() - modelB.model.layer3[i].conv3.weight.cpu(), 2),2)
        L1_varation.append(variation.detach().numpy()/nweight*scale)

    #L4_varation = []

    for i in range(len(modelA.model.layer4)):
        variation = 0
        nweight = 0
        nweight += get_size_scalar(modelA.model.layer4[i].conv1.weight)
        variation += torch.pow(torch.norm(modelA.model.layer4[i].conv1.weight.cpu() - modelB.model.layer4[i].conv1.weight.cpu(), 2),2)
        nweight += get_size_scalar(modelA.model.layer4[i].conv2.weight)
        variation += torch.pow(torch.norm(modelA.model.layer4[i].conv2.weight.cpu() - modelB.model.layer4[i].conv2.weight.cpu(), 2),2)
        nweight += get_size_scalar(modelA.model.layer4[i].conv3.weight)
        variation += torch.pow(torch.norm(modelA.model.layer4[i].conv3.weight.cpu() - modelB.model.layer4[i].conv3.weight.cpu(), 2),2)
        L1_varation.append(variation.detach().numpy()/nweight*scale)

    variation = 0
    nweight = 0
    nweight += get_size_scalarFC(modelA.classifier.add_block[0].weight)
    variation += torch.pow(torch.norm(modelA.classifier.add_block[0].weight.cpu() - modelB.classifier.add_block[0].weight.cpu(), 2),2)

    nweight += get_size_scalarFC(modelA.classifier.classifier[0].weight)
    variation += torch.pow(torch.norm(modelA.classifier.classifier[0].weight.cpu() - modelB.classifier.classifier[0].weight.cpu(), 2),2)
    L1_varation.append(variation.detach().numpy()/nweight* scale)

    return L1_varation

def compute_weight_variation_final_sqrt(modelA, modelB):

    L1_varation = []
    nweight = get_size_scalar(modelA.model.conv1.weight)
    variation = torch.norm(modelA.model.conv1.weight.cpu() - modelB.model.conv1.weight.cpu(), 2)
    L1_varation.append(variation.detach().numpy()/nweight*scale)

    for i in range(len(modelA.model.layer1)):
        variation = 0
        nweight = 0
        nweight += get_size_scalar(modelA.model.layer1[i].conv1.weight)
        variation += torch.pow(torch.norm(modelA.model.layer1[i].conv1.weight.cpu() - modelB.model.layer1[i].conv1.weight.cpu(), 2),2)
        nweight += get_size_scalar(modelA.model.layer1[i].conv2.weight)
        variation += torch.pow(torch.norm(modelA.model.layer1[i].conv2.weight.cpu() - modelB.model.layer1[i].conv2.weight.cpu(), 2),2)
        nweight += get_size_scalar(modelA.model.layer1[i].conv3.weight)
        variation += torch.pow(torch.norm(modelA.model.layer1[i].conv3.weight.cpu() - modelB.model.layer1[i].conv3.weight.cpu(), 2),2)
        L1_varation.append(variation.detach().numpy()**0.5/nweight*scale)

    for i in range(len(modelA.model.layer2)):
        variation = 0
        nweight = 0
        nweight += get_size_scalar(modelA.model.layer2[i].conv1.weight)
        variation += torch.pow(torch.norm(modelA.model.layer2[i].conv1.weight.cpu() - modelB.model.layer2[i].conv1.weight.cpu(), 2),2)
        nweight += get_size_scalar(modelA.model.layer2[i].conv2.weight)
        variation += torch.pow(torch.norm(modelA.model.layer2[i].conv2.weight.cpu() - modelB.model.layer2[i].conv2.weight.cpu(), 2),2)
        nweight += get_size_scalar(modelA.model.layer2[i].conv3.weight)
        variation += torch.pow(torch.norm(modelA.model.layer2[i].conv3.weight.cpu() - modelB.model.layer2[i].conv3.weight.cpu(), 2),2)
        L1_varation.append(variation.detach().numpy()**0.5/nweight*scale)

    for i in range(len(modelA.model.layer3)):
        variation = 0
        nweight = 0
        nweight += get_size_scalar(modelA.model.layer3[i].conv1.weight)
        variation += torch.pow(torch.norm(modelA.model.layer3[i].conv1.weight.cpu() - modelB.model.layer3[i].conv1.weight.cpu(), 2),2)
        nweight += get_size_scalar(modelA.model.layer3[i].conv2.weight)
        variation += torch.pow(torch.norm(modelA.model.layer3[i].conv2.weight.cpu() - modelB.model.layer3[i].conv2.weight.cpu(), 2),2)
        nweight += get_size_scalar(modelA.model.layer3[i].conv3.weight)
        variation += torch.pow(torch.norm(modelA.model.layer3[i].conv3.weight.cpu() - modelB.model.layer3[i].conv3.weight.cpu(), 2),2)
        L1_varation.append(variation.detach().numpy()**0.5/nweight*scale)

    for i in range(len(modelA.model.layer4)):
        variation = 0
        nweight = 0
        nweight += get_size_scalar(modelA.model.layer4[i].conv1.weight)
        variation += torch.pow(torch.norm(modelA.model.layer4[i].conv1.weight.cpu() - modelB.model.layer4[i].conv1.weight.cpu(), 2),2)
        nweight += get_size_scalar(modelA.model.layer4[i].conv2.weight)
        variation += torch.pow(torch.norm(modelA.model.layer4[i].conv2.weight.cpu() - modelB.model.layer4[i].conv2.weight.cpu(), 2),2)
        nweight += get_size_scalar(modelA.model.layer4[i].conv3.weight)
        variation += torch.pow(torch.norm(modelA.model.layer4[i].conv3.weight.cpu() - modelB.model.layer4[i].conv3.weight.cpu(), 2),2)
        L1_varation.append(variation.detach().numpy()**0.5/nweight*scale)

    variation = 0
    nweight = 0
    nweight += get_size_scalarFC(modelA.classifier.add_block[0].weight)
    variation += torch.pow(torch.norm(modelA.classifier.add_block[0].weight.cpu() - modelB.classifier.add_block[0].weight.cpu(), 2),2)

    nweight += get_size_scalarFC(modelA.classifier.classifier[0].weight)
    variation += torch.pow(torch.norm(modelA.classifier.classifier[0].weight.cpu() - modelB.classifier.classifier[0].weight.cpu(), 2),2)
    L1_varation.append(variation.detach().numpy()**0.5/nweight* scale)

    return L1_varation



def compute_weight_variation_FC_cpu(modelA, modelB):
     #In the case of resnet, three layers are contained in one block
    # modelA = modelA.cpu()
    # modelB = modelB.cpu()
    total_variation = []
    nweight = get_size_scalar(modelA.model.conv1.weight)
    variation = torch.pow(torch.norm(modelA.model.conv1.weight.cpu() - modelB.model.conv1.weight.cpu(), 2),2)/nweight
    total_variation.append(variation.detach().numpy()*scale)

    for i in range(len(modelA.model.layer1)):
        variation = 0
        nweight = get_size_scalar(modelA.model.layer1[i].conv1.weight)
        variation += torch.pow(torch.norm(modelA.model.layer1[i].conv1.weight.cpu() - modelB.model.layer1[i].conv1.weight.cpu(), 2),2)/nweight
        nweight = get_size_scalar(modelA.model.layer1[i].conv2.weight)
        variation += torch.pow(torch.norm(modelA.model.layer1[i].conv2.weight.cpu() - modelB.model.layer1[i].conv2.weight.cpu(), 2),2)/nweight
        nweight = get_size_scalar(modelA.model.layer1[i].conv3.weight)
        variation += torch.pow(torch.norm(modelA.model.layer1[i].conv3.weight.cpu() - modelB.model.layer1[i].conv3.weight.cpu(), 2),2)/nweight
        total_variation.append(variation.detach().numpy()/n_*scale)

    #L2_varation = []
    for i in range(len(modelA.model.layer2)):
        variation = 0
        nweight = get_size_scalar(modelA.model.layer2[i].conv1.weight)
        variation += torch.pow(torch.norm(modelA.model.layer2[i].conv1.weight.cpu() - modelB.model.layer2[i].conv1.weight.cpu(), 2),2)/nweight
        nweight = get_size_scalar(modelA.model.layer2[i].conv2.weight)
        variation += torch.pow(torch.norm(modelA.model.layer2[i].conv2.weight.cpu() - modelB.model.layer2[i].conv2.weight.cpu(), 2),2)/nweight
        nweight = get_size_scalar(modelA.model.layer2[i].conv3.weight)
        variation += torch.pow(torch.norm(modelA.model.layer2[i].conv3.weight.cpu() - modelB.model.layer2[i].conv3.weight.cpu(), 2),2)/nweight
        total_variation.append(variation.detach().numpy()/n_*scale)

    #L3_varation = []
    for i in range(len(modelA.model.layer3)):
        variation = 0
        nweight = get_size_scalar(modelA.model.layer3[i].conv1.weight)
        variation += torch.pow(torch.norm(modelA.model.layer3[i].conv1.weight.cpu() - modelB.model.layer3[i].conv1.weight.cpu(), 2),2)/nweight
        nweight = get_size_scalar(modelA.model.layer3[i].conv2.weight)
        variation += torch.pow(torch.norm(modelA.model.layer3[i].conv2.weight.cpu() - modelB.model.layer3[i].conv2.weight.cpu(), 2),2)/nweight
        nweight = get_size_scalar(modelA.model.layer3[i].conv3.weight)
        variation += torch.pow(torch.norm(modelA.model.layer3[i].conv3.weight.cpu() - modelB.model.layer3[i].conv3.weight.cpu(), 2),2)/nweight
        total_variation.append(variation.detach().numpy()/n_*scale)

    #L4_varation = []

    for i in range(len(modelA.model.layer4)):
        variation = 0
        nweight = get_size_scalar(modelA.model.layer4[i].conv1.weight)
        variation += torch.pow(torch.norm(modelA.model.layer4[i].conv1.weight.cpu() - modelB.model.layer4[i].conv1.weight.cpu(), 2),2)/nweight
        nweight = get_size_scalar(modelA.model.layer4[i].conv2.weight)
        variation += torch.pow(torch.norm(modelA.model.layer4[i].conv2.weight.cpu() - modelB.model.layer4[i].conv2.weight.cpu(), 2),2)/nweight
        nweight = get_size_scalar(modelA.model.layer4[i].conv3.weight)
        variation += torch.pow(torch.norm(modelA.model.layer4[i].conv3.weight.cpu() - modelB.model.layer4[i].conv3.weight.cpu(), 2),2)/nweight
        total_variation.append(variation.detach().numpy()/n_*scale)

    variation = 0
    nweight = get_size_scalarFC(modelA.classifier.add_block[0].weight)
    variation += torch.pow(torch.norm(modelA.classifier.add_block[0].weight.cpu() - modelB.classifier.add_block[0].weight.cpu(), 2),2)/nweight

    nweight = get_size_scalarFC(modelA.classifier.classifier[0].weight)
    variation += torch.pow(torch.norm(modelA.classifier.classifier[0].weight.cpu() - modelB.classifier.classifier[0].weight.cpu(), 2),2)/nweight
    total_variation.append(variation.detach().numpy()/2* scale)

    return total_variation


def compute_weight_variation_FC_cpu_non_average(modelA, modelB):
    # In the case of resnet, three layers are contained in one block
    # modelA = modelA.cpu()
    # modelB = modelB.cpu()
    total_variation = []
    #nweight = get_size_scalar(modelA.model.conv1.weight)
    variation = torch.pow(torch.norm(modelA.model.conv1.weight.cpu() - modelB.model.conv1.weight.cpu(), 2), 2) #/ nweight
    total_variation.append(variation.detach().numpy() * scale)

    for i in range(len(modelA.model.layer1)):
        variation = 0
        #nweight = get_size_scalar(modelA.model.layer1[i].conv1.weight)
        variation += torch.pow(
            torch.norm(modelA.model.layer1[i].conv1.weight.cpu() - modelB.model.layer1[i].conv1.weight.cpu(), 2),
            2) #/ nweight
        #nweight = get_size_scalar(modelA.model.layer1[i].conv2.weight)
        variation += torch.pow(
            torch.norm(modelA.model.layer1[i].conv2.weight.cpu() - modelB.model.layer1[i].conv2.weight.cpu(), 2),
            2) #/ nweight
        #nweight = get_size_scalar(modelA.model.layer1[i].conv3.weight)
        variation += torch.pow(
            torch.norm(modelA.model.layer1[i].conv3.weight.cpu() - modelB.model.layer1[i].conv3.weight.cpu(), 2),
            2) #/ nweight
        total_variation.append(variation.detach().numpy() / n_ * scale)

    # L2_varation = []
    for i in range(len(modelA.model.layer2)):
        variation = 0
        #nweight = get_size_scalar(modelA.model.layer2[i].conv1.weight)
        variation += torch.pow(
            torch.norm(modelA.model.layer2[i].conv1.weight.cpu() - modelB.model.layer2[i].conv1.weight.cpu(), 2),
            2) #/ nweight
        #nweight = get_size_scalar(modelA.model.layer2[i].conv2.weight)
        variation += torch.pow(
            torch.norm(modelA.model.layer2[i].conv2.weight.cpu() - modelB.model.layer2[i].conv2.weight.cpu(), 2),
            2) #/ nweight
        #nweight = get_size_scalar(modelA.model.layer2[i].conv3.weight)
        variation += torch.pow(
            torch.norm(modelA.model.layer2[i].conv3.weight.cpu() - modelB.model.layer2[i].conv3.weight.cpu(), 2),
            2) #/ nweight
        total_variation.append(variation.detach().numpy() / n_ * scale)

    # L3_varation = []
    for i in range(len(modelA.model.layer3)):
        variation = 0
        #nweight = get_size_scalar(modelA.model.layer3[i].conv1.weight)
        variation += torch.pow(
            torch.norm(modelA.model.layer3[i].conv1.weight.cpu() - modelB.model.layer3[i].conv1.weight.cpu(), 2),
            2) #/ nweight
        #nweight = get_size_scalar(modelA.model.layer3[i].conv2.weight)
        variation += torch.pow(
            torch.norm(modelA.model.layer3[i].conv2.weight.cpu() - modelB.model.layer3[i].conv2.weight.cpu(), 2),
            2) #/ nweight
        #nweight = get_size_scalar(modelA.model.layer3[i].conv3.weight)
        variation += torch.pow(
            torch.norm(modelA.model.layer3[i].conv3.weight.cpu() - modelB.model.layer3[i].conv3.weight.cpu(), 2),
            2) #/ nweight
        total_variation.append(variation.detach().numpy() / n_ * scale)

    # L4_varation = []

    for i in range(len(modelA.model.layer4)):
        variation = 0
        #nweight = get_size_scalar(modelA.model.layer4[i].conv1.weight)
        variation += torch.pow(
            torch.norm(modelA.model.layer4[i].conv1.weight.cpu() - modelB.model.layer4[i].conv1.weight.cpu(), 2),
            2)
        #nweight = get_size_scalar(modelA.model.layer4[i].conv2.weight)
        variation += torch.pow(
            torch.norm(modelA.model.layer4[i].conv2.weight.cpu() - modelB.model.layer4[i].conv2.weight.cpu(), 2),
            2)
        #nweight = get_size_scalar(modelA.model.layer4[i].conv3.weight)
        variation += torch.pow(
            torch.norm(modelA.model.layer4[i].conv3.weight.cpu() - modelB.model.layer4[i].conv3.weight.cpu(), 2),
            2)
        total_variation.append(variation.detach().numpy() / n_ * scale)

    variation = 0
    #nweight = get_size_scalarFC(modelA.classifier.add_block[0].weight)
    variation += torch.pow(
        torch.norm(modelA.classifier.add_block[0].weight.cpu() - modelB.classifier.add_block[0].weight.cpu(), 2),
        2)

    #nweight = get_size_scalarFC(modelA.classifier.classifier[0].weight)
    variation += torch.pow(
        torch.norm(modelA.classifier.classifier[0].weight.cpu() - modelB.classifier.classifier[0].weight.cpu(), 2),
        2)
    total_variation.append(variation.detach().numpy() / 2 * scale)

    return total_variation
def get_graedient(modelA):

    C_gradient = []
    wdecay = 5e-4
    C_gradient.append(modelA.model.conv1.weight.grad.cpu()*vlr)

    L1_gradient = []
    for i in range(len(modelA.model.layer1)):
        L1_gradient.append(modelA.model.layer1[i].conv1.weight.grad.cpu()*vlr)
        L1_gradient.append(modelA.model.layer1[i].conv2.weight.grad.cpu()*vlr)
        L1_gradient.append(modelA.model.layer1[i].conv3.weight.grad.cpu()*vlr)

    L2_gradient = []
    for i in range(len(modelA.model.layer2)):
        L2_gradient.append(modelA.model.layer2[i].conv1.weight.grad.cpu()*vlr)
        L2_gradient.append(modelA.model.layer2[i].conv2.weight.grad.cpu()*vlr)
        L2_gradient.append(modelA.model.layer2[i].conv3.weight.grad.cpu()*vlr)

    L3_gradient = []
    for i in range(len(modelA.model.layer3)):
        L3_gradient.append(modelA.model.layer3[i].conv1.weight.grad.cpu()*vlr)
        L3_gradient.append(modelA.model.layer3[i].conv2.weight.grad.cpu()*vlr)
        L3_gradient.append(modelA.model.layer3[i].conv3.weight.grad.cpu()*vlr)

    L4_gradient = []
    for i in range(len(modelA.model.layer4)):
        L4_gradient.append(modelA.model.layer4[i].conv1.weight.grad.cpu()*vlr)
        L4_gradient.append(modelA.model.layer4[i].conv2.weight.grad.cpu()*vlr)
        L4_gradient.append(modelA.model.layer4[i].conv3.weight.grad.cpu()*vlr)

    return [C_gradient, L1_gradient, L2_gradient, L3_gradient, L4_gradient]


def get_averge_weight(modelA):

    wieght = 0
    demension = 0
    means = []
    means.append(modelA.model.conv1.weight.mean())
    for i in range(len(modelA.model.layer1)):
        tempmean =0
        tempmean += modelA.model.layer1[i].conv1.weight.mean()
        tempmean += modelA.model.layer1[i].conv2.weight.mean()
        tempmean += modelA.model.layer1[i].conv3.weight.mean()
        means.append(tempmean/3)

    for i in range(len(modelA.model.layer2)):
        tempmean =0
        tempmean += modelA.model.layer2[i].conv1.weight.mean()
        tempmean += modelA.model.layer2[i].conv2.weight.mean()
        tempmean += modelA.model.layer2[i].conv3.weight.mean()
        means.append(tempmean/3)

    for i in range(len(modelA.model.layer3)):
        tempmean =0
        tempmean += modelA.model.layer3[i].conv1.weight.mean()
        tempmean += modelA.model.layer3[i].conv2.weight.mean()
        tempmean += modelA.model.layer3[i].conv3.weight.mean()
        means.append(tempmean/3)

    for i in range(len(modelA.model.layer4)):
        tempmean =0
        tempmean += modelA.model.layer4[i].conv1.weight.mean()
        tempmean += modelA.model.layer4[i].conv2.weight.mean()
        tempmean += modelA.model.layer4[i].conv3.weight.mean()
        means.append(tempmean/3)

    sum(means) / len(means)

    return means


def get_total_averge_weight(modelA):

    demension = 0
    tempmean = 0
    tempmean += ((modelA.model.conv1.weight**2)**0.5).sum()
    demension += get_size_scalar(modelA.model.conv1.weight)
    for i in range(len(modelA.model.layer1)):
        tempmean += ((modelA.model.layer1[i].conv1.weight**2)**0.5).sum()
        tempmean += ((modelA.model.layer1[i].conv2.weight**2)**0.5).sum()
        tempmean += ((modelA.model.layer1[i].conv3.weight**2)**0.5).sum()
        demension += get_size_scalar(modelA.model.layer1[i].conv1.weight)
        demension += get_size_scalar(modelA.model.layer1[i].conv2.weight)
        demension += get_size_scalar(modelA.model.layer1[i].conv3.weight)

    for i in range(len(modelA.model.layer2)):
        tempmean += ((modelA.model.layer2[i].conv1.weight**2)**0.5).sum()
        tempmean += ((modelA.model.layer2[i].conv2.weight**2)**0.5).sum()
        tempmean += ((modelA.model.layer2[i].conv3.weight**2)**0.5).sum()
        demension += get_size_scalar(modelA.model.layer2[i].conv1.weight)
        demension += get_size_scalar(modelA.model.layer2[i].conv2.weight)
        demension += get_size_scalar(modelA.model.layer2[i].conv3.weight)

    for i in range(len(modelA.model.layer3)):
        tempmean += ((modelA.model.layer3[i].conv1.weight**2)**0.5).sum()
        tempmean += ((modelA.model.layer3[i].conv2.weight**2)**0.5).sum()
        tempmean += ((modelA.model.layer3[i].conv3.weight**2)**0.5).sum()
        demension += get_size_scalar(modelA.model.layer3[i].conv1.weight)
        demension += get_size_scalar(modelA.model.layer3[i].conv2.weight)
        demension += get_size_scalar(modelA.model.layer3[i].conv3.weight)

    for i in range(len(modelA.model.layer4)):
        tempmean += ((modelA.model.layer4[i].conv1.weight**2)**0.5).sum()
        tempmean += ((modelA.model.layer4[i].conv2.weight**2)**0.5).sum()
        tempmean += ((modelA.model.layer4[i].conv3.weight**2)**0.5).sum()
        demension += get_size_scalar(modelA.model.layer4[i].conv1.weight)
        demension += get_size_scalar(modelA.model.layer4[i].conv2.weight)
        demension += get_size_scalar(modelA.model.layer4[i].conv3.weight)
    totalmean = tempmean/demension

    return float(totalmean)

def sum_graedient(modelA, total_gradients):
    [C_gradient, L1_gradient, L2_gradient, L3_gradient, L4_gradient] = total_gradients
    C_gradient[0] += modelA.model.conv1.weight.grad.cpu()*mom

    for i in range(len(modelA.model.layer1)):
        L1_gradient[i*3] += modelA.model.layer1[i].conv1.weight.grad.cpu()*mom
        L1_gradient[i*3+1] += modelA.model.layer1[i].conv2.weight.grad.cpu()*mom
        L1_gradient[i*3+2] += modelA.model.layer1[i].conv3.weight.grad.cpu()*mom

    for i in range(len(modelA.model.layer2)):
        L2_gradient[i*3] += modelA.model.layer2[i].conv1.weight.grad.cpu()*mom
        L2_gradient[i*3+1] += modelA.model.layer2[i].conv2.weight.grad.cpu()*mom
        L2_gradient[i*3+2] += modelA.model.layer2[i].conv3.weight.grad.cpu()*mom

    for i in range(len(modelA.model.layer3)):
        L3_gradient[i*3] += modelA.model.layer3[i].conv1.weight.grad.cpu()*mom
        L3_gradient[i*3+1] += modelA.model.layer3[i].conv2.weight.grad.cpu()*mom
        L3_gradient[i*3+2] += modelA.model.layer3[i].conv3.weight.grad.cpu()*mom

    for i in range(len(modelA.model.layer4)):
        L4_gradient[i*3] += modelA.model.layer4[i].conv1.weight.grad.cpu()*mom
        L4_gradient[i*3+1] += modelA.model.layer4[i].conv2.weight.grad.cpu()*mom
        L4_gradient[i*3+2] += modelA.model.layer4[i].conv3.weight.grad.cpu()*mom

    return [C_gradient, L1_gradient, L2_gradient, L3_gradient, L4_gradient]

def gradient2variation_avg(model, total_gradients, weva_try, eta):


    # for i in range(len(total_gradients)):
    #     for j in range(len(total_gradients[i])):
    #         total_gradients[i][j] = total_gradients[i][j] * eta

    [C1, L1, L2, L3, L4] = total_gradients

    total_block_gradients = []
    nweight = get_size_scalar(model.model.conv1.weight)
    variation = torch.pow(torch.norm(C1[0]*vlr2,2),2)/nweight
    total_block_gradients.append(float(variation*scale))

    for i in range(len(model.model.layer1)):
        variation = 0
        nweight = get_size_scalar(model.model.layer1[i].conv1.weight)
        variation += torch.pow(torch.norm(L1[i*3]*vlr2,2),2)/nweight
        nweight = get_size_scalar(model.model.layer1[i].conv2.weight)
        variation += torch.pow(torch.norm(L1[i*3+1]*vlr2,2),2)/nweight
        nweight = get_size_scalar(model.model.layer1[i].conv3.weight)
        variation += torch.pow(torch.norm(L1[i*3+2]*vlr2,2),2)/nweight
        total_block_gradients.append(float(variation/n_*scale))

    for i in range(len(model.model.layer2)):
        variation = 0
        nweight = get_size_scalar(model.model.layer2[i].conv1.weight)
        variation += torch.pow(torch.norm(L2[i*3]*vlr2,2),2)/nweight
        nweight = get_size_scalar(model.model.layer2[i].conv2.weight)
        variation += torch.pow(torch.norm(L2[i*3+1]*vlr2,2),2)/nweight
        nweight = get_size_scalar(model.model.layer2[i].conv3.weight)
        variation += torch.pow(torch.norm(L2[i*3+2]*vlr2,2),2)/nweight
        total_block_gradients.append(float(variation/n_*scale))

    for i in range(len(model.model.layer3)):
        variation = 0
        nweight = get_size_scalar(model.model.layer3[i].conv1.weight)
        variation += torch.pow(torch.norm(L3[i * 3]*vlr2, 2), 2)/nweight
        nweight = get_size_scalar(model.model.layer3[i].conv2.weight)
        variation += torch.pow(torch.norm(L3[i * 3 + 1]*vlr2, 2), 2)/nweight
        nweight = get_size_scalar(model.model.layer3[i].conv3.weight)
        variation += torch.pow(torch.norm(L3[i * 3 + 2]*vlr2, 2), 2)/nweight
        total_block_gradients.append(float(variation/n_*scale))

    for i in range(len(model.model.layer4)):
        variation = 0
        nweight = get_size_scalar(model.model.layer4[i].conv1.weight)
        variation += torch.pow(torch.norm(L4[i * 3]*vlr2, 2), 2)/nweight
        nweight = get_size_scalar(model.model.layer4[i].conv2.weight)
        variation += torch.pow(torch.norm(L4[i * 3 + 1]*vlr2, 2), 2)/nweight
        nweight = get_size_scalar(model.model.layer4[i].conv3.weight)
        variation += torch.pow(torch.norm(L4[i * 3 + 2]*vlr2, 2), 2)/nweight
        total_block_gradients.append(float(variation/n_*scale))

    weva_ratio = total_block_gradients[:]
    # for i in range(len(total_block_gradients)):
    #     total_block_gradients[i] = total_block_gradients[i] * eta

    for i in range(len(weva_ratio)):
        weva_ratio[i] = weva_try[i]/(total_block_gradients[i]*0.001)
    return weva_ratio, total_block_gradients


def gradient2variation(model, total_gradients, weva_try, eta):


    for i in range(len(total_gradients)):
        for j in range(len(total_gradients[i])):
            total_gradients[i][j] = total_gradients[i][j] * eta

    [C1, L1, L2, L3, L4] = total_gradients

    total_block_gradients = []
    # nweight = get_size_scalar(model.model.conv1.weight)
    variation = torch.pow(torch.norm(C1[0]*vlr2,2),2)
    total_block_gradients.append(float(variation*scale))

    for i in range(len(model.model.layer1)):
        variation = 0
        #nweight = get_size_scalar(model.model.layer1[i].conv1.weight)
        variation += torch.pow(torch.norm(L1[i*3]*vlr2,2),2)
        #nweight = get_size_scalar(model.model.layer1[i].conv2.weight)
        variation += torch.pow(torch.norm(L1[i*3+1]*vlr2,2),2)
        #nweight = get_size_scalar(model.model.layer1[i].conv3.weight)
        variation += torch.pow(torch.norm(L1[i*3+2]*vlr2,2),2)
        total_block_gradients.append(float(variation/n_*scale))

    for i in range(len(model.model.layer2)):
        variation = 0
        #nweight = get_size_scalar(model.model.layer2[i].conv1.weight)
        variation += torch.pow(torch.norm(L2[i*3]*vlr2,2),2)
        #nweight = get_size_scalar(model.model.layer2[i].conv2.weight)
        variation += torch.pow(torch.norm(L2[i*3+1]*vlr2,2),2)
        #nweight = get_size_scalar(model.model.layer2[i].conv3.weight)
        variation += torch.pow(torch.norm(L2[i*3+2]*vlr2,2),2)
        total_block_gradients.append(float(variation/n_*scale))

    for i in range(len(model.model.layer3)):
        variation = 0
        #nweight = get_size_scalar(model.model.layer3[i].conv1.weight)
        variation += torch.pow(torch.norm(L3[i * 3]*vlr2, 2), 2)
        #nweight = get_size_scalar(model.model.layer3[i].conv2.weight)
        variation += torch.pow(torch.norm(L3[i * 3 + 1]*vlr2, 2), 2)
        #nweight = get_size_scalar(model.model.layer3[i].conv3.weight)
        variation += torch.pow(torch.norm(L3[i * 3 + 2]*vlr2, 2), 2)
        total_block_gradients.append(float(variation/n_*scale))

    for i in range(len(model.model.layer4)):
        variation = 0
        #nweight = get_size_scalar(model.model.layer4[i].conv1.weight)
        variation += torch.pow(torch.norm(L4[i * 3]*vlr2, 2), 2)
        #nweight = get_size_scalar(model.model.layer4[i].conv2.weight)
        variation += torch.pow(torch.norm(L4[i * 3 + 1]*vlr2, 2), 2)
        #nweight = get_size_scalar(model.model.layer4[i].conv3.weight)
        variation += torch.pow(torch.norm(L4[i * 3 + 2]*vlr2, 2), 2)
        total_block_gradients.append(float(variation/n_*scale))

    weva_ratio = total_block_gradients[:]
    # for i in range(len(total_block_gradients)):
    #     total_block_gradients[i] = total_block_gradients[i] * eta

    for i in range(len(weva_ratio)):
        weva_ratio[i] = weva_try[i]/(total_block_gradients[i]*0.001)
    return weva_ratio, total_block_gradients