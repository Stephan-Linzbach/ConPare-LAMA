import json
import pickle
import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
from supervenn import supervenn
from collections import OrderedDict

# Change GPU here
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Change Dataset here
dataset = 'TREx'

print(dataset)
with open(f"./data/_{dataset}_data.json") as f:
    data = json.load(f)

with open(f"./data      /cardinality_information_{dataset}.json") as f:
    card = json.load(f)

categories = list(set(card.values()))
print(categories)
if '1:1' in categories:
    card_order = {'1:1': [], 'N:1': [], 'N:M': [] } #, 'Total':[]}
else:
    card_order = {c : [] for c in categories}

for c in card:
    card_order[card[c]].append(c)
    #card_order['Total'].append(c)


template_names = ['simple',
                  'compound_domain',
                  'simple_domain',
                  'complex_range',
                  'simple_range',
                  'compound_complex',
                  'simple_combined']

models = {'bert_both': "bert-base-cased",
           'roberta': "roberta-base",
          'luke': "studio-ousia/luke-base",
          }

# total and per cardinality

# appendix per relation
# performance accuracy
# ranking correlation
# performance graph with additional information no information --> one information averaged --> both information
# wasserstein distance between opposing for
# average isotropy for different subjects averaged over seperated addtional information

opposing = {'compound_domain': 'simple_domain',
            'complex_range': 'simple_range',
            'compound_complex': 'simple_combined'}


def calc_accuracy(accuracy, acc_type):
    """
    Calculate the accuracy of the comparable subset
    :param accuracy: the data that contains all accuracy information
    :param acc_type: which accuracy should be looked at (i.e., which completion strategy)
    :return: appositive and causal accuracy
    """
    import matplotlib.pyplot as plt
    template_typologies = ['simple', 'compound_domain', 'complex_range', 'compound_complex']
    template_control = ['simple', 'simple_domain', 'simple_range', 'simple_combined']

    complete_acc = {}
    accuracy_clausal = {m: [] for m in accuracy.keys()}
    accuracy_appositive = {m: [] for m in accuracy.keys()}
    support = []
    total_all = {m : {t: 0 for t in template_names} for m in accuracy.keys()}.copy()
    correct_all = {m : {t: 0 for t in template_names} for m in accuracy.keys()}.copy()

    for m in accuracy.keys():
        cardis = list(accuracy[m].keys())
        #if not 'total' in cardis:
        #    cardis.append('total')
        for k in cardis:  # + ['total']:
            correct = {t: 0 for t in template_names}.copy()
            total = {t: 0 for t in template_names}.copy()
            for a in accuracy[m][k]:

                for t in template_names:
                    values = a[1][acc_type].copy()
                    correct[t] += values[t]['correct']
                    total[t] += values[t]['total']
                    total_all[m][t] += values[t]['total']
                    correct_all[m][t] += values[t]['correct']

            #xlabels = ['relation', 'relation+1', 'relation+2']
            _accuracy_clausal = [correct[t] / total[t] for t in template_typologies]
            accuracy_clausal[m].append(_accuracy_clausal)
            support.append(total['simple'])
            _accuracy_appositive = [correct[t] / total[t] for t in template_control]
            accuracy_appositive[m].append(_accuracy_appositive)

    for m in accuracy.keys():
        _accuracy_clausal = [correct_all[m][t] / total_all[m][t] for t in template_typologies]
        accuracy_clausal[m].append(_accuracy_clausal)
        support.append(total_all[m]['simple'])
        _accuracy_appositive = [correct_all[m][t] / total_all[m][t] for t in template_control]
        accuracy_appositive[m].append(_accuracy_appositive)

    return accuracy_clausal, accuracy_appositive, support

def experiment_one(accuracy, acc_type):
    """
    Get result table for experiment one
    :param accuracy: accuracy information
    :param acc_type: accuracy of which completion strategy
    :return: print out the table in latex format
    """
    typologies, control, support = calc_accuracy(accuracy, acc_type)
    print(support)
    if '1:1' in categories:
        cardis = ['1:1', 'N:1', 'N:M', 'Total']
    else:
        cardis = categories
        if not 'Total' in cardis:
            cardis.append('Total')
    models = ['bert_both', 'roberta', 'luke']
    lines = []
    print(cardis)
    print("Grouping & Simple B & Simple R & Simple L & Cpnd B & Appo B & Cpnd R & Appo R & Cpnd L & Appo L & Cplx B & Appo B & Cplx R & Appo R & Cplx L & Appo L \\\\")
    for j, c in enumerate(cardis):
        line = f"{c} & "
        for i in [0, 1, 2]:
            if i == 0:
                line += " & ".join([str(typologies[m][j][i])[1:6] for m in models]) + " & "
            else:
                line += " & ".join([f"{str(typologies[m][j][i])[1:6]} & {str(control[m][j][i])[1:6]}" for m in models]) + " & "
        lines.append(line + "\\\\")

    for l in lines:
        print(l)

def calc_acc(appositive, clausal):
    """
    Calculate the accuracy
    :param appositive: dict with information about the appositives
    :param clausal: dict with information about the clausal
    :return: reduced version of the two dicts
    """
    for m in clausal:
        for c in clausal[m]:
            for r in clausal[m][c]:
                clausal[m][c][r] = sum(clausal[m][c][r]) / len(clausal[m][c][r])
    for m in appositive:
        for c in appositive[m]:
            for r in appositive[m][c]:
                if r == 'relation':
                    continue
                appositive[m][c][r] = sum(appositive[m][c][r]) / len(appositive[m][c][r])
    return clausal, appositive

def combination_accuracy(subjects, acc_type='confidence_acc', prob_type='confidence'):
    """
    Information per subject of the current corpora is displayed in the table.
    :param subjects: subject wise informaiton
    :param acc_type: which completion types accuracy is considered
    :param prob_type: which probability type is considered
    :return: returns the combined accuracy
    """
    if '1:1' in categories:
        cardis = ['1:1', 'N:1', 'N:M', 'Total']
    else:
        cardis = categories
        if not 'Total' in cardis:
            cardis.append('Total')
    import matplotlib.pyplot as plt
    template_typologies = ['simple', 'compound_domain', 'complex_range', 'compound_complex']
    template_control = ['simple', 'simple_domain', 'simple_range', 'simple_combined']
    information_gain = ['relation', 'relation+1', 'relation+2']

    acc_combined = {m: {c: {t: [] for t in information_gain}.copy() for c in cardis} for m in subjects.keys()}.copy()
    acc_combined_control = {m: {c: {t: [] for t in information_gain}.copy() for c in cardis} for m in subjects.keys()}.copy()

    for m in subjects:
        for c in subjects[m]:
            for s in subjects[m][c]:
                acc_combined[m][c]['relation'].append(s[1][acc_type]['simple'])
                acc_combined[m]['Total']['relation'].append(s[1][acc_type]['simple'])

                if torch.max(s[prob_type]['compound_domain']) > torch.max(s[prob_type]['complex_range']):
                    acc_combined[m][c]['relation+1'].append(s[1][acc_type]['compound_domain'])
                    acc_combined[m]['Total']['relation+1'].append(s[1][acc_type]['compound_domain'])
                else:
                    acc_combined[m][c]['relation+1'].append(s[1][acc_type]['complex_range'])
                    acc_combined[m]['Total']['relation+1'].append(s[1][acc_type]['complex_range'])

                if torch.max(s[prob_type]['simple_domain']) > torch.max(s[prob_type]['simple_range']):
                    acc_combined_control[m][c]['relation+1'].append(s[1][acc_type]['simple_domain'])
                    acc_combined_control[m]['Total']['relation+1'].append(s[1][acc_type]['simple_domain'])
                else:
                    acc_combined_control[m][c]['relation+1'].append(s[1][acc_type]['simple_range'])
                    acc_combined_control[m]['Total']['relation+1'].append(s[1][acc_type]['simple_range'])

                acc_combined[m][c]['relation+2'].append(s[1][acc_type]['compound_complex'])
                acc_combined[m]['Total']['relation+2'].append(s[1][acc_type]['compound_complex'])

                acc_combined_control[m][c]['relation+2'].append(s[1][acc_type]['simple_combined'])
                acc_combined_control[m]['Total']['relation+2'].append(s[1][acc_type]['simple_combined'])

    return acc_combined, acc_combined_control

def experiment_two(subjects, acc_type='confidence_acc', prob_type='confidence'):
    """
    Display the complete tables used to generate results in experiment two
    :param subjects: subject wise information
    :param acc_type: completion strategy accuracy
    :param prob_type: combine by different porbabilities
    :return:
    """
    typologies, control = combination_accuracy(subjects, acc_type=acc_type, prob_type=prob_type)
    if '1:1' in categories:
        cardis = ['1:1', 'N:1', 'N:M', 'Total']
    else:
        cardis = categories
    models = ['bert_both', 'roberta', 'luke']
    typologies, control = calc_acc(control, typologies)
    lines = []
    print(typologies)
    for j, c in enumerate(cardis):
        line = f"{c} & "
        for i in ['relation', 'relation+1', 'relation+2']:
            if i == 'relation':
                line += " & ".join([str(typologies[m][c][i])[1:6] for m in models]) + " & "
            else:
                line += " & ".join([f"{str(typologies[m][c][i])[1:6]} & {str(control[m][c][i])[1:6]}" for m in models]) + " & "
        lines.append(line + "\\\\")

    for l in lines:
        print(l)

def prompt_recall(subjects, acc_type):
    """
    Different Recalls on the simple set
    :param subjects: subject wise information
    :param acc_type: performance for different completion strategy
    :return:
    """
    #acc_type = 'inverse_acc'
    correct_information = {m: {t: set() for t in template_names}.copy() for m in subjects}

    i = 0
    for m in subjects:
        for k in subjects[m]:
            for s in subjects[m][k]:
                for t in correct_information[m]:
                    if s[1][acc_type][t]:
                        correct_information[m][t].add(i)
                i += 1
    return correct_information

def experiment_three(subjects, acc_type):
    """
    Create the supervenn for the subjects
    :param subjects: subject wise information
    :param acc_type: performance for different completion strategy
    :return:
    """
    lines = []
    real_names = {'simple': 'simple',
                  'simple_domain': 'domain control',
                  'compound_domain': 'compound_domain',
                  'simple_range': 'range control',
                  'complex_range': 'complex_range',
                  'simple_combined': 'combined control',
                  'compound_complex': 'compound_domain complex_range'}
    correct_information = prompt_recall(subjects, acc_type)
    for m in correct_information:
        range_or_domain_specific = set().union(*[correct_information[m]['compound_domain'], correct_information[m]['complex_range']])
        range_or_domain_control = set().union(*[correct_information[m]['simple_domain'], correct_information[m]['simple_range']])
        colors_specific = ['#D79B00', '#6C8EBF', '#B85450', '#9673A6']
        colors_control = ['#D79B00', '#D79B00','#D79B00','#D79B00']
        display_information_specific = OrderedDict({'simple': correct_information[m]['simple'],
                                                    'domain': correct_information[m]['compound_domain'],
                                                    'range': correct_information[m]['complex_range'],
                                                    'combined': correct_information[m]['compound_complex'],})
        display_information_control = OrderedDict({
                               'simple': correct_information[m]['simple'],
                               'domain': correct_information[m]['simple_domain'],
                               'range': correct_information[m]['simple_range'],
                               'combined': correct_information[m]['simple_combined'],})
        #plt.figure(figsize=(20, 10))
        """chunks_ordering = 'occurrence'
        supervenn(list(display_information_specific.values()), list(display_information_specific.keys()), widths_minmax_ratio=0.1,
                  sets_ordering=None, chunks_ordering=chunks_ordering, rotate_col_annotations=True, col_annotations_area_height=2, fontsize=25, color_cycle=colors_specific)
        plt.ylabel('', fontsize=1)
        plt.xlabel('Correctly Classified Triples', fontsize=30)
        plt.savefig(f'{dataset}_{m}_{acc_type}_{chunks_ordering}_specific_supervenn_two.png')
        plt.clf()
        plt.figure(figsize=(20, 10))
        chunks_ordering = 'occurrence'
        supervenn(list(display_information_control.values()), list(display_information_control.keys()), widths_minmax_ratio=0.1,
                  sets_ordering=None, chunks_ordering=chunks_ordering, rotate_col_annotations=True, col_annotations_area_height=2, fontsize=25, color_cycle=colors_control)
        plt.ylabel('', fontsize=1)
        plt.xlabel('Correctly Classified Triples', fontsize=30)
        plt.savefig(f'{dataset}_{m}_{acc_type}_{chunks_ordering}_control_supervenn_two.png')
        plt.clf()"""

        plt.figure(figsize=(20, 10))
        chunks_ordering = 'size'
        supervenn(list(display_information_specific.values()), list(display_information_specific.keys()), widths_minmax_ratio=0.1,
                  sets_ordering=None, chunks_ordering=chunks_ordering, rotate_col_annotations=True, col_annotations_area_height=2, fontsize=25, color_cycle=colors_specific)
        plt.ylabel('', fontsize=1)
        plt.xlabel('Correctly Retrieved Triples', fontsize=30)
        plt.savefig(f'{dataset}_{m}_{acc_type}_{chunks_ordering}_clausal_supervenn.png')
        plt.clf()
        plt.figure(figsize=(20, 10))
        chunks_ordering = 'size'
        supervenn(list(display_information_control.values()), list(display_information_control.keys()), widths_minmax_ratio=0.1,
                  sets_ordering=None, chunks_ordering=chunks_ordering, rotate_col_annotations=True, col_annotations_area_height=2, fontsize=25, color_cycle=colors_control)
        plt.ylabel('', fontsize=1)
        plt.xlabel('Correctly Retrieved Triples', fontsize=30)
        plt.savefig(f'{dataset}_{m}_{acc_type}_{chunks_ordering}_appositive_supervenn.png')
        plt.clf()
    for name in correct_information:
        ground_truth = correct_information[name]['simple']
        line = name + " & "
        for t in template_names[1:]:
            recall = str(len(correct_information[name][t].intersection(ground_truth)) / len(ground_truth))[1:6]
            line += f" {recall} &"
        line += f" {len(ground_truth)} \\\\"
        lines.append(line)

    for l in lines:
        print(l)


def plot_entropy(overview_clausal, overview_appositive):
    """
    plot the entropy in a 2 x 2 formate
    :param overview_clausal: ov
    :param overview_appositive:
    :return:
    """
    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': True})
    xlabels = ['relation', 'range \nor domain', 'combined']
    fig, axs = plt.subplots(2,2, sharex='col', sharey='all')
    right_name = {'bert_both': 'BERT',
                  'roberta': 'RoBERTa',
                  'luke': 'Luke'}

    for m in overview_clausal:
        axs[0, 0].plot(range(len(xlabels)), [np.mean(overview_clausal[m]['confidence_entropy']['Total'][r]) for r in overview_clausal[m]['confidence_entropy']['Total']],
                       label = f"{right_name[m]} - N = {len(overview_clausal[m]['quality_entropy']['Total']['relation'])}")
        axs[0, 0].set_ylabel("Confidence \n Completion \n Average Entropy")
        axs[0, 0].set_title("Clausal Syntax")
        axs[0, 0].grid(True)
        axs[0, 0].legend()
    for m in overview_clausal:
        axs[0, 1].plot(range(len(xlabels)), [np.mean(overview_appositive[m]['confidence_entropy']['Total'][r]) for r in overview_appositive[m]['confidence_entropy']['Total']])
        axs[0, 1].set_title("Appositive Syntax")
        axs[0, 1].grid(True)
    for m in overview_clausal:
        axs[1, 0].plot(range(len(xlabels)), [np.mean(overview_clausal[m]['quality_entropy']['Total'][r]) for r in overview_clausal[m]['quality_entropy']['Total']])
        axs[1, 0].set_ylabel("Quality \n Completion \n Average Entropy")
        axs[1, 0].set_xticks(range(len(xlabels)), xlabels)
        axs[1, 0].set_xticks(range(len(xlabels)), xlabels, rotation=0)
        axs[1, 0].grid(True)
    for m in overview_clausal:
        axs[1, 1].plot(range(len(xlabels)), [np.mean(overview_appositive[m]['quality_entropy']['Total'][r]) for r in overview_appositive[m]['quality_entropy']['Total']])
        axs[1, 1].set_xticks(range(len(xlabels)), xlabels, rotation=0)
        axs[1, 1].grid(True)
    plt.savefig(f"{dataset}_response_confidence_given_information_quality.png")
    plt.show()

def experiment_four(subjects, acc_types, entropy_types):
    template_typologies = ['simple', 'compound_domain', 'complex_range', 'compound_complex']
    template_control = ['simple', 'simple_domain', 'simple_range', 'simple_combined']
    relation_amount = {'simple': 'relation',
                       'compound_domain': 'relation+1',
                       'complex_range': 'relation+1',
                       'compound_complex': 'relation+2',
                       'simple_domain': 'relation+1',
                       'simple_range': 'relation+1',
                       'simple_combined': 'relation+2'}
    relation_amounts = ['relation', 'relation+1', 'relation+2']
    if '1:1' in categories:
        specific_templates_right = {m : {entropy_type :
                                        {'1:1': {t: [] for t in relation_amounts}.copy(),
                                         'N:1': {t: [] for t in relation_amounts}.copy(),
                                         'N:M': {t: [] for t in relation_amounts}.copy(),
                                         'Total': {t: [] for t in relation_amounts}.copy()}.copy()
                                    for entropy_type in entropy_types}
                               for m in subjects}
        unspecific_templates_right = {m : {entropy_type :
                                        {'1:1': {t: [] for t in relation_amounts}.copy(),
                                         'N:1': {t: [] for t in relation_amounts}.copy(),
                                         'N:M': {t: [] for t in relation_amounts}.copy(),
                                         'Total': {t: [] for t in relation_amounts}.copy()}.copy()
                                    for entropy_type in entropy_types}
                               for m in subjects}
    else:
        specific_templates_right = {m : {entropy_type :
                                        {c: {t: [] for t in relation_amounts}.copy()
                                         for c in categories}.copy()
                                    for entropy_type in entropy_types}
                               for m in subjects}
        unspecific_templates_right = {m : {entropy_type :
                                        {c: {t: [] for t in relation_amounts}.copy()
                                         for c in categories}.copy()
                                    for entropy_type in entropy_types}
                               for m in subjects}

    for m in subjects:
        for c in subjects[m]:
            for s in subjects[m][c]:
                if all([s[10][acc_type]['simple'] for acc_type in acc_types]):
                    for entropy_type in entropy_types:
                        for t in template_names:
                            if t == 'simple':
                                specific_templates_right[m][entropy_type][c][relation_amount[t]].append(s[entropy_type][t].item())
                                specific_templates_right[m][entropy_type]['Total'][relation_amount[t]].append(s[entropy_type][t].item())
                                unspecific_templates_right[m][entropy_type][c][relation_amount[t]].append(s[entropy_type][t].item())
                                unspecific_templates_right[m][entropy_type]['Total'][relation_amount[t]].append(s[entropy_type][t].item())
                            elif t in template_typologies:
                                specific_templates_right[m][entropy_type][c][relation_amount[t]].append(s[entropy_type][t].item())
                                specific_templates_right[m][entropy_type]['Total'][relation_amount[t]].append(s[entropy_type][t].item())
                            else:
                                unspecific_templates_right[m][entropy_type][c][relation_amount[t]].append(s[entropy_type][t].item())
                                unspecific_templates_right[m][entropy_type]['Total'][relation_amount[t]].append(s[entropy_type][t].item())

    plot_entropy(specific_templates_right, unspecific_templates_right)

def make_results_comparable(r, comparison_results):
    new_subs = []
    new_accuracy = {}
    for i, s in enumerate(r['subjects']):
        if all([c_r[i] for c_r in comparison_results]):
            for k in [1, 10, 50, 100, 200, 500, 1000]:
                if not new_accuracy.get(k):
                    new_accuracy[k] = {}
                for acc_type in s[k]:
                    if not new_accuracy[k].get(acc_type):
                        new_accuracy[k][acc_type] = {}
                    for t in s[k][acc_type]:
                        if not new_accuracy[k][acc_type].get(t):
                            new_accuracy[k][acc_type][t] = []
                        new_accuracy[k][acc_type][t].append(s[k][acc_type][t])
            new_subs.append(s)
        else:
            new_subs.append(False)
    r['accuracy'] = {k: {acc_type:
                             {t: {'correct': sum(subs), 'total': len(subs)}
                                for t, subs in subs_per.items()}
                                for acc_type, subs_per in k_at_acc.items()}
                     for k, k_at_acc in new_accuracy.items()}
    r['subjects'] = new_subs
    return r


def load_data():
    res_dict = {}
    acc_dict = {}
    rel_dict = {}
    results = pickle.load(
        open(f"./{dataset}_results.p", "rb"))
    for name, m in models.items():
        if '1:1' in categories:
            res_dict[name] = {'1:1': [], 'N:1': [], 'N:M': []}.copy()
            acc_dict[name] = {'1:1': [], 'N:1': [], 'N:M': []}.copy()
        else:
            res_dict[name] = {c: [] for c in categories}.copy()
            acc_dict[name] = {c: [] for c in categories}.copy()

        rel_dict[name] = {}

        for k, relations in card_order.items():
            for p in relations:
                comparison = []
                for comp, comp_path in models.items():
                    if name == comp:
                        continue
                    comparison.append(results[comp_path][p]['subjects'])
                results[m][p] = make_results_comparable(results[m][p], comparison)

                acc_dict[name][k].append(results[m][p]['accuracy'])
                rel_dict[name][p] = results[m][p]['subjects']
                res_dict[name][k] += results[m][p]['subjects']
                """results = {s: {k : num for k, num in v.items() if k in [1, 10, 50, 100, 200, 500, 1000]}
                           for s, v in results.items()}"""
            # subjects.update(results)
    for name in res_dict:
        res_dict[name] = {k: [x for x in v if x] for k, v in res_dict[name].items()}
    return res_dict, acc_dict, rel_dict, results

print("Getting comparable results...")
res_dict, acc_dict, rel_dict, results = load_data()

print("EXPERIMENT 1")

print("Completion 1")
experiment_one(acc_dict, 'quality_acc')

print("Completion 2")
experiment_one(acc_dict, 'confidence_acc')

print("EXPERIMENT 2")

print("Completion 1")
experiment_two(res_dict, acc_type='quality_acc', prob_type='quality')

print("Completion 1 Combination 2")
experiment_two(res_dict, acc_type='quality_acc')

print("Completion 2")
experiment_two(res_dict)

print("EXPERIMENT 3")

print("Completion 1")
experiment_three(res_dict, acc_type='quality_acc')
print("Completion 2")
experiment_three(res_dict, acc_type='confidence_acc')

print("EXPERIMENT 4")

print("Completion 1")
experiment_four(res_dict, ['quality_acc', 'confidence_acc'], ['quality_entropy', 'confidence_entropy'])
