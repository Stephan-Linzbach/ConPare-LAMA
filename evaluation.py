import itertools
import json
import os
import logging
from tqdm import tqdm
from timeit import default_timer as timer
import pickle
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
import torch
import numpy as np


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def tokenize(s):
    return tokenizer(s, padding=True, truncation=True, return_tensors='pt')


def get_allowed_letters(data):
    """
    Get the set of characters that are necessary to display the answers
    :param data: Containing subjects and objects
    :return: a set of characters needed to display subject and objects
    """
    allowed_letters = set()
    for p in data:
        for s, o in zip(data[p]['subjects'], data[p]['objects']):
            allowed_letters = allowed_letters.union(*[set(s), set(o)])
    return allowed_letters


def get_model_environment(model_name, allowed_letters):
    """
    All necessary model specific objects are created here
    :param model_name: Which huggingface model do we want to load?
    :param allowed_letters: The set of allowed letters - to exclude tokenizer specific pre-fixes
    :return: model-specific configurations
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).cuda().eval()

    mask = tokenizer.mask_token
    mask_id = tokenizer.mask_token_id

    vocab = {}
    all_letters = set().union(*[set(k) for k in tokenizer.get_vocab().keys()])
    inter = all_letters.intersection(allowed_letters)

    vocab_sorted = {k: v for k, v in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])}
    vocab_words = np.array(["".join([letter for letter in word if letter in inter]) for word in vocab_sorted.keys()])

    vocab = [False for _ in range(len(vocab_sorted.keys()))]
    for i, s in enumerate(vocab_words):
        if s in allowed_vocab:
            vocab[i] = True
    return tokenizer, model, mask, mask_id, vocab, vocab_words


def create_combined(temp, sub, mask, _domain, replace_domain, _range, replace_range):
    """
    Create template with combined information
    :param temp: template that is used for instantiation
    :param t: the name of the template
    :param sub: the subject that should be used to populate the template
    :param mask: the mask as a place-holder for the object
    :param _domain: the _domain constrain information that we want to replace
    :param replace_domain: the domain information that should be used to instantiate
    :param _range: the _range constrain information that we want to replace
    :param replace_range: the range information that should be used to instantiate
    :return: list of instantiated prompts
    """
    temps = []
    for r_d, r_r in zip(replace_domain, replace_range):
        t = temp.replace(_domain, r_d)
        t = t.replace(_range, r_r)
        t = t.replace("[X]", sub)
        t = t[0].upper() + t[1:]
        t = t.replace("[Y]", mask)
        temps.append(t)
    return temps


def create_sample(template, template_name, subject, mask, domain, poss_domain, range, poss_range):
    """
    Independent prompt instantiation
    :param template: the template of the prompt
    :param template_name: the name of the template (i.e., compound_domain, complex_range etc.)
    :param subject: subject of the triple
    :param mask: mask as placeholder for the object
    :param domain: domain placeholder
    :param poss_domain: all possible domain type constraints for the relation of the triple
    :param range: range placeholder
    :param poss_range: all possible range type constraints for the relation of the triple
    :return: list of instantiated prompts
    """
    result_templates = [template]

    if template_name in ['compound_domain', 'simple_domain']:
        result_templates += [template.replace(domain, p_d) for p_d in poss_domain]
    if template_name in ['complex_range', 'simple_range']:
        result_templates += [template.replace(range, p_r) for p_r in poss_range]

    results_t = []
    for t in result_templates:
        t = t.replace("[X]", subject)
        t = t[0].upper() + t[1:]
        t = t.replace("[Y]", mask)
        results_t.append(t)
    return results_t


def is_word_in_vocab(to_exclude, vocab, vocab_words, object):
    """
    Test if word is in the vocabulary for this it reduces the vocabulary and returns the id of the object
    :param to_exclude: set of tokens to exclude
    :param vocab: vocab vector
    :param vocab_words: vocab words
    :param object: object we are currently querying for
    :return: vocab fitting for the triple, the index of the object and indication if the object is in the vocab
    """
    vocab_exclude = vocab
    if to_exclude:
        vocab_exclude = torch.tensor(np.where(np.isin(vocab_words, to_exclude), False, vocab))
    try:
        o_id = np.where(vocab_words[vocab_exclude] == object)[0][0]
    except IndexError:
        return vocab_exclude, 0, False
    return vocab_exclude, o_id, True


@torch.no_grad()
def get_results(encodings, model, vocab_per_triple, masked_positions):
    """
    Return model results
    :param encodings: model specific encodings
    :param model: model that we currently test
    :param vocab_per_triple: used vocab
    :param masked_positions: position of the mask
    :return: probability distribution over the vocab for the masked token, and the hidden_states of the masked token
    """
    answers = model(**encodings, output_hidden_states=True)
    hidden_states = answers['hidden_states'][-1][masked_positions[0], masked_positions[1], :].cpu()
    logits = answers['logits'][masked_positions[0], masked_positions[1], :].cpu()
    logits = logits[:, vocab_per_triple]
    probability_distribution = torch.nn.Softmax(1)(logits)
    del answers
    return probability_distribution, hidden_states


def get_performance(probability_distribution, obj, reference_vocab, vocab_words, counts, hidden_states, performance, domain=None, range=None, opposing=None, completion=False):
    """
    Calculates several measurements over the results per object.
    :param probability_distribution: of the masked token
    :param obj: object that we query for
    :param reference_vocab: vocab that we use for this triple
    :param vocab_words: words of the vocab
    :param counts: information which vector belongs to what sentencec type
    :param hidden_states: hidden states of the masked tokens
    :param performance: already calculated performance
    :param domain: all possible domain constraints
    :param range: all possible range constraints
    :param opposing: information which sentence types carry the same information
    :param completion: should we apply the completion strategies
    :return: result dictionary with all measurements
    """
    sorted_args = torch.flip(torch.argsort(probability_distribution, dim=1), [1])

    end = timer()
    # print("flip", end - start)
    rel_vocab = vocab_words[vocab_per_triple]
    # print(rel_vocab.shape)
    ranking = torch.stack((sorted_args == obj).nonzero(as_tuple=True), dim=1)

    quality = probability_distribution[ranking[:, 0], sorted_args[ranking[:, 0], ranking[:, 1]]]
    entropy = torch.sum(probability_distribution * (-torch.log2(probability_distribution)), dim=1)
    confidence = torch.max(probability_distribution, dim=1)[0]

    for value, _tensor in {'entropy': entropy,
                           'quality': quality,
                           'distros': probability_distribution,
                           'hidden_states': hidden_states,
                           'confidence': confidence}.items():
        if not performance.get(value):
            performance[value] = {}.copy()

        for t, v in counts.items():
            performance[value][t] = _tensor[v[0]:v[1]].cpu()

    quality_information = {t: 0 for t in counts.keys()}
    quality_information_opposing = {t: 1 for t in counts.keys()}
    confidence_information = {t: 2 for t in counts.keys()}
    confidence_information_opposing = {t: 3 for t in counts.keys()}

    if completion:
        quality_information = {t: torch.argmax(performance['quality'][t]).item() for t in counts.keys()}
        quality_information_opposing = {t: torch.argmax(performance['quality'][opposing[t]]).item() for t in counts.keys()}
        confidence_information = {t: torch.argmax(performance['confidence'][t]).item() for t in counts.keys()}
        confidence_information_opposing = {t: torch.argmax(performance['confidence'][opposing[t]]).item() for t in counts.keys()}

        #logger.info(f"Confidence {confidence_information}, Range {range}")
        performance['confidence_range'] = {t: range[confidence_information[t]] for t in ['complex_range', 'simple_range']}
        performance['confidence_domain'] = {t: domain[confidence_information[t]] for t in ['compound_domain', 'simple_domain']}
        performance['confidence_distros'] = {t: performance['distros'][t][confidence_information[t]].cpu() for t in counts.keys()}
        performance['confidence_hidden_states'] = {t: performance['hidden_states'][t][confidence_information[t]:confidence_information[t]+1].cpu() for t in counts.keys()}

        performance['quality_range'] = {t: range[quality_information[t]] for t in ['complex_range', 'simple_range']}
        performance['quality_domain'] = {t: domain[quality_information[t]] for t in ['compound_domain', 'simple_domain']}
        performance['quality_distros'] = {t: performance['distros'][t][quality_information[t]].cpu() for t in counts.keys()}
        performance['quality_hidden_states'] = {t: performance['hidden_states'][t][quality_information[t]:quality_information[t]+1].cpu() for t in counts.keys()}

        if not performance.get(value):
            performance[value] = {}.copy()

    if not performance.get('confidence_entropy'):
        performance['confidence_entropy'] = {}
        performance['confidence_inverse_entropy'] = {}
        performance['quality_inverse_entropy'] = {}
        performance['quality_entropy'] = {}

    for t, v in counts.items():
        performance['confidence_entropy'][t] = performance['entropy'][t][confidence_information[t]].cpu()
        performance['confidence_inverse_entropy'][t] = performance['entropy'][t][confidence_information_opposing[t]].cpu()
        performance['quality_inverse_entropy'][t] = performance['entropy'][t][quality_information_opposing[t]].cpu()
        performance['quality_entropy'][t] = performance['entropy'][t][quality_information[t]].cpu()

    for k in [1, 10, 50, 100, 200, 500, 1000]:
        if k not in performance:
            performance[k] = {'quality_acc': {}, 'quality_inverse_acc': {}, 'confidence_acc': {}, 'confidence_inverse_acc': {}}.copy()
        start = timer()
        #performance[f"{k}_pred"].append([rel_vocab[i] for i in s_a[:k]])
        pred = [obj in s_a for s_a in sorted_args[:, :k]]
        for t, v in counts.items():
            performance[k]['quality_acc'][t] = pred[v[0]:v[1]][quality_information[t]]
            performance[k]['quality_inverse_acc'][t] = pred[v[0]:v[1]][quality_information_opposing[t]]
            performance[k]['confidence_acc'][t] = pred[v[0]:v[1]][confidence_information[t]]
            performance[k]['confidence_inverse_acc'][t] = pred[v[0]:v[1]][confidence_information_opposing[t]]

    performance = {k: v if not torch.is_tensor(v) else v.cpu() for k, v in performance.items()}
    return performance


# Change tested models here

models = {'bert': "bert-base-cased",
          'roberta': "roberta-base",
          'luke': "studio-ousia/luke-base"}

# Change tested dataset here

dataset = "GoogleRE"

# Change GPU here

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


with open(f"./data/_{dataset}_data.json") as f:
    data = json.load(f)

with open("./data/common_vocab_cased.txt") as f:
    allowed_vocab = set([x[:-1] for x in f.readlines()])

with open(f"./data/exclusion_information_{dataset}.json") as f:
    exclusion_information = json.load(f)
    exclusion_information = {k: {kk: set(vv) for kk, vv in v.items()} for k, v in exclusion_information.items()}

results = {}

mapping = {'compound': 'compound_domain',
           'simple_compound': 'simple_domain',
           'complex': 'complex_range',
           'simple_complex': 'simple_range',
           'simple_compound_complex': 'simple_combined'}

template_independent_instantiation = ['simple',
                                      'compound_domain',
                                      'simple_domain',
                                      'complex_range',
                                      'simple_range']

template_combined = ['compound_complex',
                     'simple_combined']

_domain_info = ['compound_domain', 'simple_domain']
_range_info = ['complex_range', 'simple_range']

opposing = {'simple': 'simple',
            'compound_domain': 'simple_domain',
            'complex_range': 'simple_range',
            'compound_complex': 'simple_combined',
            'simple_domain': 'compound_domain',
            'simple_range': 'complex_range',
            'simple_combined': 'compound_complex'}

for name, m in models.items():
    logger.info(f"Evaluating performance of {name}")
    tokenizer, model, mask, mask_id, vocab, vocab_words = get_model_environment(m,
                                                                                get_allowed_letters(data))
    results[m] = {}
    for relation in data:
        results[m][relation] = {}
        results[m][relation]['subjects'] = []
        for s, o in tqdm(list(zip(data[relation]['subjects'], data[relation]['objects']))):
            to_exclude = exclusion_information[relation][s].difference({o})
            vocab_per_triple, o_id, is_in_vocab = is_word_in_vocab(to_exclude=to_exclude,
                                                                   vocab=vocab,
                                                                   vocab_words=vocab_words,
                                                                   object=o)

            if not is_in_vocab:
                logger.info(f"Skipping {o} for {name}")
                results[m][relation]['subjects'].append(False)
                continue


            ## Indepent Instances
            _domain = data[relation]['domain']
            _possible_domain = data[relation]['possible_domain'] if 'possible_domain' in data[relation] else []
            _range = data[relation]['range']
            _possible_range = data[relation]['possible_range'] if 'possible_range' in data[relation] else []

            sentences = [create_sample(template=data[relation][t],
                                       template_name=t,
                                       subject=s,
                                       mask=mask,
                                       domain=_domain,
                                       poss_domain=_possible_domain,
                                       range=_range,
                                       poss_range=_possible_range)
                                for t in
                                        template_independent_instantiation]

            counts = {t: (sum(len(sentences[j]) for j in range(0, i)),
                          sum(len(sentences[j]) for j in range(0, i+1)))
                      for i, t in enumerate(template_independent_instantiation)}

            #logger.info(sentences)

            sentences = list(itertools.chain(*sentences))

            results[m][relation]['domain'] = [_domain] + _possible_domain
            results[m][relation]['range'] = [_range] + _possible_range

            encodings = {k: v.cuda() for k, v in tokenize(sentences).items()}
            masked_positions = (encodings['input_ids'] == mask_id).nonzero(as_tuple=True)

            probability_distribution, hidden_states = get_results(encodings, model, vocab_per_triple, masked_positions)

            performance = get_performance(probability_distribution,
                                                 o_id,
                                                 vocab_per_triple,
                                                 vocab_words,
                                                 counts,
                                                 hidden_states,
                                                 {}.copy(),
                                                 results[m][relation]['domain'],
                                                 results[m][relation]['range'],
                                                 opposing,
                                                 True)
            performance['object'] = o
            performance['subject'] = s
            #results[m][relation]['subjects'].append(performance)

            # Combined Prompts
            sentences = [create_combined(data[relation][t],
                                        s,
                                        mask,
                                        data[relation]['domain'],
                                        [performance['quality_domain'][d],
                                         performance['quality_domain'][opposing[d]],
                                         performance['confidence_domain'][d],
                                         performance['confidence_domain'][opposing[d]]],
                                        data[relation]['range'],
                                        [performance['quality_range'][r],
                                         performance['quality_range'][opposing[r]],
                                         performance['confidence_range'][r],
                                         performance['confidence_range'][opposing[r]]]) for t, r, d in
                        zip(template_combined, _range_info, _domain_info)]

            counts = {t: (sum(len(sentences[j]) for j in range(0, i)), sum(len(sentences[j]) for j in range(0, i + 1)))
                      for i, t in enumerate(template_combined)}

            sentences = list(itertools.chain(*sentences))

            encodings = {k: v.cuda() for k, v in tokenize(sentences).items()}
            masked_positions = (encodings['input_ids'] == mask_id).nonzero(as_tuple=True)

            probability_distribution, hidden_states = get_results(encodings, model, vocab_per_triple, masked_positions)

            new_performance = get_performance(probability_distribution,
                                                     o_id,
                                                     vocab_per_triple,
                                                     vocab_words,
                                                     counts,
                                                     hidden_states,
                                                     performance)

            del new_performance['quality_distros'], new_performance['confidence_distros']
            del new_performance['quality_hidden_states'], new_performance['confidence_hidden_states']
            del new_performance['distros'], new_performance['hidden_states'],
            results[m][relation]['subjects'].append(new_performance)

        acc = {}
        template_names = template_independent_instantiation + template_combined

        for k in [1, 10, 50, 100, 200, 500, 1000]:
            acc[k] = {}
            for accs in ['quality_acc', 'quality_inverse_acc', 'confidence_acc', 'confidence_inverse_acc']:
                acc[k][accs] = {}
                for t in template_names:
                    acc[k][accs][t] = {}
                    acc[k][accs][t]['correct'] = sum([s[k][accs][t] for s in results[m][relation]['subjects'] if s])
                    acc[k][accs][t]['total'] = len([x for x in results[m][relation]['subjects'] if x])
        results[m][relation]['accuracy'] = acc.copy()
        results[m][relation]['total'] = len([x for x in results[m][relation]['subjects'] if x])
    del model
pickle.dump(results, open(f"./{dataset}_results.p", "wb"))