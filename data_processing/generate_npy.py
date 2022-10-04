import copy
import csv
import glob
import os
import argparse
import sys
from collections import defaultdict
from PIL import Image
import pickle as pkl
import pickle
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import random
from multiprocessing import Pool
import re


def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


def squash(path): # squash from 480x640 to 128x128 and flattened as a tensor
    im = Image.open(path)
    im = im.resize((128,128), Image.ANTIALIAS)
    out = np.asarray(im).astype(np.uint8)
    return out

def process_images(path): #processes images at a trajectory level
    names = sorted([x for x in os.listdir(path) if 'images' in x], key = lambda x: int(x.split('images')[1]))
    image_path = [os.path.join(path, x) for x in os.listdir(path) if 'images' in x]
    image_path = sorted(image_path, key=lambda x: int(x.split('images')[1]))

    images_out = defaultdict(list)
    
    tlen = len(glob.glob(image_path[0] + '/im_*.jpg'))

    for i, name in enumerate(names):
        for t in range(tlen):
            images_out[name].append(squash(image_path[i] + '/im_{}.jpg'.format(t)))

    images_out = dict(images_out)

    obs, next_obs = dict(), dict()

    for n in names:
        obs[n] = images_out[n][:-1]
        next_obs[n] = images_out[n][1:]
    return obs, next_obs

def process_state(path):
    fp = os.path.join(path, 'obs_dict.pkl')
    x = pickle.load(open(fp, 'rb'))  
    return x['full_state'][:-1], x['full_state'][1:]

def process_actions(path): # gets actions
    fp = os.path.join(path, 'policy_out.pkl')
    act_list = pickle.load(open(fp, 'rb'))
    if isinstance(act_list[0], dict):
        act_list = [x['actions'] for x in act_list]
    return act_list

def process_dc(args, path, train_ratio=0.9): #processes each data collection attempt
    # print(path)
    if 'lmdb' in path:
        return [], [], [], []

    all_dicts_train = list()
    all_dicts_test = list()
    all_rews_train= list()
    all_rews_test = list()

    print('processing path ', path)
    try:
        task_name, date_time = path.split('/')[-2:]
        date = date_time.split('_')[0]
        year_month_day_number = int(date.split('-')[0] + date.split('-')[1] + date.split('-')[2])
    except:
        print('folder name does not contain valid date!')
        return [], [], [], []
    # Data collected prior to 7-23 has a delay of 2, otherwise a delay of 1
    if year_month_day_number > 20210723:  # date  2021 / 7 / 23
        latency_shift = False
    else:
        latency_shift = True

    search_path = os.path.join(path, 'raw', 'traj_group*','traj*')
    all_traj = glob.glob(search_path)
    if all_traj == []:
        print('no trajs found in ', search_path)
        return [], [], [], []

    if args.only_use_first_n != -1:
        all_traj = sorted_nicely(all_traj)
        all_traj = all_traj[:args.only_use_first_n]

    random.shuffle(all_traj)

    num_traj = len(all_traj)
    print('num_traj ', num_traj)
    for itraj, tp in tqdm(enumerate(all_traj)):
        try:
            out = dict()

            ld = os.listdir(tp)

            assert 'obs_dict.pkl' in ld,  tp + ':' + str(ld)
            assert 'policy_out.pkl' in  ld,  tp + ':' + str(ld)
            assert 'agent_data.pkl' in  ld,  tp + ':' + str(ld)

            obs, next_obs = process_images(tp)
            acts = process_actions(tp)
            state, next_state = process_state(tp)
            term = [0] * len(acts)

            out['observations'] = obs
            out['observations']['state'] = state
            out['next_observations'] = next_obs
            out['next_observations']['state'] = next_state

            out['observations'] = [dict(zip(out['observations'],t)) for t in zip(*out['observations'].values())]
            out['next_observations'] = [dict(zip(out['next_observations'],t)) for t in zip(*out['next_observations'].values())]

            out['actions'] = acts

            # shift the actions according to camera latency

            out['terminals'] = term
            if latency_shift:
                out['observations'] = out['observations'][1:]
                out['next_observations'] = out['next_observations'][1:]
                out['actions'] = out['actions'][:-1]
                out['terminals'] = term[:-1]

            labeled_rew = copy.deepcopy(out['terminals'])[:]
            labeled_rew[-2:] = [1, 1]

            traj_len = len(out['observations'])
            assert len(out['next_observations']) == traj_len
            assert len(out['actions']) == traj_len
            assert len(out['terminals']) == traj_len
            assert len(labeled_rew) == traj_len

            if itraj < int(num_traj * train_ratio):
                all_dicts_train.append(out)
                all_rews_train.append(labeled_rew)
            else:
                all_dicts_test.append(out)
                all_rews_test.append(labeled_rew)
        except FileNotFoundError as e:
            print(e)
            continue
        except AssertionError as e:
            print(e)
            continue

    return all_dicts_train, all_dicts_test, all_rews_train, all_rews_test

def make_numpy(tuple): # overarching npy creation
    print('thread', os.getpid())
    args, path, outpath = tuple
    outpath_train = os.path.join(outpath, 'train')
    outpath_val = os.path.join(outpath, 'val')
    Path(outpath_train).mkdir(parents=True, exist_ok=True)
    Path(outpath_val).mkdir(parents=True, exist_ok=True)
    lst_train = list()
    lst_val = list()
    rew_train_l = list()
    rew_val_l = list()
    print(path)

    for dated_folder in os.listdir(path):
        curr_train, curr_val, rew_train, rew_val = process_dc(args, os.path.join(path,dated_folder))
        lst_train.extend(curr_train)
        lst_val.extend(curr_val)
        rew_train_l.extend(rew_train)
        rew_val_l.extend(rew_val)
    np.save(os.path.join(outpath_train, 'out.npy'), lst_train)
    np.save(os.path.join(outpath_val, 'out.npy'), lst_val)
    print('saved', os.path.join(outpath_train, 'out.npy'))

    np.save(os.path.join(outpath_train, 'out_rew.npy'), rew_train_l)
    np.save(os.path.join(outpath_val, 'out_rew.npy'), rew_val_l)
    print('saved', os.path.join(outpath_train, 'out_rew.npy'))

    print('task {} number of train traj {}'.format(outpath, len(lst_train)))

def get_outpath(input_path, output_path, task_path):
    return os.path.join(output_path, task_path.split(input_path)[1].strip("/"))

def update_domain_task_dict(domain_task_dict, taskpath, extracted_domain_task_dict):
    task_name = str.split(taskpath, '/')[-1]
    domain_name = str.split(taskpath, '/')[-2]
    entity_name = str.split(taskpath, '/')[-3]

    traj_list = glob.glob(taskpath + '/*/*/*/traj*')

    if entity_name not in domain_task_dict:
        domain_task_dict[entity_name] = {domain_name: {}}

    if domain_name in domain_task_dict[entity_name]:
        domain_task_dict[entity_name][domain_name][task_name] = len(traj_list)
    else:
        domain_task_dict[entity_name][domain_name] = {task_name: len(traj_list)}

    reextract = True
    try:
        if domain_task_dict[entity_name][domain_name][task_name] == extracted_domain_task_dict[entity_name][domain_name][task_name]:
            reextract = False
        else:
            print(f'{entity_name} {domain_name} {task_name}  found folders {domain_task_dict[entity_name][domain_name][task_name]}, existing extracted: {extracted_domain_task_dict[entity_name][domain_name][task_name]}')
    except:
        reextract = True
    if reextract:
        print(f'{entity_name} {domain_name} {task_name}  found folders {domain_task_dict[entity_name][domain_name][task_name]}')
    return reextract



def process_entity(args, entity_dir, tps, entity_domain_task_dict, extracted_domain_task_dict):
    for domain in os.listdir(entity_dir):  # domain directory, e.g. toykitchen1
        domain_path = os.path.join(entity_dir, domain)
        for task in os.listdir(domain_path):  # task directory
            task_path = os.path.join(domain_path, task)
            if 'lmdb' in task_path: continue
            output_path = get_outpath(args.input, args.output, task_path)
            extract = update_domain_task_dict(entity_domain_task_dict, task_path, extracted_domain_task_dict)
            if args.exclude_existing:
                if not os.path.exists(output_path):
                    tps.append((task_path, output_path))
                else:
                    print('skipping because it exists: ', output_path)
            elif extracted_domain_task_dict is not None:
                if extract:
                    tps.append((args, task_path, output_path))
            else:
                tps.append((args, task_path, output_path))

def write_entity_domain_task_dict_csv_pkl(dict, outputpath):
    with open(outputpath + '/dataset_summary.csv', 'w', encoding='UTF8') as f:
        # create the csv writer
        writer = csv.writer(f)
        writer.writerow(['task number','entity', 'domain', 'task', 'number of demos'])
        total_demos = 0
        num_tasks = 0
        for entity in dict.keys():
            for domain in dict[entity].keys():
                for task in dict[entity][domain].keys():
                    if dict[entity][domain][task] > 0:
                        num_tasks += 1
                        writer.writerow([num_tasks, entity, domain, task, dict[entity][domain][task]])
                        total_demos += dict[entity][domain][task]

        writer.writerow(['total demos', total_demos])
        print('total num demos', total_demos)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="path to raw data", required=True)
    parser.add_argument("--output", help="path to datafiles", required=True)
    parser.add_argument("--exclude_existing", help="", action='store_true')
    parser.add_argument("--allent", help="process all entities, e.g. berkely, upenn", action='store_true')
    parser.add_argument("--only_use_first_n", default=-1, help="only use first n, if -1 use all", type=int)
    args = parser.parse_args()

    entity_domain_task_dict = {}

    extracted_domain_task_csv = args.output + '/dataset_summary.csv'
    if os.path.exists(extracted_domain_task_csv):
        extracted_domain_task_dict = parse_csv(extracted_domain_task_csv)
    else:
        if not input('dataset summary pkl does not exist, shall everthing be re-etracted? y/n') == 'y':
            sys.exit()
        extracted_domain_task_dict = None


    tps = []
    if args.allent:
        for entity in os.listdir(args.input): # e.g. berkeley upenn
            fd = os.path.join(args.input, entity)
            process_entity(args, fd, tps, entity_domain_task_dict, extracted_domain_task_dict)
    else:
        process_entity(args, args.input, tps, entity_domain_task_dict, extracted_domain_task_dict)

    print('number of tasks to be processed', len(tps))
    print(entity_domain_task_dict)

    with Pool(10) as pool:
        pool.map(make_numpy, tps)
    write_entity_domain_task_dict_csv_pkl(entity_domain_task_dict, outputpath=args.output)

    # for task in tps:
    #     # if 'take_carrot_off_plate_cardboardfence' in task[1]:
    #     make_numpy(task)
    # write_entity_domain_task_dict_csv_pkl(entity_domain_task_dict, outputpath=args.output)



def parse_csv(extracted_domain_task_dict):
    rows = []
    with open(extracted_domain_task_dict, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            rows.append(str.split(row[0], ','))
    rows = rows[1:]
    _entity_domain_task_dict = {}
    for row in rows:
        if len(row) != 5:
            continue
        entity = row[1]
        domain = row[2]
        task = row[3]
        num_traj = int(row[4])
        if entity not in _entity_domain_task_dict:
            _entity_domain_task_dict[entity] = {}
        if domain not in _entity_domain_task_dict[entity]:
            _entity_domain_task_dict[entity][domain] = {}
        _entity_domain_task_dict[entity][domain][task] =  num_traj
    return _entity_domain_task_dict


if __name__ == "__main__":
    run()


