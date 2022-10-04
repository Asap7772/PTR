import os

train_dataset_single_task = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen1/put_broccoli_in_pot_cardboardfence/train/out.npy']
eval_dataset_single_task = [os.environ['DATA'] + '/robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen1/put_broccoli_in_pot_cardboardfence/val/out.npy']


train_dataset_single_task_openmicro = ['robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen5/open_microwave/train/out.npy']
eval_dataset_single_task_openmicro = ['robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen5/open_microwave/val/out.npy']


train_tasks = val_tasks = [
    'put_broccoli_in_pot_cardboardfence',
    'put_carrot_on_plate_cardboardfence',
    'put_broccoli_in_pot_or_pan',
    'put_broccoli_in_bowl',
    'put_carrot_on_plate',
    'put_sushi_on_plate',
    'put_corn_into_bowl',
    'put_sweet_potato_in_pan_which_is_on_stove',
    'put_sweet_potato_in_pan_which_is_on_stove_distractors',
    'put_sweet_potato_in_pot_which_is_in_sink_distractors',

    'take_broccoli_out_of_pan_cardboardfence',
    'take_carrot_off_plate_cardboardfence',
    'take_broccoli_out_of_pan',
    'take_can_out_of_pan',
    'take_carrot_off_plate',
    'take_lid_off_pot_or_pan',
]

# train_dataset_11_task = [f'robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8/toykitchen1/{task}/train/out.npy' for task in train_tasks]
train_dataset_11_task = [f'robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen1/{task}/train/out.npy' for task in train_tasks]

# eval_dataset_11_task = [f'robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8/toykitchen1/{task}/val/out.npy' for task in val_tasks]
eval_dataset_11_task = [f'robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen1/{task}/val/out.npy' for task in val_tasks]

train_tasks_tk5 = val_tasks_tk5 = [
    'close_fridge',
    'close_microwave',
    'open_fridge',
    'open_microwave',
    'open_cabinet',
    'open_low_fridge',
    'open_oven',
    'close_cabinet',
    'close_low_fridge',
    'close_oven',
]

train_tasks_tk6 = val_tasks_tk6 = [
    'close_microwave',
    'open_microwave',
    'open_oven',
    'close_oven',
    'open_fridge',
    'close_fridge',
]


train_dataset_openclose = [f'robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen5/{task}/train/out.npy' for task in train_tasks_tk5]
train_dataset_openclose.extend([f'robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen1/{task}/train/out.npy' for task in train_tasks_tk6])
train_dataset_openclose.extend([f'robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen6/{task}/train/out.npy' for task in train_tasks_tk6])
eval_dataset_openclose = [f'robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen5/{task}/val/out.npy' for task in train_tasks_tk5]
eval_dataset_openclose.extend([f'robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen1/{task}/val/out.npy' for task in train_tasks_tk6])
eval_dataset_openclose.extend([f'robonetv2/toykitchen_numpy_shifted/bridge_data_numpy_shifted_split_uint8_5-26/toykitchen6/{task}/val/out.npy' for task in train_tasks_tk6])

ALIASING_DICT = {
        "flip_pot_upright_in_sink_distractors": "flip_pot_upright_which_is_in_sink",
        "put_eggplant_into_pan": "put_eggplant_in_pot_or_pan",
        "put_eggplant_into_pot_or_pan": "put_eggplant_in_pot_or_pan",
        "faucet_front_to_left": "turn_faucet_front_to_left",
        "put_cup_from_counter_or_drying_rack_into_sink": "put_cup_from_anywhere_into_sink",
        "put_green_squash_into_pot_or_pan": "put_green_squash_in_pot_or_pan",
        "turn_lever_vertical_to-front": "turn_lever_vertical_to_front",
        "turn_lever_vertical_to_front_distractors": "turn_lever_vertical_to_front",
        "put_pan_from_sink_into_drying_rack": "put_pot_or_pan_from_sink_into_drying_rack",
        "put_corn_in_pan_which_is_on_stove_distractors": "put_corn_into_pot_or_pan",
        "put_corn_in_pan_which-is_on_stove_distractors": "put_corn_into_pot_or_pan",
        "put_corn_in_pot_which_is_in_sink_distractors": "put_corn_into_pot_or_pan",
        "take_broccoli_out_of_pan": "take_broccoli_out_of_pot_or_pan",
        "put_pepper_in_pan": "put_pepper_in_pot_or_pan",
        "put_sweet_potato_in_pot_which_is_in_sink_distractors": "put_sweet_potato_in_pot",
        "put_sweet_potato_in_pan_which_is_on_stove_distractors": "put_sweet_potato_in_pan_which_is_on_stove",
        "put_pan_in_sink": "put_pot_or_pan_in_sink",
        "put_pot_in_sink": "put_pot_or_pan_in_sink",
        "put_pan_from_stove_to_sink": "put_pot_or_pan_in_sink",
        "put_pot_on_stove_which_is_near_stove_distractors": "put_pot_or_pan_on_stove",
        "put_pan_on_stove_from_sink": "put_pot_or_pan_on_stove",

        "put_broccoli_in_pot_cardboardfence": "put_broccoli_in_pot_or_pan",
        "put_carrot_on_plate_cardboardfence": "put_carrot_on_plate",
        "take_broccoli_out_of_pan_cardboardfence": "take_broccoli_out_of_pot_or_pan",
        "take_carrot_off_plate_cardboardfence": "take_carrot_off_plate",

        'open_cabinet': 'open_door',
        'open_oven': 'open_door',
        'open_low_fridge': 'open_door',
        'open_fridge': 'open_door',
        'open_microwave': 'open_door',

        'close_oven': 'close_door',
        'close_fridge': 'close_door',
        'close_cabinet': 'close_door',
        'close_low_fridge': 'close_door',
        'close_microwave': 'close_door'
}

# data from online reaching
online_reaching_pixels_first100 = ['/robonetv2/online_datacollection/extract/online_reaching_first100/toykitchen1/pix_policy_std0.3_initscale0.0_collectdata/train/out.npy']
online_reaching_pixels_val_first100 = ['/robonetv2/online_datacollection/extract/online_reaching_first100/toykitchen1/pix_policy_std0.3_initscale0.0_collectdata/val/out.npy']
online_reaching_pixels= ['/robonetv2/online_datacollection/extract/online_reaching/toykitchen1/pix_policy_std0.3_initscale0.0_collectdata/train/out.npy']
online_reaching_pixels_val= ['/robonetv2/online_datacollection/extract/online_reaching/toykitchen1/pix_policy_std0.3_initscale0.0_collectdata/val/out.npy']

