def get_task_id_mapping(task_folders, task_aliasing_dict=None, index=-3):
    task_descriptions = set()
    for task_folder in task_folders:
        task_description = str.split(task_folder, '/')[index]
        if task_aliasing_dict and task_description in task_aliasing_dict:
            task_description = task_aliasing_dict[task_description]
        task_descriptions.add(task_description)
    task_descriptions = sorted(task_descriptions)
    task_dict = {task_descp: index for task_descp, index in
            zip(task_descriptions, range(len(task_descriptions)))}
    print ('Printing task descriptions..............')
    for idx, desc in task_dict.items():
        print (idx, ' : ', desc)
    print ('........................................')
    return task_dict

def exclude_tasks(paths, excluded_tasks):
    new_paths = []
    for d in paths:
        reject = False
        for exdir in excluded_tasks:
            if exdir in d:
                # print('excluding', d)
                reject = True
                break
        if not reject:
            new_paths.append(d)
    return new_paths

def include_tasks(paths, included_tasks):
    new_paths = []
    for d in paths:
        accept = False
        for exdir in included_tasks:
            if exdir in d:
                accept = True
                break
        if accept:
            new_paths.append(d)
    return new_paths
