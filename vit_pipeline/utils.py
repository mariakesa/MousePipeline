import pandas as pd
import os
import Path

def make_container_dict(boc):
    '''
    Parses which experimental id's (values)
    correspond to which experiment containers (keys).
    '''
    experiment_container = boc.get_experiment_containers()
    container_ids = [dct['id'] for dct in experiment_container]
    eids = boc.get_ophys_experiments(experiment_container_ids=container_ids)
    df = pd.DataFrame(eids)
    reduced_df = df[['id', 'experiment_container_id', 'session_type']]
    grouped_df = reduced_df.groupby(['experiment_container_id', 'session_type'])[
        'id'].agg(list).reset_index()
    eid_dict = {}
    for row in grouped_df.itertuples(index=False):
        container_id, session_type, ids = row
        if container_id not in eid_dict:
            eid_dict[container_id] = {}
        eid_dict[container_id][session_type] = ids[0]
    return eid_dict

def get_existing_containers(config):
    path=str(Path(config.allen_cache_path) / Path('ophys_experiment_events'))
    # List all files in the directory
    filenames = os.listdir(path)
    containers=[]
    for f in filenames:
        parsed=f.split('_')
        container_id=int(parsed[0])
        containers.append(container_id)
    return containers
