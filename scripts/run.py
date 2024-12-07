from allen_vit_pipeline.pipeline import Config, EIDRepository, STAProcessEID

config = Config('three_session_A', 'natural_movie_one')

eids = EIDRepository(config).get_eids_to_process()
print(eids)
print(len(eids))

'''

processor=STAProcessEID(config)

import time
start=time.time()
for eid in eids:
    processor(eid)
end=time.time()
print(end-start)'''