from allen_vit_pipeline.pipeline import Config, EIDRepository

config = Config('three_session_A', 'natural_movie_one')

eids = EIDRepository(config).get_eids_to_process()
print(eids)
print(len(eids))
