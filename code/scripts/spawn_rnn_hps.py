from scripts.spawn_hp_cluster_jobs import JobSpawner

spawner = JobSpawner(namespace="extschell")

groups = [
    ("rnn-sr", 0),
    ("rnn-br", 0),
    ("rnn-brv", 5),
]

for group, num_jobs in groups:
    spawner.spawn(num_jobs,
                  template="job_template.jinja2.yaml",
                  template_parameters={
                      "job_name": group,
                      "hps_name": f"stage_1_{group.replace('-', '_')}",
                      "image_tag": "latest",
                      "cpus": 8,
                      "memory": 24,
                      "use_gpu": True,
                      "priority_class": "research-med",
                      "script": "run.py"
                  })
