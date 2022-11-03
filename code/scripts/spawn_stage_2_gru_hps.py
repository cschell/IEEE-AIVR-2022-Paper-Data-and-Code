from scripts.spawn_hp_cluster_jobs import JobSpawner

spawner = JobSpawner(namespace="extschell")

groups = [
    ("gru-sr", 0),
    ("gru-br", 0),
    ("gru-brv", 9)
]

for group, num_jobs in groups:
    spawner.spawn(num_jobs,
                  template="job_template.jinja2.yaml",
                  template_parameters={
                      "job_name": group + "-data",
                      "hps_name": f"stage_2_{group.replace('-', '_')}",
                      "image_tag": "latest",
                      "cpus": 8,
                      "memory": 24,
                      "use_gpu": True,
                      "priority_class": "research-low",
                      "script": "run.py"
                  })
