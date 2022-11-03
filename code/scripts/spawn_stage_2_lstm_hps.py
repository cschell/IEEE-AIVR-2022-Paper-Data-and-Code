from scripts.spawn_hp_cluster_jobs import JobSpawner

spawner = JobSpawner(namespace="extschell")

groups = [
    ("lstm-sr", 2),
    ("lstm-br", 0),
    ("lstm-brv", 0)
]

for group, num_jobs in groups:
    spawner.spawn(num_jobs,
                  template="job_template.jinja2.yaml",
                  template_parameters={
                      "job_name": group + "-data",
                      "hps_name": f"stage_2_{group.replace('-', '_')}",
                      "image_tag": "latest",
                      "cpus": 8,
                      "memory": 30,
                      "use_gpu": True,
                      "priority_class": "research-med",
                      "script": "run.py"
                  })
