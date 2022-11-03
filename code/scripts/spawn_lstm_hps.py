from scripts.spawn_hp_cluster_jobs import JobSpawner

spawner = JobSpawner(namespace="extschell")

groups = [
    ("lstm-brv", 0),
    ("lstm-br", 0),
    ("lstm-sr", 10),
]

for group, num_jobs in groups:
    spawner.spawn(num_jobs,
                  template="job_template.jinja2.yaml",
                  template_parameters={
                      "job_name": group,
                      "hps_name": f"stage_1_{group.replace('-', '_')}",
                      "image_tag": "latest",
                      "cpus": 8,
                      "memory": 20,
                      "use_gpu": True,
                      "priority_class": "research-low",
                      "script": "run.py"
                  })
