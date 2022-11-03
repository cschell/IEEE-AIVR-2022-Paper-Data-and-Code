from scripts.spawn_hp_cluster_jobs import JobSpawner

spawner = JobSpawner(namespace="extschell")

groups = [
    ("rf-sr", 0),
    ("rf-br", 0),
    ("rf-brv", 3),
]

for group, num_jobs in groups:
    spawner.spawn(num_jobs,
                  template="job_template.jinja2.yaml",
                  template_parameters={
                      "job_name": group,
                      "hps_name": f"stage_1_{group.replace('-', '_')}",
                      "image_tag": "latest",
                      "cpus": 8,
                      "memory": 30,
                      "use_gpu": False,
                      "priority_class": "research-low",
                      "script": "run_sklearn.py"
                  })
