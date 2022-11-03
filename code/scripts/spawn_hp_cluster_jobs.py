import json
import pathlib
import time

import typer
import yaml
from jinja2 import Environment, FileSystemLoader

import kubernetes


class JobSpawner:
    def __init__(self, namespace):
        self.namespace = namespace
        typer.echo("importing kubernetes libs, this may take a while")

        self.client = kubernetes.client
        typer.echo("load kube config")
        kubernetes.config.load_kube_config()
        typer.echo("initialising kubernetes client")
        self.core_v1 = kubernetes.client.BatchV1Api(kubernetes.client.ApiClient())

    def _create_job_config_from_template(self, template, parameters):
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        JOB_CONFIG_PATH = "_job_configs/%s" % timestamp

        pathlib.Path(JOB_CONFIG_PATH).mkdir(parents=True, exist_ok=True)

        job_config_raw = Environment(loader=FileSystemLoader("")).get_template(template).render(**parameters)

        with open("%s/%s.yml" % (JOB_CONFIG_PATH, parameters["job_name"]), "w") as job_config_file:
            job_config_file.write(job_config_raw)

        job_config = yaml.load(job_config_raw, Loader=yaml.Loader)

        return job_config

    def spawn(self, num_jobs: int, template, template_parameters):
        typer.echo("start spawn loop")

        for job_idx in range(num_jobs):
            copied_template_parameters =    template_parameters.copy()
            copied_template_parameters["job_name"] = f"{copied_template_parameters['job_name']}-{job_idx}"

            job_config = self._create_job_config_from_template(template, copied_template_parameters)

            try:
                resp = self.core_v1.create_namespaced_job(body=job_config, namespace=self.namespace)
            except kubernetes.client.exceptions.ApiException as e:

                body = json.loads(e.body)
                if body.get("reason") == "AlreadyExists":
                    typer.echo(f"job {copied_template_parameters['job_name']} already exists")
                    continue
                else:
                    typer.echo(body["message"])
                    raise
            job_name = resp.metadata.name
            typer.echo("job %s created" % job_name)


if __name__ == "__main__":
    spawner = JobSpawner(namespace="extschell")
    spawner.spawn(1,
                  template="job_template.jinja2.yaml",
                  template_parameters={
                      "job_name": "gru-brv-hps",
                      "hps_name": "stage_1_gru_brv_test",
                      "image_tag": "latest",
                      "cpus": 15,
                      "memory": 15,
                      "use_gpu": True,
                      "priority_class": "research-low",
                  })
