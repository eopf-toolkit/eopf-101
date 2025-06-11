# eopf-toolkit-jupterhub-k8s

The initial setup of the JupyterHub k8s cluster was done by following the JupyterHub for Kubernetes documentation here: https://z2jh.jupyter.org/en/latest/jupyterhub/installation.html

# Steps Carried Out:

## Created a new namespace

`eopf-tooklit-dev` with `resources/0-eopf-toolkit-namespace.yml`

## Installed the JupyterHub Helm Chart

```
helm upgrade --cleanup-on-fail \
  --install 0.0.1 jupyterhub/jupyterhub \
  --namespace eopf-toolkit-dev \
  --version=4.2.0 \
  --values config.yaml \
  --kubeconfig=kubeconfig.yml
```

## Created a .ds.io record

Service is now available at eopf-toolkit-dev.ds.io

## Enabled HTTPS with LetsEncrypt

Updated `config.yaml` with:

```
proxy:
  https:
    enabled: true
    hosts:
      - eopf-toolkit-dev.ds.io
    letsencrypt:
      contactEmail: ciaran@developmentseed.org
```

And applied changes:

```
helm upgrade --kubeconfig=kubeconfig.yml --cleanup-on-fail \
  0.0.1 jupyterhub/jupyterhub \
  --namespace eopf-toolkit-dev \
  --version=4.2.0 \
  --values config.yaml
```

## Added Github OAuth access

Firstly created a new Github OAuth application within the `eopf-toolkit` org

Updated `config.yaml` with:

```
hub:
  config:
    GitHubOAuthenticator:
      client_id: <id>
      client_secret: <secret>
      oauth_callback_url: https://eopf-toolkit-dev.ds.io/hub/oauth_callback
    JupyterHub:
      authenticator_class: github
```

Run:

```
helm upgrade --kubeconfig=kubeconfig.yml --cleanup-on-fail \
  0.0.1 jupyterhub/jupyterhub \
  --namespace eopf-toolkit-dev \
  --version=4.2.0 \
  --values config.yaml
```

## Limit to `eopf-toolkit` org for OAuth

Modified config.yaml with:

```
allowed_organizations:
  - my-github-organization
scope:
  - read:org
```

In the `GitHubOAuthenticator` block, then ran:

```
helm upgrade --kubeconfig=kubeconfig.yml --cleanup-on-fail \
  0.0.1 jupyterhub/jupyterhub \
  --namespace eopf-toolkit-dev \
  --version=4.2.0 \
  --values config.yaml
```

## Made Ciaran an admin user

Updated `hub.config.Authenticator` with:

```
admin_users:
  - ciaransweet
```

Ran:

```
helm upgrade --kubeconfig=kubeconfig.yml --cleanup-on-fail \
  0.0.1 jupyterhub/jupyterhub \
  --namespace eopf-toolkit-dev \
  --version=4.2.0 \
  --values config.yaml
```

## Updated image for users to include more packages

This involved taking the https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html#jupyter-datascience-notebook image and installing any packages not included that weren't in the list in the ticket _or_ in the environment provided in the CNG workshop

Full list of extra installed packages is in the `requirements.txt` file.

The `Dockerfile` and `Makefile` can be used to create and push a new image to the private registry in OVH which the cluster will use for single users.

The tag pushed must be updated in the `tag` section of the `singleUsers` configuration in `config.yaml`

## Manually created a secret for GitHub OAuth values

I created `github-outh-secret` manually on the cluster and modified `config.yaml` so this is injected into the hub service. Saves having these accidentally commited in the `config.yaml` file.
