# Master Thesis
[![License](https://img.shields.io/github/license/tomdewildt/master-thesis)](https://github.com/tomdewildt/master-thesis/blob/master/LICENSE)

Code for my thesis project of my master's degree.

# How To Run

Prerequisites:
* terraform version ```1.4.6``` or later
* virtualenv version ```20.0.3``` or later
* python version ```3.8.5``` or later

### Development

1. Run ```make init``` to initialize the environment.
2. Run ```make notebook``` to start the notebook server.

### Infrastructure

1. Run ```make init``` to initialize the environment.
2. Run ```make deploy/plan``` to create the deployment plan.
3. Run ```make deploy/apply``` to apply the deployment plan.

Run ```make deploy/destroy``` to destroy the deployment.

# References

[Jupyter Docs](https://jupyter.org/documentation)

[Terraform Docs](https://developer.hashicorp.com/terraform/docs)

[Terraform Language Docs](https://developer.hashicorp.com/terraform/language)

[Terraform Azure Docs](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs)
