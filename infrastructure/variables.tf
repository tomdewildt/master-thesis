variable "namespace" {
  type        = string
  description = "The namespace used for all resources."
  default     = "thesis"
}

variable "name" {
  type        = string
  description = "The name used for all resources."
  default     = "jupyter"
}

variable "location" {
  type        = string
  description = "The location used for all resources."
  default     = "westeurope"
}

variable "environment" {
  type        = string
  description = "The environment used for all resources."
  default     = "dev"

  validation {
    condition     = var.environment == "dev" || var.environment == "test" || var.environment == "prod"
    error_message = "The environment must be dev, test, or prod."
  }
}

variable "type" {
  type        = string
  description = "The type of virtual machine to deploy (cpu or gpu)."
  default     = "cpu"

  validation {
    condition     = var.type == "cpu" || var.type == "gpu"
    error_message = "The type of virtual machine must be cpu or gpu."
  }
}

variable "username" {
  type        = string
  description = "The username used for the virtual machine."
  default     = "jupyter"
  sensitive   = true
}

variable "password" {
  type        = string
  description = "The password used for the virtual machine."
  default     = null
  sensitive   = true
}
