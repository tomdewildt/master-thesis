variable "namespace" {
  type        = string
  description = "The namespace used for all resources."
  default     = "thesis"
}

variable "location" {
  type        = string
  description = "The location used for all resources."
  default     = "west-eu"
}

variable "environment" {
  type        = string
  description = "The environment used for all resources."
  default     = "dev"
}

variable "type" {
  type        = string
  description = "The type of virtual machine to deploy (cpu or gpu)."

  validation {
    condition     = var.type == "cpu" || var.type == "gpu"
    error_message = "The type of virtual machine must be cpu or gpu."
  }
}

variable "username" {
  type        = string
  description = "The username used for the virutal machine."
  default     = "tomdewildt"
}

