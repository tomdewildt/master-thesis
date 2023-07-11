output "group" {
  value = azurerm_resource_group.rg.name
}

output "address" {
  value = azurerm_linux_virtual_machine.vm.public_ip_address
}

output "private_key" {
  value     = tls_private_key.ssh.private_key_pem
  sensitive = true
}

output "public_key" {
  value     = tls_private_key.ssh.public_key_pem
  sensitive = true
}
