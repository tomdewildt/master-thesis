data "template_cloudinit_config" "cic" {
  gzip          = true
  base64_encode = true

  part {
    filename     = "init.cfg"
    content_type = "text/cloud-config"
    content      = <<EOF
    #cloud-config
    package_update: true
    package_upgrade: true
    packages: ['python3-pip']
    EOF
  }
}

resource "azurerm_resource_group" "rg" {
  name     = "rg-${var.namespace}-vm-${var.environment}-${var.location}-01"
  location = "westeurope"
}

resource "azurerm_network_watcher" "nw" {
  name                = "nw-${var.namespace}-vm-${var.environment}-${var.location}-01"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
}

resource "azurerm_virtual_network" "vnet" {
  name                = "vnet-${var.namespace}-vm-${var.environment}-${var.location}-01"
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
}


resource "azurerm_subnet" "snet" {
  name                 = "snet-${var.namespace}-vm-${var.environment}-${var.location}-01"
  resource_group_name  = azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.0.1.0/24"]
}

resource "azurerm_public_ip" "pip" {
  name                = "pip-${var.namespace}-vm-${var.environment}-${var.location}-01"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  allocation_method   = "Dynamic"
}

resource "azurerm_network_security_group" "nsg" {
  name                = "nsg-${var.namespace}-vm-${var.environment}-${var.location}-01"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name

  security_rule {
    name                       = "AllowSSHInBound"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
}

resource "azurerm_network_interface" "nic" {
  name                = "nic-${var.namespace}-vm-${var.environment}-${var.location}-01"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name

  ip_configuration {
    name                          = "nc-${var.namespace}-vm-${var.environment}-${var.location}-01"
    subnet_id                     = azurerm_subnet.snet.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.pip.id
  }
}

resource "azurerm_network_interface_security_group_association" "nna" {
  network_interface_id      = azurerm_network_interface.nic.id
  network_security_group_id = azurerm_network_security_group.nsg.id
}

resource "tls_private_key" "ssh" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "azurerm_linux_virtual_machine" "vm" {
  name                  = "vm-${var.namespace}-vm-${var.environment}-${var.location}-01"
  location              = azurerm_resource_group.rg.location
  resource_group_name   = azurerm_resource_group.rg.name
  network_interface_ids = [azurerm_network_interface.nic.id]
  size                  = var.type == "gpu" ? "Standard_NC6s_v3" : "Standard_B1s"

  os_disk {
    name                 = "osdisk-${var.namespace}-vm-${var.environment}-${var.location}-01"
    caching              = "ReadWrite"
    storage_account_type = "Standard_LRS"
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-focal"
    sku       = "20_04-lts-gen2"
    version   = "latest"
  }

  computer_name                   = "vm"
  user_data                       = data.template_cloudinit_config.cic.rendered
  admin_username                  = var.username
  disable_password_authentication = true

  admin_ssh_key {
    username   = var.username
    public_key = tls_private_key.ssh.public_key_openssh
  }
}

resource "azurerm_virtual_machine_extension" "vme" {
  name                 = "vm"
  virtual_machine_id   = azurerm_linux_virtual_machine.vm.id
  publisher            = "Microsoft.HpcCompute"
  type                 = "NvidiaGpuDriverLinux"
  type_handler_version = "1.6"
}
