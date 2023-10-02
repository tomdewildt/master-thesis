data "template_cloudinit_config" "cic" {
  gzip          = false
  base64_encode = true

  part {
    filename     = "init.cfg"
    content_type = "text/cloud-config"
    content      = <<EOF
    #cloud-config
    locale: en_US.UTF-8
    timezone: Europe/Amsterdam
    apt:
      sources:
        deadsnakes-ppa:
          source: ppa:deadsnakes/ppa
    package_update: false
    package_upgrade: false
    packages:
      - curl
      - git
      - python3.10
      - ufw
      - fail2ban
    write_files:
      - path: /etc/sysctl.d/10-configure-ipv6.conf
        permissions: "644"
        content: |
          net.ipv6.conf.all.disable_ipv6=1
          net.ipv6.conf.default.disable_ipv6=1
          net.ipv6.conf.lo.disable_ipv6=1
          net.ipv6.conf.eth0.disable_ipv6=1
      - path: /etc/sysctl.d/10-configure-logging.conf
        permissions: "644"
        content: |
          kernel.printk = 3 4 1 3
      - path: /etc/ssh/sshd_config
        permissions: "644"
        content: |
          Port 22
          SyslogFacility AUTH
          LogLevel INFO
          PermitRootLogin no
          PermitEmptyPasswords no
          ChallengeResponseAuthentication no
          HostbasedAuthentication no
          PasswordAuthentication no
          PubkeyAuthentication yes
          LoginGraceTime 2m
          MaxAuthTries 6
          MaxSessions 10
          StrictModes yes
          UsePAM yes
          UseDNS no
          HostKey /etc/ssh/ssh_host_rsa_key
          HostKey /etc/ssh/ssh_host_ecdsa_key
          HostKey /etc/ssh/ssh_host_ed25519_key
          IgnoreRhosts yes
          AcceptEnv LANG LC_*
          X11Forwarding no
          PrintMotd no
          Subsystem sftp /usr/lib/openssh/sftp-server
      - path: /etc/fail2ban/jail.d/defaults-debian.conf
        permissions: "644"
        content: |
          [DEFAULT]
          ignoreself = true
          bantime  = 1h
          findtime = 30m
          maxretry = 6
      - path: /etc/fail2ban/jail.d/sshd.conf
        permissions: "644"
        content: |
          [sshd]
          enabled  = true
          mode     = aggressive
          port     = ssh
          logpath  = %(sshd_log)s
          backend  = %(sshd_backend)s
          bantime  = 24h
          findtime = 12h
          maxretry = 3
      - path: /usr/lib/systemd/system/jupyter.service
        permissions: "644"
        content: |
          [Unit]
          Description=Jupyter Notebook
          [Service]
          Type=simple
          PIDFile=/run/jupyter.pid
          ExecStart=/home/${var.username}/.local/bin/jupyter-notebook --config=/home/${var.username}/.jupyter/jupyter_notebook_config.py
          User=${var.username}
          Group=${var.username}
          WorkingDirectory=/home/${var.username}
          Environment=PYTHONPATH=/home/${var.username}/master-thesis/src
          Restart=always
          RestartSec=10
          [Install]
          WantedBy=multi-user.target
    runcmd:
      - sed -i "/IPV6=/c\IPV6=no" /etc/default/ufw
      - ufw default deny incoming
      - ufw default allow outgoing
      - ufw allow 22 comment "Allow SSH"
      - ufw allow 8888 comment "Allow Jupyter"
      - ufw --force enable
      - sudo -H -i -u ${var.username} bash -c "curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10"
      - sudo -H -i -u ${var.username} bash -c "pip3.10 install -U jsonschema jupyter zipp"
      - sudo -H -i -u ${var.username} bash -c "jupyter notebook --generate-config"
      - echo "" >> /home/${var.username}/.jupyter/jupyter_notebook_config.py
      - echo "c.ServerApp.ip = '0.0.0.0'" >> /home/${var.username}/.jupyter/jupyter_notebook_config.py
      - echo "c.ServerApp.port = 8888" >> /home/${var.username}/.jupyter/jupyter_notebook_config.py
      - echo "c.ServerApp.port_retries = 0" >> /home/${var.username}/.jupyter/jupyter_notebook_config.py
      - echo "c.ServerApp.open_browser = False" >> /home/${var.username}/.jupyter/jupyter_notebook_config.py
      - echo "c.ServerApp.quit_button = False" >> /home/${var.username}/.jupyter/jupyter_notebook_config.py
      - echo "c.PasswordIdentityProvider.password_required = True" >> /home/${var.username}/.jupyter/jupyter_notebook_config.py
      - echo "c.PasswordIdentityProvider.allow_password_change = True" >> /home/${var.username}/.jupyter/jupyter_notebook_config.py
      - echo "c.PasswordIdentityProvider.hashed_password = 'sha512::${sha512(var.password != null ? var.password : random_password.pw.result)}'" >> /home/${var.username}/.jupyter/jupyter_notebook_config.py
      - systemctl enable jupyter.service
      - systemctl restart jupyter.service
      - sudo -H -i -u ${var.username} bash -c "git clone ${var.repository}"
    EOF
  }
}

resource "random_password" "pw" {
  length  = 16
  lower   = true
  upper   = true
  numeric = true
  special = false
}

resource "azurerm_resource_group" "rg" {
  name     = "${var.namespace}-${var.environment}-${var.location}"
  location = var.location
}

resource "azurerm_network_watcher" "nw" {
  name                = "${var.namespace}-${var.name}-${var.environment}-nw-${var.location}-01"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
}

resource "azurerm_virtual_network" "vnet" {
  name                = "${var.namespace}-${var.name}-${var.environment}-vnet-${var.location}-01"
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
}


resource "azurerm_subnet" "snet" {
  name                 = "${var.namespace}-${var.name}-${var.environment}-snet-${var.location}-01"
  resource_group_name  = azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.0.1.0/24"]
}

resource "azurerm_public_ip" "pip" {
  name                = "${var.namespace}-${var.name}-${var.environment}-pip-${var.location}-01"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  allocation_method   = "Dynamic"
}

resource "azurerm_network_security_group" "nsg" {
  name                = "${var.namespace}-${var.name}-${var.environment}-nsg-${var.location}-01"
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

  security_rule {
    name                       = "AllowJupyterInBound"
    priority                   = 1002
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "8888"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
}

resource "azurerm_network_interface" "nic" {
  name                = "${var.namespace}-${var.name}-${var.environment}-nic-${var.location}-01"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name

  ip_configuration {
    name                          = "${var.namespace}-${var.name}-${var.environment}-nc-${var.location}-01"
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
  name                  = "${var.namespace}-${var.name}-${var.environment}-vm-${var.location}-01"
  location              = azurerm_resource_group.rg.location
  resource_group_name   = azurerm_resource_group.rg.name
  network_interface_ids = [azurerm_network_interface.nic.id]
  size                  = var.type == "gpu" ? "Standard_NC6s_v3" : "Standard_B1s"

  os_disk {
    name                 = "${var.namespace}-${var.name}-${var.environment}-osdisk-${var.location}-01"
    caching              = "ReadWrite"
    storage_account_type = "Standard_LRS"
    disk_size_gb         = 512
  }

  source_image_reference {
    publisher = "Microsoft-DSVM"
    offer     = "Ubuntu-HPC"
    sku       = "2004"
    version   = "latest"
  }

  computer_name                   = "${var.namespace}-${var.name}-${var.environment}-${var.location}-01"
  user_data                       = data.template_cloudinit_config.cic.rendered
  admin_username                  = var.username
  admin_password                  = var.password != null ? var.password : random_password.pw.result
  disable_password_authentication = true

  admin_ssh_key {
    username   = var.username
    public_key = tls_private_key.ssh.public_key_openssh
  }
}
