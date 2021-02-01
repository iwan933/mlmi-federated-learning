sudo mkfs.ext4 /dev/nvme1n1

sudo mount -t ext4 /dev/nvme1n1 /mnt/SSDdrive/

sudo mkdir -p /mnt/SSDdrive/temp/run/states
sudo chown -R ubuntu:ubuntu /mnt/SSDdrive/temp/run

ln -s /mnt/SSDdrive/temp/run/states /home/ubuntu/mlmi-federated-learning/run/states
