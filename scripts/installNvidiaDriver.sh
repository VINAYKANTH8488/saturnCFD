#!/usr/bin/env bash
# Install the proprietary NVIDIA driver for the local Quadro P600.
#
# This must be run as root (sudo). It does NOT reboot automatically: after it
# finishes, reboot and run scripts/verifyGpu.sh.
#
# Driver choice: the distro recommends 470, but CUDA 12.2 needs a driver newer
# than 525. On Ubuntu 24.04 the 535/550/570 metapackages transition to the 580
# series, which still supports Pascal (GP107) and is new enough for CUDA 12.2.
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "error: run this script with sudo" >&2
  exit 1
fi

if ! lspci | grep -qi 'nvidia'; then
  echo "error: no NVIDIA PCI device found" >&2
  exit 1
fi

echo "NVIDIA device:"
lspci | grep -i 'nvidia'

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y \
  "linux-headers-$(uname -r)" \
  nvidia-driver-580 \
  nvidia-utils-580

# Nouveau and the proprietary driver cannot own the GPU at the same time.
cat >/etc/modprobe.d/blacklist-nouveau.conf <<'EOF'
blacklist nouveau
options nouveau modeset=0
EOF
update-initramfs -u

if command -v mokutil >/dev/null 2>&1; then
  echo "Secure Boot state:"
  mokutil --sb-state || true
fi

cat <<'EOF'

NVIDIA driver installed.

Next steps:
  1. reboot:            sudo reboot
  2. verify the driver: scripts/verifyGpu.sh
  3. build/run CUDA:    source scripts/env.sh
                        cmake --preset cuda-local
                        cmake --build build/cuda-local -j4

The driver package normally blacklists nouveau; the explicit modprobe file
above makes that deterministic. If Secure Boot were enabled, the DKMS module
would need MOK enrollment, but it is disabled on this machine.
EOF
