#!/bin/bash
PROJECT="project-3dc275fe-b008-41e6-899"
BASE_NAME="aitemplate"
GPU_TYPE="nvidia-tesla-t4"
MACHINE_TYPE="n1-standard-8"
IMAGE="projects/ml-images/global/images/common-cu129-ubuntu-2204-nvidia-580-v20260408"

ZONES=$(gcloud compute accelerator-types list \
  --filter="nvidia-tesla-t4" \
  --format="value(zone.basename())")

for Z in $ZONES; do
  NAME="${BASE_NAME}-$(date +%Y%m%d-%H%M%S)-${Z//-}"
  echo "Checking $Z with instance $NAME ..."

  OUTPUT=$(gcloud compute instances create "$NAME" \
    --project="$PROJECT" \
    --zone="$Z" \
    --machine-type="$MACHINE_TYPE" \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --service-account=848289763513-compute@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/trace.append \
    --accelerator="count=1,type=$GPU_TYPE" \
    --tags=https-server \
    --create-disk="auto-delete=yes,boot=yes,device-name=$NAME,image=$IMAGE,mode=rw,size=100,type=pd-balanced" \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud \
    --reservation-affinity=any \
    --quiet 2>&1)

  STATUS=$?

  if [ $STATUS -eq 0 ]; then
    echo "SUCCESS: GPU available in $Z"
    echo "Instance name: $NAME"
    exit 0
  fi

  echo "FAILED in $Z"
  echo "$OUTPUT"

  gcloud compute instances delete "$NAME" \
    --project="$PROJECT" \
    --zone="$Z" \
    --quiet >/dev/null 2>&1
done

echo "No available zone found."
exit 1