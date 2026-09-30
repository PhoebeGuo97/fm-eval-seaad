# GPU run on AWS

One spot instance in **us-west-2** (same region as the SEA-AD bucket, so the
~31 GB of per-donor objects stream in-region at no egress cost). It does the
84-donor subsample and the Geneformer extraction, then you copy back one
tarball (~250 MB) and terminate. Everything downstream runs on the Mac.

Instance: `g5.xlarge` (A10G 24 GB, 4 vCPU, 16 GB RAM). `g4dn.xlarge` (T4 16 GB)
works with `BATCH=8` and roughly twice the runtime. Expected: ~30 min data,
~1 to 2 h extraction for the two models.

## 0. Quota (new accounts often have 0 G-family vCPUs)

```bash
# spot G and VT vCPUs; need >= 4
aws service-quotas get-service-quota --region us-west-2 \
  --service-code ec2 --quota-code L-3819A6DF --query 'Quota.Value'
# on-demand G and VT, as a fallback
aws service-quotas get-service-quota --region us-west-2 \
  --service-code ec2 --quota-code L-DB2E81BA --query 'Quota.Value'
```

If 0, request 4 (or 8) in the Service Quotas console. Approval can take hours,
so do this first.

## 0b. Data first, on a CPU instance (no GPU quota needed)

Same launch command as step 1 with three changes: `--instance-type m5.xlarge`
(~$0.19/h), drop the `--instance-market-options` line, and use a plain Ubuntu AMI:

```bash
AMI=$(aws ssm get-parameter --region us-west-2 \
  --name /aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id \
  --query Parameter.Value --output text)
```

Keep `VolumeSize=150`. On the instance:

```bash
git clone https://github.com/PhoebeGuo97/fm-eval-seaad.git && cd fm-eval-seaad
tmux new -s data
bash aws/run_data_cpu.sh 2>&1 | tee work/data_run.log     # ~45 min
```

Back on the Mac, from the repo root:

```bash
mkdir -p data work
scp -i ~/.ssh/<key>.pem ubuntu@<dns>:fm-eval-seaad/data/seaad_mtg_84.h5ad* data/
scp -i ~/.ssh/<key>.pem ubuntu@<dns>:fm-eval-seaad/config/donors.txt config/
scp -i ~/.ssh/<key>.pem ubuntu@<dns>:fm-eval-seaad/work/data_run.log work/
md5 data/seaad_mtg_84.h5ad; cat data/seaad_mtg_84.h5ad.md5     # must match
aws ec2 terminate-instances --region us-west-2 --instance-ids <id>
```

The baselines (02, 04, 05, 06) can run on the Mac now; the Geneformer
embeddings get added when the GPU run finishes. Before running
`aws/run_gpu.sh`, copy the file up so the GPU instance skips the download:

```bash
ssh ... 'mkdir -p fm-eval-seaad/data'
scp -i ~/.ssh/<key>.pem data/seaad_mtg_84.h5ad* ubuntu@<gpu-dns>:fm-eval-seaad/data/
```

## 1. Launch

```bash
REGION=us-west-2
AMI=$(aws ssm get-parameter --region $REGION \
  --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
  --query Parameter.Value --output text)
KEY=<your key pair name in us-west-2>
SG=<security group allowing ssh from your IP>

aws ec2 run-instances --region $REGION --image-id $AMI --instance-type g5.xlarge \
  --key-name $KEY --security-group-ids $SG \
  --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3,DeleteOnTermination=true}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=fm-eval-seaad}]' \
  --query 'Instances[0].InstanceId' --output text
```

## 2. Run

```bash
ssh -i ~/.ssh/<key>.pem ubuntu@<public-dns>
git clone https://github.com/PhoebeGuo97/fm-eval-seaad.git && cd fm-eval-seaad
mkdir -p work
# start inside a detached tmux session so an ssh drop or a closed laptop cannot kill it
tmux new-session -d -s run "bash aws/run_gpu.sh >> work/gpu_run.log 2>&1"
tail -f work/gpu_run.log      # Ctrl-C stops the tail only, not the job
```

The script stops after a 200-cell debug extraction if anything is wrong, before
the long run.

## 3. Copy back and terminate

```bash
# on the Mac, from the repo root
scp -i ~/.ssh/<key>.pem ubuntu@<public-dns>:fm-eval-seaad/work/fm_eval_outputs.tgz .
scp -i ~/.ssh/<key>.pem ubuntu@<public-dns>:fm-eval-seaad/work/gpu_run.log work/
tar xzf fm_eval_outputs.tgz
aws ec2 terminate-instances --region us-west-2 --instance-ids <id>
```

Check the console that the instance and its volume are gone.
