# 从检查点把识别训练跑完, 然后依次做: LFW 复测 -> 损失函数消融 -> 量化部署对比 -> INT8 模型 LFW 复测。
# 每一步都能重复执行: 已完成的会跳过, 训练中断会从 last.pt 续跑 (核显在系统睡眠后可能丢失设备, 所以重试)。
# 先激活虚拟环境, 再运行:
#   powershell -ExecutionPolicy Bypass -File scripts\finish_experiments.ps1
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONWARNINGS = 'ignore'
$env:PYTHONPATH = (Get-Location).Path
$run = 'r18_arcface_kd'
New-Item -ItemType Directory -Force logs | Out-Null
$status = 'logs\finish_status.txt'

function Note($msg) { "$(Get-Date -Format 'MM-dd HH:mm:ss')  $msg" | Out-File $status -Append -Encoding utf8 }

Note 'start'

for ($i = 0; $i -lt 8 -and -not (Test-Path "runs\$run\final.pt"); $i++) {
    Note "train attempt $i"
    python tools\train_recognition.py --backbone r18 --head arcface --kd-weight 50 --epochs 10 --lr 0.05 `
        --batch-size 128 --name $run --resume "runs\$run\last.pt" *>> 'logs\train_r18_resume.log'
    Note "train exit $LASTEXITCODE"
    Start-Sleep -Seconds 30
}
if (-not (Test-Path "runs\$run\final.pt")) { Note 'FAILED: no final.pt'; exit 1 }

python tools\eval_lfw.py --torch-ckpt "runs\$run\final.pt" --device xpu --tag $run *>> 'logs\eval_lfw_r18.log'
Note "eval_lfw exit $LASTEXITCODE"

for ($i = 0; $i -lt 3; $i++) {
    python tools\ablation_losses.py --device xpu --epochs 5 *>> 'logs\ablation_resume.log'
    Note "ablation exit $LASTEXITCODE"
    if ($LASTEXITCODE -eq 0) { break }
    Start-Sleep -Seconds 30
}

python tools\deploy_benchmark.py --ckpt "runs\$run\final.pt" *>> 'logs\deploy.log'
Note "deploy exit $LASTEXITCODE"

python tools\eval_lfw.py --onnx 'outputs\deploy\r18_int8.onnx' --tag "${run}_int8" *>> 'logs\eval_lfw_int8.log'
Note "eval_lfw int8 exit $LASTEXITCODE"
Note 'done'
