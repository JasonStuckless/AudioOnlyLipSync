$envName = "audioenv"
$requirements = "requirements.txt"

if (Test-Path ".\$envName\Scripts\Activate.ps1") {
    Write-Host "Virtual environment '$envName' already exists. Activating.."
} else {
    Write-Host "Creating new virtual environment '$envName'.."
    python -m venv $envName

    Write-Host "Activating environment and installing dependencies.."
    & ".\$envName\Scripts\Activate.ps1"
    if (Test-Path $requirements) {
        pip install -r $requirements
    } else {
        Write-Host "No requirements.txt found. Skipping dependency installation."
    }
}

Write-Host "Activating environment..."
& ".\$envName\Scripts\Activate.ps1"