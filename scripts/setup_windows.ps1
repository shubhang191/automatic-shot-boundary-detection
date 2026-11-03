$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $root
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PIP_NO_INPUT = '1'
if (-not $env:PIP_DEFAULT_TIMEOUT) { $env:PIP_DEFAULT_TIMEOUT = '120' }
if (-not $env:PIP_RETRIES) { $env:PIP_RETRIES = '5' }
if (-not $env:PIP_RESUME_RETRIES) { $env:PIP_RESUME_RETRIES = '5' }
$logs = Join-Path $root 'installation_logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$log = Join-Path $logs ('setup-' + (Get-Date -Format 'yyyyMMdd-HHmmss-ffff') + '.log')
Start-Transcript -LiteralPath $log | Out-Null
$script:step = 'Python discovery'
function Probe([string]$exe, [string[]]$prefixArgs, [bool]$venv) {
 $saved = $ErrorActionPreference
 $ErrorActionPreference = 'Continue'
 try {
  $arguments = @($prefixArgs) + @('-B', (Join-Path $PSScriptRoot 'interpreter_probe.py'))
  if ($venv) { $arguments += '--venv' }
  $output = & $exe @arguments 2>$null
  if ($LASTEXITCODE -ne 0) { return $null }
  $result = ($output -join "`n") | ConvertFrom-Json
  if ($result.status -eq 'PASS') { return $result }
 } catch { return $null } finally { $ErrorActionPreference = $saved }
 return $null
}
function Step([string]$name, [string]$exe, [string[]]$arguments) {
 $script:step = $name
 Write-Host "--- $name ---"
 $saved = $ErrorActionPreference
 $ErrorActionPreference = 'Continue'
 & $exe @arguments 2>&1 | ForEach-Object { Write-Host $_ }
 $code = $LASTEXITCODE
 $ErrorActionPreference = $saved
 if ($code -ne 0) { throw "$name failed with exit code $code" }
}
try {
 if ($root.Length -gt 80) { throw 'Application folder path is too long for the pinned Windows dependencies. Extract the ZIP to a shallower writable folder; the full extracted application folder path must be at most 80 characters. No environment repair or dependency installation was attempted.' }
 Write-Host 'Automatic Shot Boundary Detection - Windows setup'
 Write-Host 'Requires local 64-bit Python 3.13. Initial dependency installation needs internet.'
 $venv = Join-Path $root '.venv'
 $python = Join-Path $venv 'Scripts\python.exe'
 $valid = $null
 if (Test-Path -LiteralPath $python) { $valid = Probe $python @() $true }
 $statusFile = Join-Path $venv 'sbd_setup_status.txt'
 if ((Test-Path -LiteralPath $statusFile) -and ((Get-Content -LiteralPath $statusFile -Raw).Trim() -ne 'PASS')) {
  Write-Host 'Previous setup did not finish. Its environment will be preserved and recreated.'
  $valid = $null
 }
 if (-not $valid) {
  $local = $null
  $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
  if ($launcher) { $local = Probe $launcher.Source @('-3.13') $false }
  if (-not $local) {
   foreach ($command in @(Get-Command python.exe -All -ErrorAction SilentlyContinue)) {
    if ($command.Source -match 'WindowsApps') { continue }
    $local = Probe $command.Source @() $false
    if ($local) { break }
   }
  }
  if (-not $local) {
   throw 'No supported local interpreter found. Install Python 3.13 Windows installer (64-bit) from https://www.python.org/downloads/windows/ with the Python launcher, reopen the terminal, and rerun setup.bat. Store aliases and unsupported versions are not used. Existing environments are preserved.'
  }
  Write-Host ('Using current computer Python: ' + $local.executable)
  if (Test-Path -LiteralPath $venv) {
   Write-Host 'Existing .venv is invalid, stale, incomplete, or unsupported. Preserving it before repair.'
   $backup = Join-Path $root ('.venv.invalid-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0,8))
   if ([IO.Path]::GetFullPath($venv) -ne (Join-Path $root '.venv')) { throw 'Unsafe environment path' }
   Move-Item -LiteralPath $venv -Destination $backup
   Write-Host "Environment backup: $backup"
  }
  Step 'Create local environment' $local.executable @('-B','-m','venv',$venv)
  if (-not (Probe $python @() $true)) { throw 'New environment failed interpreter validation' }
 } else { Write-Host 'Reusing validated local Python 3.13 x64 environment.' }
 Set-Content -LiteralPath (Join-Path $venv 'sbd_setup_status.txt') -Value 'INCOMPLETE'
 if (Test-Path -LiteralPath (Join-Path $root 'wheelhouse')) {
  Step 'Install pinned dependencies offline' $python @('-B','-m','pip','install','--no-index','--find-links','wheelhouse','-r','requirements-lock.txt')
 } else {
  Step 'Install pinned dependencies' $python @('-B','-m','pip','install','-r','requirements.txt','-c','requirements-lock.txt')
 }
 Step 'Dependency consistency' $python @('-B','-m','pip','check')
 Step 'FFmpeg provisioning' $python @('-B','scripts/bootstrap_ffmpeg.py')
 if (-not (Test-Path -LiteralPath 'third_party\sbdv2\bin\sbdv.dll')) {
  Step 'Native verifier build' $python @('-B','scripts/build_verifier.py')
 }
 Step 'Installation verification' $python @('-B','verify_installation.py')
 Set-Content -LiteralPath (Join-Path $venv 'sbd_setup_status.txt') -Value 'PASS'
 Write-Host 'Setup complete. Run run.bat. Setup verification is not an inference test.'
 Write-Host "Log: $log"
 Stop-Transcript | Out-Null
 exit 0
} catch {
 Write-Host "SETUP FAILED during: $script:step"
 Write-Host $_.Exception.Message
 Write-Host "Log: $log"
 Write-Host 'Correct the reported problem and rerun setup.bat. Environment backups and model files are not deleted.'
 Stop-Transcript | Out-Null
 exit 1
}
